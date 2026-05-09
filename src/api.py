import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

from infer import TumorPredictor


app = FastAPI(
    title="API de Clasificación de Tumores Cerebrales",
    description="Recibe una imagen MRI, devuelve la predicción y genera un heatmap",
    version="1.0.0"
)

# Esto captura automáticamente: total de peticiones, peticiones en curso, tamaño de la respuesta,
# códigos de estado (200, 400, 500) y latencias operativas.
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)
# 'instrument' envuelve la app para espiarla, y 'expose' crea la ruta /metrics
instrumentator.instrument(app).expose(app)
# ==========================================================
# MÉTRICAS CUSTOM DEL MODELO PARA GRAFANA
# ==========================================================
PREDICCIONES_TOTALES = Counter(
    "modelo_predicciones_totales",
    "Número total de predicciones separadas por clase",
    ["clase_predicha"]
)

CONFIANZA_MODELO = Histogram(
    "modelo_confianza",
    "Distribución de la confianza del modelo"
)
# ==========================================================



HEATMAPS_DIR = os.getenv("HEATMAPS_DIR", "/app/generated/heatmaps")

predictor = TumorPredictor(
    model_path=os.getenv("MODEL_PATH", "/app/models/best_model.keras"),
    classes_path=os.getenv("CLASSES_PATH", "/app/models/class_names.json"),
    heatmaps_dir=HEATMAPS_DIR,
)

Path(HEATMAPS_DIR).mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "API activa",
        "docs_url": "/docs",
        "predict_endpoint": "/predict"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Recibe una imagen MRI y devuelve:
    - clase predicha
    - confianza
    - probabilidades
    - URL para descargar el heatmap PNG
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Debes subir un archivo de imagen válido."
            )

        image_bytes = await file.read()
        # --- Inicio de la Inferencia ---
        start_time = time.time()
        result = predictor.predict_from_bytes(image_bytes)
        inference_time = time.time() - start_time

        # --- Registrar Métricas Custom para Grafana ---
        clase = str(result["predicted_class"])
        confianza = float(result["confidence"])
        
        PREDICCIONES_TOTALES.labels(clase_predicha=clase).inc()
        CONFIANZA_MODELO.observe(confianza)

        base_url = str(request.base_url).rstrip("/")
        heatmap_filename = result["heatmap_filename"]
        heatmap_url = f"{base_url}/downloads/{heatmap_filename}"

        return {
            "status": "success",
            "filename": file.filename,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "heatmap_filename": heatmap_filename,
            "heatmap_download_url": heatmap_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia: {str(e)}"
        )


@app.get("/downloads/{filename}")
def download_heatmap(filename: str):
    """
    Devuelve el archivo PNG del heatmap para descargar.
    """
    file_path = Path(HEATMAPS_DIR) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No se encontró el heatmap solicitado.")

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename,
    )