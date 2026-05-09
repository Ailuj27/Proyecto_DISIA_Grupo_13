from fastapi import FastAPI, Request
import subprocess
import os

app = FastAPI()

@app.post("/trigger-retrain")
async def trigger_retrain(request: Request):
    try:
        # Intentamos leer el JSON, pero si falla no morimos
        try:
            data = await request.json()
            print(f"🔔 Datos recibidos: {data}")
        except Exception:
            print("🔔 Petición recibida sin cuerpo JSON (usando valores por defecto)")
            data = {}

        print("🚀 Iniciando reentrenamiento automático...")
        
        # Ejecutamos el script de bash
        result = subprocess.run(
            ["bash", "/app/src/feedback_loop.sh"], 
            capture_output=True, 
            text=True
        )
        
        return {
            "status": "success", 
            "message": "Script ejecutado",
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9010)