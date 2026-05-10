#!/bin/bash

echo "========================================"
echo "🔄 INICIANDO FEEDBACK LOOP DE MLOPS 🔄"
echo "========================================"

echo "[1/3] Lanzando contenedor de reentrenamiento (train-manual)..."
# Levantamos solo el contenedor de entrenamiento. 
# Como comparte el volumen ./models, cuando termine sobrescribirá best_model.keras
# docker compose --build train-manual
docker compose run train-manual

echo "[2/3] Entrenamiento finalizado. El nuevo modelo se ha guardado."

echo "[3/3] Reiniciando la API en producción para cargar los nuevos pesos..."
# Reiniciamos la API para que vuelva a hacer el "load_model()" con el archivo actualizado
docker compose restart api-manual

echo "========================================"
echo "✅ DESPLIEGUE ACTUALIZADO CON ÉXITO ✅"
echo "========================================"