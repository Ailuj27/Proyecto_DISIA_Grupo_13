# Proyecto_DISIA_Grupo_13

Guia de funcionamiento del sistema:

## Entrenamiento:
Realizar el comando `docker compose build --no-cache train-manual`
El modelo se almacenará en `models/best_model.keras`

## Inferencia:
Realizar los comandos 
`docker compose build --no-cache api-manual`
`docker compose up api-manual`
Esto arrojará la IP y puerto donde debemos conectarnos (0.0.0.0:8000 por defecto)
Debemos poner esta IP y puerto segido de `/docs` (ej: 0.0.0.0:8000/docs)
Una vez hecho esto, podemos usar el metodo Predict, donde debemos subir una imagen MRI y el modelo nos devolverá la clase, la confianza y un enlace para descargar la imagen después de procesada (explicabilidad del modelo)