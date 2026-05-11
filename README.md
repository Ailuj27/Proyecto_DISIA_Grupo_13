# Proyecto_DISIA_Grupo_13

**Nota de Despliegue**: Para garantizar la portabilidad del sistema de reentrenamiento MLOps, se ha externalizado la ruta del sistema de archivos a un archivo .env. Antes de levantar los contenedores en una máquina nueva, el evaluador debe modificar la variable ``PROJECT_ROOT`` en el archivo **.env** con la ruta absoluta donde haya descomprimido el proyecto.

# 1. Despliegue del proyecto
Estando en la carpeta principal del proyecto y con el Docker encendido, abre una CMD y ejecuta el comando ``docker-compose up``. Esto descargagá la imagen de los contenedores creados y levantará el sistema.