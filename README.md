# Cariety - API

## Requerimientos

* Install FastAPI: `pip install fastapi`
* Install Uvicorn: `pip install uvicorn`
* Install NumPy: `pip install numpy`
* Install TensorFlow: `pip install tensorflow`
* Install TensorFlow Hub: `pip install tensorflow_hub`
* Correr `pip install -r requirements.txt` para instalar otras dependencias fundamentales.

## Utilización

1. Clonar repositorio.
2. En la raíz del projecto, correr `uvicorn main:app --reload` para cargar la API.
3. Probar mandando el base64 de una imagen en el body de una solicitud HTTP como body a `localhost:8080/predict/`.
4. En la raíz del proyecto se encuentra el Jupyter Notebook donde se realizó el entrenamiento del modelo predictivo.

**Nota:** Es importante tener el frontend encendido para visualizar los resultados en el panel. Sin embargo, puede verse la respuesta de la predicción en un JSON que devuelve la API sin necesidad de acceder al panel.

* Autores: Nicolás Cousiño y Gonzalo Paz.
