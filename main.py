import os
import base64 
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL = tf.keras.models.load_model('models/resnet/ResnetModel.h5', custom_objects={"KerasLayer":hub.KerasLayer})

unique_labels = ['bad', 'good']

class ImageInput(BaseModel):
    base64: str

@app.post('/predict/')
async def predict(ImageInput: ImageInput):
    code = ImageInput.base64

    test_filenames = ["./images/" + file_name for file_name in os.listdir("./images/")]

    custom_data = create_img_batches(test_filenames)

    base64 = []

    try:
      for image_path in test_filenames:
        base64_encoded = encode_image_to_base64(image_path)
        print(f"Base64 encoded image '{image_path}':\n", base64_encoded)
        base64.append(base64_encoded)
    except Exception as e:
      print(f"Error: {e}")

    prediction = MODEL.predict(custom_data)

    custom_prediction_labels = [get_predicted_label(prediction[i]) for i in range(len(prediction))]

    return {"prediction": (test_filenames, base64, custom_prediction_labels)}


BATCH_SIZE = 32
IMG_SIZE = 224


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return base64_encoded


def get_predicted_label(prediction_probabilities):
  """
  Recibe un array de probabilidades y matchea la más alta con la label correspondiente.
  """
  return unique_labels[np.argmax(prediction_probabilities)]

def preprocess_image(img_path, img_size=IMG_SIZE):
  """
  Recibe un path de una imagen y convierte a la misma en un Tensor.
  """
  image = tf.io.read_file(img_path)

  image = tf.image.decode_jpeg(image, channels=3) 

  image = tf.image.convert_image_dtype(image, tf.float32)

  image = tf.image.resize(image, size=[img_size, img_size])

  return image


def create_img_batches(X, batch_size=BATCH_SIZE):
  """
  Crea batches a partir de una imagen (X) y su label (y).
  """
  print("Creando batches de test...")
  data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
  data_batch = data.map(preprocess_image).batch(batch_size)
  return data_batch