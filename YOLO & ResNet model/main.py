import sys
import cv2
import torch
import numpy as np
from numpy import random

sys.path.append('./model')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

from fastapi import FastAPI

from pydantic import BaseModel
import base64
import PIL
from io import BytesIO

import requests

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import base64

from fastapi import FastAPI
from pydantic import BaseModel

# /-------------------------------------------------------------- Definicion de Clases y Constantes --------------------------------------------------------------/

ROOT_DIR = './'
SAVE_PATH = '/saved_images_temp'
BATCH_SIZE = 32
IMG_SIZE = 224

source_image_path = 'good1.jpeg'
custom_data_path = 'custom_data.yaml'
weights_path = 'best.pt'

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


unique_labels = ['bad', 'good']

class UserInput(BaseModel):
    user_input: str

# /-------------------------------------------------------------- Metodos para YOLO v7 --------------------------------------------------------------/

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt  = {

    "weights": weights_path, # Path to weights file default weights are for nano model
    "yaml"   : custom_data_path,
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
}

def save_image(base_64_image, relative_save_path):
    image_path = ROOT_DIR + relative_save_path + "/Resultado.jpg"
    im = PIL.Image.open(BytesIO(base64.b64decode(base_64_image)))
    im.save(ROOT_DIR + relative_save_path + "/Resultado.jpg")
    return image_path

def get_image_position_if_valid(box):
    """El parametro model_prediction es un ImagesDetectionPrediction que sale directamente del model.predict()"""
    IMG_MEDIUM = 512/2
    
    medium_point = (box[0]+box[2])/2
    if (True  or (medium_point < IMG_MEDIUM + 10 and medium_point > IMG_MEDIUM - 10)):
        #string =Restnet.classify
        return box
    else:
        return []

def predict_image(path, model):
    
    weights, imgsz = opt['weights'], opt['img-size']
    device = select_device(opt['device'])
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    half = device.type != 'cpu'
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    img0 = cv2.imread(path)
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment= False)[0]
    
    # Apply NMS
    classes = None
    if opt['classes']:
        classes = []
        for class_name in opt['classes']:
            classes.append(opt['classes'].index(class_name))
            
    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
    t2 = time_synchronized()
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
            for *xyxy, conf, cls in reversed(det):
                print(*xyxy)
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            return [x1, y1, x2, y2]

def load_model():
    with torch.no_grad():
      weights, imgsz = opt['weights'], opt['img-size']
      set_logging()
      device = select_device(opt['device'])
      half = device.type != 'cpu'
      model = attempt_load(weights, map_location=device)  # load FP32 model
      stride = int(model.stride.max())  # model stride
      imgsz = check_img_size(imgsz, s=stride)  # check img_size
      if half:
        model.half()

      names = model.module.names if hasattr(model, 'module') else model.names
      colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
      if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
      return model

# /-------------------------------------------------------------- Metodos Auxiliares Clasificacion --------------------------------------------------------------/

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
  Si se trata de imágenes de entrenamiento, randomiza a la misma. No lo hará si se trata de validación.
  Acepta imágenes del dataset de test (no hay labels).
  """
  print("Creando batches de test...")
  data = tf.data.Dataset.from_tensor_slices(tf.constant(X)) # Constant devuelve un Tensor constante. from_tensor_slices crea un Dataset cuyos elementos son slices del tensor pasado como parámetro.
  data_batch = data.map(preprocess_image).batch(batch_size)
  return data_batch

# /-------------------------------------------------------------- Metodos Clasificacion --------------------------------------------------------------/

def classify_image(path):
    custom_data = create_img_batches(path)
    
    prediction = MODEL.predict(custom_data)

    custom_prediction_labels = [get_predicted_label(prediction[i]) for i in range(len(prediction))]

    return custom_prediction_labels

# /-------------------------------------------------------------- Carga de YOLO v7 --------------------------------------------------------------/

model = load_model()

# /-------------------------------------------------------------- Carga de ResNet --------------------------------------------------------------/

MODEL = tf.keras.models.load_model('ResnetModel.h5', custom_objects={"KerasLayer":hub.KerasLayer})

# /-------------------------------------------------------------- Endpoints FastAPI --------------------------------------------------------------/

def send_image_to_microbakend(url, json):
    return requests.post(url, json= json)

app = FastAPI()

class Image(BaseModel):
    base_64: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/image")
async def upload_image(image: Image):
    base_64 = image.base_64
    image_save_base_64 = base_64.split(",", 1)[1]
    path = save_image(image_save_base_64, SAVE_PATH)


    out = predict_image(path, model)
    box = get_image_position_if_valid(out)
    print("Prediccion completa: ")
    print(box)
    if box: #not y un array es preguntar si el array esta vacio
        print("Aca llega")
        print(path)
        prediction = classify_image([path])
        print(prediction)
        print("Llamando a la API")
        MICROBACKEND_URL = 'http://localhost:3000/image'
        json = {"base_64": base_64, "box": box, "prediction": prediction[0]}

        status_code = send_image_to_microbakend(MICROBACKEND_URL, json)
        print(status_code)

        return { "base_64": base_64, "box": box, "prediction": prediction[0]}
        #out.save( ROOT_DIR)
    else:
        return "Imagen Invalida"




