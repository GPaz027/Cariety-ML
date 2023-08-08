from fastapi import FastAPI, WebSocket
import numpy as np
import time 

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict")
async def predict():
    return {}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        time.sleep(3)
        await websocket.send_text("Hello from Python Websocket:")