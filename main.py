# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated
import h5py

import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

from google.cloud import bigquery, storage
from google.oauth2 import service_account

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

class FileRequest(BaseModel):
    file_path: str

@app.post("/report")
async def report_file(request: Request,image:Annotated[UploadFile, File(...)],
                       patient_Name: Annotated[str,Form(...)],
                       patient_Age: Annotated[str,Form(...)],
                       patient_Email: Annotated[str,Form(...)],
                       Gender: Annotated[str,Form(...)],
                       image_Type:Annotated[str,Form(...)]
                       ):
    # Process the file path
    # Your logic here
    # form = await request.form()
    # file_field = form["image"]
    # file_path = file_request.file_path
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    bucket_name = "monika1"
    models = ["covid_sequential (1).h5","pnuemonia_sequential1.h5","tuberculosis_functional.h5"]
    # Create a client instance
    key_path = "cloudkarya-internship-1c013aa63f5f.json"
    client = storage.Client.from_service_account_json(key_path)


    # Retrieve the bucket
    bucket = client.get_bucket(bucket_name)

    # Retrieve the blob
    pred=[]

    for model_file in models:
        blob = bucket.blob(model_file)
        blob.download_to_filename(model_file)

        model = tf.keras.models.load_model(model_file)
        predictions = model.predict(img)
        pred.append(predictions)

        os.remove(model_file)

    predictions1 = pred[0]
    predictions2 = pred[1]
    predictions3 = pred[2]
    predi1 = predictions1 * 100
    predi2 = predictions2 * 100
    predi3 = predictions3 * 100
    threshold = 0.5
    binary_outputs = (predictions1 > threshold).astype(int)
    pred1 = predi1[0][0]
    pred2 = predi2[0][0]
    pred3 = predi3[0][0]

    return templates.TemplateResponse("base.html", {"request": request,  "result1":pred1,"result2":pred2,"result2":pred3, "img":image, "patient_Name":patient_Name,"patient_Age":patient_Age,"patient_Email":patient_Email,"Gender":Gender,"Uploaded_image":image_Type})

# # if __name__ == '__dynamic__':
# #    uvicorn.run(app, host='0.0.0.0', port=8000)

# demo = Image.open("/workspace/Pheonix_Squadron/PDR-OS-LRG.jpg")
# plt.imshow(demo)

"""@app.post("/images")
async def create_image(image: UploadFile = File(...)):
    db = SessionLocal()
    image_data = image.file.read()

    image_obj = schemas.ImageCreate(
        filename=image.filename,
        filepath=image.filepath,
        image_data=image_data
    )

    db_image = models.Image(**image_obj.dict())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    return {"message": "Image stored in the database."}"""


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     img = Image.open(io.BytesIO(contents))
#     img = img.resize((150, 150))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)

#     model = tf.keras.models.load_model("/workspace/X-ray/pnuemonia_sequential1.h5")
#     predictions = model.predict(img)
#     predictions1 = predictions * 100
#     threshold = 0.5
#     binary_outputs = (predictions1 > threshold).astype(int)

#     result = {
#         "img": img,
#         "prediction": predictions1[0][0]
#     }

#     return templates.TemplateResponse("report.html", {"request": file, "result": result})