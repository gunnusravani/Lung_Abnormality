# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated
from datetime import date, datetime

import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import json
from google.cloud import bigquery, storage
from google.oauth2 import service_account

from fastapi.responses import HTMLResponse
import pandas as pd
import os

key_path = "cloudkarya-internship-32c8df6a1507.json"
bigquery_client = bigquery.Client.from_service_account_json(key_path)
storage_client = storage.Client.from_service_account_json(key_path)

project_id = "cloudkarya-internship"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageData(BaseModel):
    img_file: str
    img_type: str
    patient_id: str
    patient_name: str
    patient_dob: date
    patient_gender: str
    patient_email: str
    pneumonia_prob: float
    tuberculosis_prob: float
    cancer_prob: float
    covid19_prob: float


templates = Jinja2Templates(directory="templates")

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

class FileRequest(BaseModel):
    file_path: str

@app.post("/report")
async def report_file(request: Request,image:Annotated[UploadFile, File(...)],
                       patient_name: Annotated[str,Form(...)],
                       patient_dob: Annotated[str,Form(...)],
                       patient_email: Annotated[str,Form(...)],
                       Gender: Annotated[str,Form(...)],
                       image_type:Annotated[str,Form(...)]
                       ):

    # Process the file path
    # Your logic here
    # form = await request.form()
    # file_field = form["image"]
    # file_path = file_request.file_path
    contents = await image.read()

    encoded_img=base64.b64encode(contents).decode('utf-8')
    

    img = Image.open(io.BytesIO(contents))
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    bucket_name = "monika1"

    # Create a client instance

    # Retrieve the bucket
    bucket1 = storage_client .get_bucket(bucket_name)

    # Retrieve the blob
    pred=[]

    dob = patient_dob
    dob1= patient_dob.replace("-", "")
    patient_id=patient_name+dob1
    filename = image.filename
    bucket = storage_client.get_bucket('lung_abn')
    blob = bucket.blob(f"Lung_Images/{filename}")
    image_path = f'https://storage.cloud.google.com/lung_abn/Lung_Images/{filename}'
    image.file.seek(0)
    blob.upload_from_file(image.file, content_type=image.content_type)
    image.close()
    
    if(image_type=='X-ray'):
            models = ["covid_sequential (1).h5","pnuemonia_sequential1.h5","tuberculosis_functional.h5"]
            for model_file in models:
                blob = bucket1.blob(model_file)
                blob.download_to_filename(model_file)
                model = tf.keras.models.load_model(model_file)
                predictions = model.predict(img)
                pred.append(predictions)
                os.remove(model_file)
            
            pred1 = round(pred[0][0][1]*100,2)
            pred2 = round(pred[1][0][0]*100,2)
            pred4 = round(pred[2][0][0]*100,2)
            pred3=0.0
  

            
            query = f"""
            INSERT INTO `{project_id}.ImageData.ImageDataTable`
            VALUES ('{image_path}', '{image_type}', '{patient_id}', '{patient_name}', 
                    DATE('{dob}'), '{Gender}', '{patient_email}', 
                    {pred1}, {pred2}, {pred3}, {pred4})
            """
            job = bigquery_client.query(query)
            job.result()
            

            return templates.TemplateResponse("base.html", {"request": request,  "result1":pred1, "result2":pred2,"result3":pred3, "result4":pred4, "img":encoded_img, "patient_name":patient_name,"patient_dob":patient_dob,"patient_email":patient_email,"Gender":Gender,"Uploaded_image":image_type})
     
    else:
        models = ["cancer_sequential.h5"]
        for model_file in models:
                blob = bucket1.blob(model_file)
                blob.download_to_filename(model_file)
                model = tf.keras.models.load_model(model_file)
                pred4 = model.predict(img)
                os.remove(model_file)

                pred1 = 0.0
                pred2 = 0.0
                pred3 = round(pred[0][0][0]*100,2)
                pred4 = 0.0


        query = f"""
            INSERT INTO `{project_id}.ImageData.ImageDataTable`
            VALUES ('{image_path}', '{image_Type}', '{patient_id}', '{patient_name}', 
                    DATE('{dob}'), '{Gender}', '{patient_email}', 
                    {pred1}, {pred2}, {pred3}, {pred4})
            """
        job = bigquery_client.query(query)
        job.result()     

        return templates.TemplateResponse("base.html", {"request": request, "result1":pred1,"result2":pred2,"result3":pred3,"result4":pred4, "img":image_path, "patient_name":patient_name,"patient_dob":patient_dob,"patient_email":patient_email,"Gender":Gender,"Uploaded_image":image_type})



@app.post("/ImageData/")
async def create_image_data(item: ImageData):
    query = f"""
    INSERT INTO `{project_id}.ImageData.ImageDataTable`
    VALUES ('{item.img_file}', '{item.img_type}', '{item.patient_id}', '{item.patient_name}', 
            DATE('{item.patient_dob}'), '{item.patient_gender}', '{item.patient_email}', 
            {item.pneumonia_prob}, {item.tuberculosis_prob}, {item.cancer_prob}, {item.covid19_prob})
    """
    job = bigquery_client.query(query)
    job.result()  # Wait for the query to complete

    return {"message": "Data inserted successfully"}


@app.get("/ImageDatas",response_class=HTMLResponse)
async def get_image_data():
   query = f"""
         SELECT  * FROM {project_id}.ImageData.ImageDataTable;
   """
   df = bigquery_client.query(query).to_dataframe()
   # df.head()
   return df.to_html()


@app.get("/ImageData/{id}",response_class=HTMLResponse)
async def get_image_data(id):
   query = f"""
         SELECT  * FROM {project_id}.ImageData.ImageDataTable
         WHERE patient_id = '{id}';
   """
   df = bigquery_client.query(query).to_dataframe()
   # df.head()
   return df.to_html()

@app.post("/getdata")
async def get_data(request: Request,patient_id:Annotated[str,Form(...)]):
   query = f"""
         SELECT  * FROM {project_id}.ImageData.ImageDataTable
         WHERE patient_id ='{patient_id}';
   """
   df = bigquery_client.query(query).to_dataframe()
   print(df.head())
   image_path=df.iloc[0]['img_file']
   pred1=df.iloc[0]['pneumonia_prob']
   pred2=df.iloc[0]['tuberculosis_prob']
   pred4=df.iloc[0]['covid19_prob']
   pred3=df.iloc[0]['cancer_prob']
   patient_name=df.iloc[0]['patient_name']
   patient_email=df.iloc[0]['patient_email']
   patient_dob=df.iloc[0]['patient_dob']
   Gender=df.iloc[0]['patient_gender']
   image_type=df.iloc[0]['img_type']

   return templates.TemplateResponse("base.html", {"request": request, "result1":pred1,"result2":pred2,"result3":pred3, "result4":pred4, "img":image_path , "patient_name":patient_name,"patient_dob":patient_dob,"patient_email":patient_email,"Gender":Gender,"Uploaded_image":image_type})
   
   # df.head()
   #    return df.to_html()                   
                
        







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