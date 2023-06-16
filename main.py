# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated

import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

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
   
    model = tf.keras.models.load_model("pnuemonia_sequential1.h5")
    predictions = model.predict(img)
    predictions1 = predictions * 100
    threshold = 0.5
    binary_outputs = (predictions1 > threshold).astype(int)
    predictions= predictions1[0][0]
    result = {
        "img": img,
        "prediction": predictions1[0][0]
    }

    return templates.TemplateResponse("base.html", {"request": request,  "result":predictions, "img":image, "patient_Name":patient_Name,"patient_Age":patient_Age,"patient_Email":patient_Email,"Gender":Gender,"Uploaded_image":image_Type})

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