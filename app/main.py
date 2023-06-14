# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow
# from tensorflow import keras
# from PIL import Image
# import matplotlib.pyplot as plt

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.post("/base")
async def base(request: Request, file: UploadFile = File()):
    data = file.file.read()
    file.file.close()

    # encoding the image
    encoded_image = base64.b64encode(data).decode("utf-8")

    #Demo Model Linking
    # probability = model.predict(img2Array(image))

    return templates.TemplateResponse("report.html", {"request": request,  "img": encoded_image})

# # if __name__ == '__dynamic__':
# #    uvicorn.run(app, host='0.0.0.0', port=8000)

# demo = Image.open("/workspace/Pheonix_Squadron/PDR-OS-LRG.jpg")
# plt.imshow(demo)

@app.post("/images")
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

    return {"message": "Image stored in the database."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model("/workspace/X-ray/pnuemonia_sequential1.h5")
    predictions = model.predict(img)
    predictions1 = predictions * 100
    threshold = 0.5
    binary_outputs = (predictions1 > threshold).astype(int)

    result = {
        "img": img,
        "prediction": predictions1[0][0]
    }

    return templates.TemplateResponse("report.html", {"request": file, "result": result})