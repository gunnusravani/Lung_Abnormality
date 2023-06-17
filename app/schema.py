from typing import Optional
from pydantic import BaseModel, HttpUrl


class ImageData(BaseModel):
    img_file: str
    patient_id: str
    patient_name: str
    patient_dob: date
    patient_gender: str
    patient_email: EmailStr
    pneumonia_prob: float
    tuberculosis_prob: float
    cancer_prob: float
    covid19_prob: float
    
      # Add image_data field
