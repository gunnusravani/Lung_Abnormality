from typing import Optional
from pydantic import BaseModel, HttpUrl

class ImageCreate(BaseModel):
    filename: str
    filepath: str
    image_data: bytes  # Add image_data field
