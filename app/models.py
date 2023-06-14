from sqlalchemy import Column, Integer, String, LargeBinary  # Add LargeBinary import

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String)
    image_data = Column(LargeBinary)  # Add a new column for storing image data
