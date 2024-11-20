from sqlalchemy import Column, Integer, String, Float
from app.database import Base

class TrainingMetadata(Base):
    __tablename__ = "training_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    model_type = Column(String)
    accuracy = Column(Float)
