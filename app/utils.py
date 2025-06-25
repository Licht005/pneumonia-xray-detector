
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

# Set up
Base = declarative_base()
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pneumonia_results.db")
engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    est_type = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def save_prediction(filename, prediction, confidence, est_type):
    session = Session()
    new_record = Prediction(
        filename=filename,
        prediction=prediction,
        confidence=confidence,
        est_type=est_type
    )
    session.add(new_record)
    session.commit()
    session.close()

def get_latest_predictions(n=5):
    session = Session()
    results = session.query(Prediction).order_by(Prediction.timestamp.desc()).limit(n).all()
    session.close()
    return results
