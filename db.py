from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pathlib import Path
from typing import Optional
from config import SQLITE_PATH
from sqlalchemy import text

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    path = Column(Text, unique=True, nullable=False)
    ts = Column(DateTime, nullable=True)  # timestamp
    gps = Column(String, nullable=True)   # "lat,lon" if available
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    caption = Column(Text, nullable=True) # BLIP caption
    clip_id = Column(String, nullable=True)  # chroma doc id
    tags = Column(Text, nullable=True)  # JSON-encoded list of strings

    faces = relationship("Face", back_populates="image")

class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    person_name = Column(String, nullable=True)  # assigned after recognition
    bbox = Column(String, nullable=True)         # "x1,y1,x2,y2"
    red_ratio = Column(Float, nullable=True)     # heuristic for red shirt in torso crop (0..1)

    image = relationship("Image", back_populates="faces")
    
def _ensure_migrations(engine):
    with engine.connect() as conn:
        info = conn.execute(text("PRAGMA table_info(images)")).fetchall()
        cols = {row[1] for row in info}  # column names
        if "tags" not in cols:
            conn.execute(text("ALTER TABLE images ADD COLUMN tags TEXT"))
            conn.commit()

def get_session():
    engine = create_engine(f"sqlite:///{Path(SQLITE_PATH)}")
    Base.metadata.create_all(engine)
    _ensure_migrations(engine)
    return sessionmaker(bind=engine)()
