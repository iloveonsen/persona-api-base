from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DB_USERNAME = os.environ.get("DB_USERNAME") 
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_URL = f"postgresql+asyncpg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"


class Config(BaseSettings):
    db_url: str = Field(default=DB_URL, env="DB_URL")
    model_path: Path = Field(default=Path("models"), env="MODEL_PATH")


config = Config()
    