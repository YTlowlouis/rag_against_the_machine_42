from pydantic import BaseModel

from src.models.provided_models import MinimalSource


class Chunk(BaseModel):
    text: str
    source: MinimalSource
