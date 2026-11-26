from pydantic import BaseModel

from src.models.provided_models import MinimalSource


class Chunk(BaseModel):
    """Text of a chunk and its location in the corpus."""

    text: str
    source: MinimalSource
