import uuid
from pydantic import BaseModel, Field


# source file with snippet in it
class MinimalSource(BaseModel):
    """Location of a snippet: a file and a character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question with an identifier."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with its expected sources and answer."""

    sources: list[MinimalSource]
    answer: str


# list of questions provided
class RagDataset(BaseModel):
    """Dataset of answered or unanswered questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


# search result, contains the question and its id and a list of MinimalSource
class MinimalSearchResults(BaseModel):
    """Sources retrieved for a question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Sources retrieved for a question and the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results of a dataset and the k used."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Answers of a dataset and the k used."""

    search_results: list[MinimalAnswer]
    k: int
