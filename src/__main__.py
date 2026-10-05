import sys
import fire

from src import indexing
from src.indexing import Index
from src.answer import AnswerGeneration
from src.models.provided_models import UnansweredQuestion
from src.answer.answering import AnswerError
from src.indexing.index import SearchError


class Cli:
    def __init__(self) -> None:
        self.indexer = Index()
        self.generator: AnswerGeneration | None = None

    def index(self, max_chunk_size: int = 2000):
        self.indexer.chunk_text(max_chunk_size)
        self.indexer.index_code(max_chunk_size)
        self.indexer.build_index()
        self.indexer.save()

    def search(self, query: str, k: int = 5):
        self.indexer.load()
        result = self.indexer.search(UnansweredQuestion(question=query), k)
        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        self.indexer.load()
        self.indexer.search_dataset_rag(dataset_path, save_directory, k)

    def answer(self, query: str, k: int = 5) -> None:
        if self.generator is None:
            self.generator = AnswerGeneration(self.indexer)
        try:
            result = self.generator.generate_answer(query, k)
        except (AnswerError, SearchError) as e:
            print(f"Error: {e}")
            return
        print(result.model_dump_json(indent=2))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        if self.generator is None:
            self.generator = AnswerGeneration(self.indexer)
        try:
            self.generator.answer_dataset(
                student_search_results_path, save_directory
            )
        except (AnswerError, SearchError) as e:
            print(f"Error: {e}")

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        pass


def main():
    fire.Fire(Cli())


if __name__ == "__main__":
    sys.exit(main())
