import sys
import fire

from src import indexing
from src.indexing import Index
from src.answer import AnswerGeneration
from src.models.provided_models import UnansweredQuestion
from src.answer.answering import AnswerError
from src.indexing.index import IndexingError, SearchError
from src.evaluate.evaluate import Evaluate, EvaluateError


class Cli:
    def __init__(self) -> None:
        self.indexer = Index()
        self.generator: AnswerGeneration | None = None
        self.evaluator = Evaluate()

    def index(
        self, max_chunk_size: int = 2000, path: str = "data/raw"
    ) -> None:
        try:
            self.indexer._set_path(path)
            self.indexer.check_corpus()
            self.indexer.chunk_text(max_chunk_size)
            self.indexer.index_code(max_chunk_size)
            self.indexer.build_index()
            self.indexer.save()
        except IndexingError as e:
            print(e)
            return

    def search(self, query: str, k: int = 5) -> None:
        try:
            self.indexer.load()
            result = self.indexer.search(UnansweredQuestion(question=query), k)
            print(result.model_dump_json(indent=2))
        except (IndexingError, SearchError) as e:
            print(e)
            return

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        try:
            self.indexer.load()
            self.indexer.search_dataset_rag(dataset_path, save_directory, k)
        except (SearchError, IndexingError) as e:
            print(e)
            return

    def answer(self, query: str, k: int = 5) -> None:
        if self.generator is None:
            self.generator = AnswerGeneration(self.indexer)
        try:
            result = self.generator.generate_answer(query, k)
        except (AnswerError, SearchError, IndexingError) as e:
            print(f"Error: {e}")
            return
        print(result.model_dump_json(indent=2))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        if self.generator is None:
            self.generator = AnswerGeneration(self.indexer)
        try:
            self.generator.answer_dataset(
                student_search_results_path, save_directory
            )
        except (AnswerError, SearchError) as e:
            print(f"Error: {e}")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        try:
            self.evaluator.evaluate(student_search_results_path, dataset_path)
        except EvaluateError as e:
            print(e)
            return


def main():
    fire.Fire(Cli())


if __name__ == "__main__":
    sys.exit(main())
