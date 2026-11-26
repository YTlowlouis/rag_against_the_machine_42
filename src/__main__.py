import fire

from src.indexing import Index
from src.answer import AnswerGeneration
from src.models.provided_models import UnansweredQuestion
from src.answer.answering import AnswerError
from src.indexing.index import IndexingError, SearchError
from src.evaluate.evaluate import Evaluate, EvaluateError


class Cli:
    """Command-line interface of the RAG system, exposed with Python Fire."""

    def __init__(self) -> None:
        """Create the indexer and the evaluator; the generator is lazy."""
        self.indexer = Index()
        self.generator: AnswerGeneration | None = None
        self.evaluator = Evaluate()

    def index(
        self, max_chunk_size: int = 2000, path: str = "data/raw"
    ) -> None:
        """Chunk the corpus and build the BM25 index under data/processed.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters (<= 2000).
            path: Directory containing the raw corpus.
        """
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
        """Print the top-k sources for a single query.

        Args:
            query: Question to search for.
            k: Number of sources to return.
        """
        query = str(query)
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
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: Path to a RagDataset JSON file.
            k: Number of sources to return per question.
            save_directory: Directory where the search results are written.
        """
        try:
            self.indexer.load()
            self.indexer.search_dataset_rag(dataset_path, save_directory, k)
        except (SearchError, IndexingError) as e:
            print(e)
            return

    def answer(self, query: str, k: int = 5) -> None:
        """Print an answer to a query, generated from the top-k sources.

        Args:
            query: Question to answer.
            k: Number of sources given to the model as context.
        """
        query = str(query)
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
        """Answer every question of a search results file and save them.

        Args:
            student_search_results_path: Path to a StudentSearchResults file.
            save_directory: Directory where the answers are written.
        """
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
        """Print the recall@k of search results against a ground truth.

        Args:
            student_search_results_path: Path to a StudentSearchResults file.
            dataset_path: Path to a RagDataset file with answered questions.
        """
        try:
            self.evaluator.evaluate(student_search_results_path, dataset_path)
        except EvaluateError as e:
            print(e)
            return


def main() -> None:
    """Run the Fire command-line interface."""
    fire.Fire(Cli())


if __name__ == "__main__":
    main()
