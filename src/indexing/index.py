import bm25s
import Stemmer
from pydantic import ValidationError
from pathlib import Path
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.models.provided_models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
    RagDataset,
)


MAX_CHUNK_SIZE = 2000


class IndexingError(Exception):
    """Raised when the corpus cannot be chunked, indexed or saved."""


class SearchError(Exception):
    """Raised when a search cannot be performed."""


class Index:
    """Chunk a corpus, build a BM25 index over it and search it."""

    def __init__(self) -> None:
        """Initialise an empty corpus and an unloaded retriever."""
        self.corpus: list[Document] = []
        self.raw_path: Path
        self.retriever: bm25s.BM25 | None = None
        self.splitter: RecursiveCharacterTextSplitter | None = None
        self.stemmer = Stemmer.Stemmer("english")

    def _set_path(self, path: str) -> None:
        """Set the directory of the raw corpus.

        Args:
            path: Directory containing the files to index.
        """
        self.raw_path = Path(path)

    def _check_chunk_size(self, chunk_size: int) -> None:
        """Check that the chunk size is accepted by the evaluator.

        Args:
            chunk_size: Maximum size of a chunk, in characters.

        Raises:
            IndexingError: If chunk_size is not an integer in [1, 2000].
        """
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or not 0 < chunk_size <= MAX_CHUNK_SIZE
        ):
            raise IndexingError(
                "max_chunk_size must be an integer between 1 and "
                f"{MAX_CHUNK_SIZE}, got {chunk_size}"
            )

    def chunk_text(self, chunk_size: int) -> None:
        """Split the Markdown and text files of the corpus into chunks.

        Args:
            chunk_size: Maximum size of a chunk, in characters.

        Raises:
            IndexingError: If chunk_size is invalid.
        """
        self._check_chunk_size(chunk_size)
        try:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.2),
                add_start_index=True,
            )
        except (ValueError, TypeError) as e:
            raise IndexingError(
                f"Invalid max_chunk_size: {chunk_size}"
            ) from e
        extensions = {".md", ".txt"}
        files = [
            file
            for file in self.raw_path.rglob("*")
            if file.is_file() and file.suffix in extensions
        ]

        for file in tqdm(files, unit="file", desc="chunking text"):
            try:
                text = file.read_text(encoding="utf-8")
            except PermissionError:
                tqdm.write(f"Invalid permissions: {file}")
                continue
            except UnicodeDecodeError:
                tqdm.write(f"Couldn't decode: {file}")
                continue

            docs = self.splitter.create_documents(
                [text], metadatas=[{"source": str(file)}]
            )
            for doc in docs:
                start = doc.metadata["start_index"]
                doc.metadata["end_index"] = start + len(doc.page_content)
            self.corpus.extend(docs)

    def index_code(self, chunk_size: int) -> None:
        """Split the Python files of the corpus into chunks.

        Args:
            chunk_size: Maximum size of a chunk, in characters.

        Raises:
            IndexingError: If chunk_size is invalid.
        """
        self._check_chunk_size(chunk_size)
        try:
            self.splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.2),
                add_start_index=True,
            )
        except (ValueError, TypeError) as e:
            raise IndexingError(
                f"Invalid max_chunk_size: {chunk_size}"
            ) from e
        extensions = {".py"}
        files = [
            file
            for file in self.raw_path.rglob("*")
            if file.is_file() and file.suffix in extensions
        ]

        for file in tqdm(files, unit="file", desc="chunking code"):
            try:
                code = file.read_text(encoding="utf-8")
            except PermissionError:
                tqdm.write(f"Invalid permissions: {file}")
                continue
            except UnicodeDecodeError:
                tqdm.write(f"Couldn't decode: {file}")
                continue

            docs = self.splitter.create_documents(
                [code], metadatas=[{"source": str(file)}]
            )

            for doc in docs:
                start = doc.metadata["start_index"]
                doc.metadata["end_index"] = start + len(doc.page_content)
            self.corpus.extend(docs)

    def check_corpus(self) -> None:
        """Check that the corpus path exists and is a directory.

        Raises:
            IndexingError: If the path is missing or is not a directory.
        """
        if not self.raw_path.exists():
            raise IndexingError(f"Corpus not found at {self.raw_path}")
        if not self.raw_path.is_dir():
            raise IndexingError("Path is not a directory")

    def build_index(self) -> None:
        """Tokenize the chunks and build the BM25 index.

        Raises:
            IndexingError: If the corpus is empty.
        """
        if not self.corpus:
            raise IndexingError("Empty corpus, chunk text before")
        texts = [doc.page_content for doc in self.corpus]

        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer)

        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

    def save(self, path: str = "data/processed") -> None:
        """Save the index and its corpus to disk.

        Args:
            path: Directory where the index is saved.

        Raises:
            IndexingError: If the index is not built or cannot be written.
        """
        if self.retriever is None:
            raise IndexingError(
                "Retriever is not initialised, run build_index"
            )

        corpus = [
            {
                "text": doc.page_content,
                **MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["end_index"],
                ).model_dump(),
            }
            for doc in self.corpus
        ]
        try:
            self.retriever.save(path, corpus=corpus)
        except OSError as e:
            raise IndexingError(f"Couldn't save index to {path}") from e

    def load(self, path: str = "data/processed") -> None:
        """Load a saved index and its corpus.

        Args:
            path: Directory where the index was saved.

        Raises:
            IndexingError: If the index cannot be loaded.
        """
        try:
            self.retriever = bm25s.BM25.load(path, load_corpus=True)
        except (OSError, ValueError) as e:
            raise IndexingError(
                f"Couldn't load index from {path}, run index first"
            ) from e

    def search(
        self, question: UnansweredQuestion, k: int = 5
    ) -> MinimalSearchResults:
        """Return the top-k sources for a single question.

        Args:
            question: Question to search for.
            k: Number of sources to return.

        Returns:
            The search results of the question.

        Raises:
            SearchError: If the query is empty or the search fails.
        """
        if not question.question.lstrip():
            raise SearchError("Empty query")
        return self.search_dataset([question], k).search_results[0]

    def search_dataset(
        self, questions: list[UnansweredQuestion], k: int = 5
    ) -> StudentSearchResults:
        """Return the top-k sources for each question, in a single batch.

        Args:
            questions: Questions to search for.
            k: Number of sources to return per question.

        Returns:
            The search results of every question.

        Raises:
            SearchError: If the index is not loaded, there is no question,
                k is invalid or the search fails.
        """
        if self.retriever is None:
            raise SearchError("retriever is not loaded, run index")
        if not questions:
            raise SearchError("No question to search")
        if not isinstance(k, int) or k <= 0:
            raise SearchError(f"k must be a positive integer, got {k}")
        query_tokens = bm25s.tokenize(
            [q.question for q in questions],
            stopwords="en",
            stemmer=self.stemmer,
        )
        try:
            results, _ = self.retriever.retrieve(query_tokens, k=k)
        except (ValueError, TypeError) as e:
            raise SearchError(f"Search failed with k={k}: {e}") from e

        search_results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=doc["file_path"],
                        first_character_index=doc["first_character_index"],
                        last_character_index=doc["last_character_index"],
                    )
                    for doc in docs
                ],
            )
            for question, docs in zip(questions, results)
        ]
        return StudentSearchResults(search_results=search_results, k=k)

    def search_dataset_rag(
        self, dataset_path: str, save_directory: str, k: int = 5
    ) -> None:
        """Search every question of a dataset file and save the results.

        Args:
            dataset_path: Path to a RagDataset JSON file.
            save_directory: Directory where the results are written.
            k: Number of sources to return per question.

        Raises:
            SearchError: If the dataset cannot be read or the results cannot
                be written.
        """
        try:
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
        except ValidationError as e:
            raise SearchError(
                f"Invalid dataset format in {dataset_path}"
            ) from e
        except PermissionError as e:
            raise SearchError(f"Invalid permissions for {dataset_path}") from e
        except OSError as e:
            raise SearchError(f"Error when opening {dataset_path}") from e

        self.load()
        results = self.search_dataset(dataset.rag_questions, k)

        out_dir = Path(save_directory)
        out_file = out_dir / Path(dataset_path).name
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file.write_text(
                results.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as e:
            raise SearchError(f"Couldn't write to {out_file}") from e
        print(f"Saved student_search_results to {out_file}")
