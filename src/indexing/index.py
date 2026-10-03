from json.decoder import JSONDecodeError
import bm25s
import json
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


class IndexingError(Exception):
    pass


class SearchError(Exception):
    pass


class Index:
    def __init__(self):
        self.corpus: list[Document] = []
        self.raw_path = Path("data/raw")
        self.retriever = None
        self.splitter: RecursiveCharacterTextSplitter | None = None

    def chunk_text(self, chunk_size: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),
            add_start_index=True,
        )
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
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),
            add_start_index=True,
        )
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

    def build_index(self) -> None:
        if not self.corpus:
            raise IndexingError("Empty corpus, chunk text before")
        texts = [doc.page_content for doc in self.corpus]

        tokens = bm25s.tokenize(texts, stopwords="en")

        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

    def save(self, path: str = "data/processed") -> None:
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
        self.retriever.save(path, corpus=corpus)

    def load(self, path: str = "data/processed") -> None:
        self.retriever = bm25s.BM25.load(path, load_corpus=True)

    def search(
        self, question: UnansweredQuestion, k: int = 5
    ) -> MinimalSearchResults:
        return self.search_dataset([question], k).search_results[0]

    def search_dataset(
        self, questions: list[UnansweredQuestion], k: int = 5
    ) -> StudentSearchResults:
        if self.retriever is None:
            raise SearchError("retriever is not loaded, run index")
        query_tokens = bm25s.tokenize(
            [q.question for q in questions], stopwords="en"
        )
        results, _ = self.retriever.retrieve(query_tokens, k=k)

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
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / Path(dataset_path).name
        out_file.write_text(
            results.model_dump_json(indent=2), encoding="utf-8"
        )
