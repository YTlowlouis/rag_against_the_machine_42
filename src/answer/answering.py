from pathlib import Path
from pydantic import ValidationError
from tqdm import tqdm

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from src.indexing import Index
from src.models.provided_models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    StudentSearchResults,
)


class AnswerError(Exception):
    pass


class AnswerGeneration:
    def __init__(self, indexer: Index) -> None:
        self.indexer = indexer
        self.model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self.max_tokens = 500

    def generate_answer(self, query: str, k: int) -> MinimalAnswer:
        if k <= 0 or not query.strip():
            raise AnswerError("Invalid query or k <= 0")

        self.indexer.load()
        result = self.indexer.search(UnansweredQuestion(question=query), k)
        return self.answer_search_result(result)

    def answer_search_result(
        self, search_result: MinimalSearchResults
    ) -> MinimalAnswer:

        self._load_model()

        if self.tokenizer is None:
            raise AnswerError("Tokenizer is not loaded, run _load_model")
        if self.model is None:
            raise AnswerError("Model is not loaded, run _load_model")

        sources = self._build_context(search_result.retrieved_sources)

        for source in sources:
            ids = self.tokenizer.encode(str(source["text"]))
            if len(ids) > self.max_tokens:
                source["text"] = self.tokenizer.decode(ids[: self.max_tokens])

        prompt = self._build_prompt(search_result.question, sources)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=False
            )

        prompt_length = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).strip()
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        return MinimalAnswer(**search_result.model_dump(), answer=answer)

    def _build_context(
        self, retrieved_sources: list[MinimalSource]
    ) -> list[dict[str, str | int]]:
        sources: list[dict[str, str | int]] = []
        for n, src in enumerate(retrieved_sources):
            try:
                content = Path(src.file_path).read_text(encoding="utf-8")
            except OSError:
                print(f"Error when opening file: {src.file_path}")
                continue
            except UnicodeDecodeError:
                print(f"Couldn't decode: {src.file_path}")
                continue

            start = src.first_character_index
            end = src.last_character_index
            sources.append(
                {"id": n, "file": src.file_path, "text": content[start:end]}
            )
        return sources

    def _build_prompt(
        self, query: str, sources: list[dict[str, str | int]]
    ) -> str:
        if self.tokenizer is None:
            raise AnswerError("Tokenizer is not loaded, run _load_model")

        context = "\n\n".join(
            f"[Source {source['id']}: {source['file']}]\n{source['text']}"
            for source in sources
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant answering questions about the vLLM "
                    "codebase. Answer only using the provided context. "
                    "If the context does not contain the answer, say that "
                    "you don't know. Be concise and precise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _load_model(self) -> None:
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto"
            )
            self.model.eval()

    def answer_dataset(self, search_result: str, save_directory: str):
        try:
            dataset = StudentSearchResults.model_validate_json(
                Path(search_result).read_text(encoding="utf-8")
            )
        except ValidationError as e:
            raise AnswerError(
                f"Invalid dataset format in {search_result}"
            ) from e
        except PermissionError as e:
            raise AnswerError(
                f"Invalid permissions for {search_result}"
            ) from e
        except OSError as e:
            raise AnswerError(f"Error when opening {search_result}") from e

        answers = [
            self.answer_search_result(res)
            for res in tqdm(dataset.search_results, desc="answering")
        ]
        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=dataset.k
        )
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / Path(search_result).name).write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
