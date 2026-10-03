from pathlib import Path

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
    UnansweredQuestion,
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

        self._load_model()

        self.indexer.load()
        result = self.indexer.search(UnansweredQuestion(question=query), k)

        sources = self._build_context(result.retrieved_sources)

        if self.tokenizer is None:
            raise AnswerError("Tokenizer is not loaded, run _load_model")
        if self.model is None:
            raise AnswerError("Model is not loaded, run _load_model")

        for source in sources:
            ids = self.tokenizer.encode(str(source["text"]))
            if len(ids) > self.max_tokens:
                source["text"] = self.tokenizer.decode(ids[: self.max_tokens])

        prompt = self._build_prompt(query, sources)

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

        return MinimalAnswer(**result.model_dump(), answer=answer)

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
