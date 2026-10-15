from pathlib import Path
from pydantic import ValidationError
from src.models.provided_models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
)


class EvaluateError(Exception):
    pass


class Evaluate:
    def __init__(self) -> None:
        pass

    def _check_overlap(
        self, source_1: MinimalSource, source_2: MinimalSource
    ) -> bool:
        start, end = (
            source_1.first_character_index,
            source_1.last_character_index,
        )
        start1, end1 = (
            source_2.first_character_index,
            source_2.last_character_index,
        )
        common_start = max(start, start1)
        common_end = min(end, end1)

        intersection = max(0, common_end - common_start)
        union = (end - start) + (end1 - start1) - intersection

        return union > 0 and intersection / union >= 0.05

    def recall_question(
        self, student: MinimalSearchResults, truth: AnsweredQuestion, k: int
    ) -> float:
        if not truth.sources:
            return 0.0
        top_k = student.retrieved_sources[:k]
        found = 0
        for true_source in truth.sources:
            if any(
                s.file_path == true_source.file_path
                and self._check_overlap(s, true_source)
                for s in top_k
            ):
                found += 1
        return found / len(truth.sources)

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        try:
            results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )

            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
        except ValidationError as e:
            raise EvaluateError(f"Invalid {e.title} file format") from e
        except OSError as e:
            raise EvaluateError(f"Error when opening {e.filename}") from e

        truths = {
            q.question_id: q
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion)
        }

        pairs = [
            (res, truths[res.question_id])
            for res in results.search_results
            if res.question_id in truths
        ]

        if not pairs:
            raise EvaluateError("No matching question id for the two files")
        for k in (1, 3, 5, 10):
            if k > results.k:
                break
            scores = sum(
                self.recall_question(r, t, k) for r, t in pairs
            ) / len(pairs)
            print(f"Recall@{k}: {scores:.3f}")
