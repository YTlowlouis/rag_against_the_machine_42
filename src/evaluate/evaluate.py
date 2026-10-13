from pathlib import Path
from pydantic import ValidationError
from src.models.provided_models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
)


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
        lenght_common = max(0, common_end - common_start)
        return True

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
        results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )

        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        for question in results.search_results:
            id = question.question_id
