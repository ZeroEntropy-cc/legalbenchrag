import asyncio
from collections.abc import Coroutine
from typing import Any

from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self

from legalbenchrag.benchmark_types import (
    Document,
    QAGroundTruth,
    RetrievalMethod,
    RetrievedSnippet,
)


class QAResult(BaseModel):
    qa_gt: QAGroundTruth
    retrieved_snippets: list[RetrievedSnippet]

    @computed_field  # type: ignore[misc]
    @property
    def precision(self) -> float:
        total_retrieved_len = 0
        relevant_retrieved_len = 0
        for snippet in self.retrieved_snippets:
            total_retrieved_len += snippet.span[1] - snippet.span[0]
            # It's guaranteed that gt_snippets don't overlap
            for gt_snippet in self.qa_gt.snippets:
                if snippet.file_path == gt_snippet.file_path:
                    common_min = max(snippet.span[0], gt_snippet.span[0])
                    common_max = min(snippet.span[1], gt_snippet.span[1])
                    if common_max > common_min:
                        relevant_retrieved_len += common_max - common_min
        if total_retrieved_len == 0:
            return 0
        return relevant_retrieved_len / total_retrieved_len

    @computed_field  # type: ignore[misc]
    @property
    def recall(self) -> float:
        total_relevant_len = 0
        relevant_retrieved_len = 0
        for gt_snippet in self.qa_gt.snippets:
            total_relevant_len += gt_snippet.span[1] - gt_snippet.span[0]
            # It's guaranteed that gt_snippets don't overlap
            for snippet in self.retrieved_snippets:
                if snippet.file_path == gt_snippet.file_path:
                    common_min = max(snippet.span[0], gt_snippet.span[0])
                    common_max = min(snippet.span[1], gt_snippet.span[1])
                    if common_max > common_min:
                        relevant_retrieved_len += common_max - common_min
        if total_relevant_len == 0:
            return 0
        return relevant_retrieved_len / total_relevant_len


def avg(arr: list[float]) -> float:
    if len(arr) == 0:
        return float("nan")
    return sum(arr) / len(arr)


class BenchmarkResult(BaseModel):
    qa_result_list: list[QAResult]
    weights: list[float]

    def get_avg_recall_and_precision(
        self, tag_filter: str | None = None
    ) -> tuple[float, float]:
        indices = [
            i
            for i, qa_result in enumerate(self.qa_result_list)
            if (tag_filter is None or tag_filter in qa_result.qa_gt.tags)
        ]
        filtered_qa_results = [self.qa_result_list[i] for i in indices]
        filtered_weights = [self.weights[i] for i in indices]
        avg_weight = avg(filtered_weights)
        return (
            avg(
                [
                    qa_result.recall * weight / avg_weight
                    for qa_result, weight in zip(filtered_qa_results, filtered_weights)
                ]
            ),
            avg(
                [
                    qa_result.precision * weight / avg_weight
                    for qa_result, weight in zip(filtered_qa_results, filtered_weights)
                ]
            ),
        )

    @computed_field  # type: ignore[misc]
    @property
    def avg_precision(self) -> float:
        return self.get_avg_recall_and_precision()[1]

    @computed_field  # type: ignore[misc]
    @property
    def avg_recall(self) -> float:
        return self.get_avg_recall_and_precision()[0]

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if len(self.qa_result_list) != len(self.weights):
            raise ValueError("length of qa_result_list and weights do not match!")
        return self


async def run_benchmark(
    qa_gt_list: list[QAGroundTruth],
    corpus: list[Document],
    retrieval_method: RetrievalMethod,
    *,
    weights: list[float] | None = None,
) -> BenchmarkResult:
    # Process the documents
    for document in corpus:
        await retrieval_method.ingest_document(document)
    await retrieval_method.sync_all_documents()

    # Run the benchmark
    async def run_query(qa_gt: QAGroundTruth) -> QAResult:
        query_response = await retrieval_method.query(qa_gt.query)
        return QAResult(
            qa_gt=qa_gt, retrieved_snippets=query_response.retrieved_snippets
        )

    tasks: list[Coroutine[Any, Any, QAResult]] = [
        run_query(qa_gt) for qa_gt in qa_gt_list
    ]
    results = await asyncio.gather(*tasks)

    await retrieval_method.cleanup()

    return BenchmarkResult(
        qa_result_list=results,
        weights=weights if weights is not None else [1.0] * len(results),
    )
