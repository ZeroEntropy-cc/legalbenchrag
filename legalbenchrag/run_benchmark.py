import asyncio
from collections.abc import Coroutine
from typing import Any

from pydantic import BaseModel, computed_field

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

    @computed_field  # type: ignore[misc]
    @property
    def avg_precision(self) -> float:
        return avg([qa_result.precision for qa_result in self.qa_result_list])

    @computed_field  # type: ignore[misc]
    @property
    def avg_recall(self) -> float:
        return avg([qa_result.recall for qa_result in self.qa_result_list])


async def run_benchmark(
    qa_gt_list: list[QAGroundTruth],
    corpus: list[Document],
    retrieval_method: RetrievalMethod,
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

    return BenchmarkResult(qa_result_list=results)
