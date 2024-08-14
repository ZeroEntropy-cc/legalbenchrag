import asyncio
import datetime as dt
import glob
import os

from legalbenchrag.benchmark_types import Benchmark, Document, QAGroundTruth
from legalbenchrag.methods.baseline import BaselineRetrievalMethod, RetrievalStrategy
from legalbenchrag.run_benchmark import run_benchmark
from legalbenchrag.utils.ai import AIEmbeddingModel, AIRerankModel


async def main() -> None:
    all_tests: list[QAGroundTruth] = []
    for file_path in glob.glob("./data/benchmarks/*.json"):
        # Temporary filter due to lack of batching
        if "privacy_qa" not in file_path:
            continue
        with open(file_path) as f:
            benchmark = Benchmark.model_validate_json(f.read())
            all_tests.extend(benchmark.tests)
    benchmark = Benchmark(
        tests=all_tests,
    )

    retrieval_method = BaselineRetrievalMethod(
        retrieval_strategy=RetrievalStrategy(
            name="OpenAI Embeddings",
            embedding_model=AIEmbeddingModel(
                company="openai",
                model="text-embedding-3-large",
                # company="cohere", model="embed-english-v3.0"
            ),
            embedding_topk=300,
            rerank_model=AIRerankModel(company="cohere", model="rerank-english-v3.0"),
            rerank_topk=100,
            token_limit=10000,
        ),
    )

    used_document_file_paths_set: set[str] = {
        snippet.file_path for test in benchmark.tests for snippet in test.snippets
    }
    used_document_file_paths = sorted(used_document_file_paths_set)
    corpus: list[Document] = []
    for document_file_path in used_document_file_paths:
        with open(f"./data/corpus/{document_file_path}") as f:
            corpus.append(
                Document(
                    file_path=document_file_path,
                    content=f.read(),
                )
            )
    print(f"Num Documents: {len(corpus)}")
    print(f"Num Corpus Characters: {sum(len(document.content) for document in corpus)}")
    print(f"Num Queries: {len(benchmark.tests)}")

    benchmark_result = await run_benchmark(
        benchmark.tests,
        corpus,
        retrieval_method,
    )

    # Save the results
    run_name = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    benchmark_path = f"./benchmark_results/{run_name}"
    os.makedirs(benchmark_path, exist_ok=True)
    with open(f"{benchmark_path}/results.json", "w") as f:
        f.write(benchmark_result.model_dump_json(indent=4))

    print(f"Avg Recall: {benchmark_result.avg_recall}")
    print(f"Avg Precision: {benchmark_result.avg_precision}")


if __name__ == "__main__":
    asyncio.run(main())
