import asyncio
import os
from typing import cast

import pandas as pd
from pydantic import BaseModel

from legalbenchrag.benchmark_types import (
    Benchmark,
    QAGroundTruth,
    Snippet,
    sort_and_merge_spans,
)
from legalbenchrag.generate.utils import download_zip

save_path = "./data/raw_data/privacy_qa"


def download_privacy_qa() -> None:
    download_zip(
        name="PrivacyQA",
        url="https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP/archive/refs/heads/master.zip",
        save_path=save_path,
        check_path="PrivacyQA_EMNLP-master",
    )


async def generate_privacy_qa() -> None:
    download_privacy_qa()

    # Gather the data
    df = pd.concat(
        [
            pd.read_csv(
                "./data/raw_data/privacy_qa/PrivacyQA_EMNLP-master/data/policy_test_data.csv",
                sep="\t",
            ),
            # Train doesn't include data from individual annotators, for confidence interval.
            # pd.read_csv("./data/raw_data/privacy_qa/PrivacyQA_EMNLP-master/data/policy_train_data.csv", sep="\t"),
        ]
    )

    # Extract data from DFs
    class AnnotationInfo(BaseModel):
        sent_id: int
        relevants: int
        irrelevants: int
        no_responses: int

    doc_id_to_sent_id_to_sentence: dict[str, dict[int, str]] = {}
    doc_id_and_query_id_to_question: dict[tuple[str, int], str] = {}
    doc_id_and_query_id_to_annotations: dict[tuple[str, int], list[AnnotationInfo]] = {}
    for i, row in df.iterrows():
        doc_id = cast(str, row["DocID"]).split("_")[0].strip()
        query_id = int(cast(str, row["QueryID"]).split("_")[-1])
        sent_id = int(cast(str, row["SentID"]).split("_")[-1])
        if doc_id not in doc_id_to_sent_id_to_sentence:
            doc_id_to_sent_id_to_sentence[doc_id] = {}
        doc_id_to_sent_id_to_sentence[doc_id][sent_id] = cast(str, row["Segment"])
        doc_id_and_query_id_to_question[(doc_id, query_id)] = cast(
            str, row["Query"]
        ).strip()

        relevants = 0
        irrelevants = 0
        no_responses = 0
        for i in range(1, 7):
            # Cast to str, to turn float "nan" into a string "nan".
            relevance = str(row[f"Ann{i}"]).lower()
            if "irrelevant" in relevance:
                irrelevants += 1
            elif "relevant" in relevance:
                relevants += 1
            elif "nan" in relevance:
                no_responses += 1
            else:
                raise RuntimeError(f"Unexpected Relevance: {relevance}")

        if (doc_id, query_id) not in doc_id_and_query_id_to_annotations:
            doc_id_and_query_id_to_annotations[(doc_id, query_id)] = []
        doc_id_and_query_id_to_annotations[(doc_id, query_id)].append(
            AnnotationInfo(
                sent_id=sent_id,
                relevants=relevants,
                irrelevants=irrelevants,
                no_responses=no_responses,
            )
        )

    # Save the files, and the mapping from sent_ids to spans
    doc_id_to_text: dict[str, str] = {}
    doc_id_to_sent_id_to_span: dict[str, dict[int, tuple[int, int]]] = {}
    corpus_dir = "./data/corpus/privacy_qa"
    os.makedirs(corpus_dir, exist_ok=True)
    for doc_id, sent_id_to_sentence in doc_id_to_sent_id_to_sentence.items():
        doc_id_to_sent_id_to_span[doc_id] = {}
        total_text = ""
        for sent_id, sentence in sorted(
            sent_id_to_sentence.items(), key=lambda x: x[0]
        ):
            sentence += "\n"
            doc_id_to_sent_id_to_span[doc_id][sent_id] = (
                len(total_text),
                len(total_text) + len(sentence),
            )
            total_text += sentence
        doc_id_to_text[doc_id] = total_text

    # Create the qa_list
    used_doc_ids: set[str] = set()
    qa_list: list[QAGroundTruth] = []
    for (doc_id, query_id), annotations in doc_id_and_query_id_to_annotations.items():
        # Skip if any annotations had too many none's
        if any(annotation.no_responses > 2 for annotation in annotations):
            continue
        sent_id_and_scores: list[tuple[int, float]] = [
            (
                annotation.sent_id,
                annotation.relevants / (annotation.relevants + annotation.irrelevants),
            )
            for annotation in annotations
        ]
        sent_id_and_scores = sorted(
            sent_id_and_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        # Get spans above the score threshold
        spans: list[tuple[int, int]] = []
        for sent_id, score in sent_id_and_scores:
            if score >= 0.5:
                span = doc_id_to_sent_id_to_span[doc_id][sent_id]
                spans.append(span)
        span = doc_id_to_sent_id_to_span[doc_id][sent_id_and_scores[0][0]]
        spans = sort_and_merge_spans(spans, max_bridge_gap_len=2)
        if len(spans) == 0:
            continue

        # Store the QA
        used_doc_ids.add(doc_id)
        qa_list.append(
            QAGroundTruth(
                query=f'Consider "{doc_id}"\'s privacy policy; {doc_id_and_query_id_to_question[(doc_id, query_id)]}',
                snippets=[
                    Snippet(
                        file_path=f"privacy_qa/{doc_id}.txt",
                        span=span,
                    )
                    for span in spans
                ],
            )
        )

    # Write the documents and qa_list out
    for doc_id in used_doc_ids:
        with open(f"{corpus_dir}/{doc_id}.txt", "w") as f:
            f.write(doc_id_to_text[doc_id])

    benchmark_dir = "./data/benchmarks"
    os.makedirs(benchmark_dir, exist_ok=True)
    with open(f"{benchmark_dir}/privacy_qa.json", "w") as f:
        f.write(Benchmark(tests=qa_list).model_dump_json(indent=4))


if __name__ == "__main__":

    async def main() -> None:
        await generate_privacy_qa()

    asyncio.run(main())
