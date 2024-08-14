import asyncio
import os
from collections.abc import Coroutine
from typing import Any

from pydantic import BaseModel

from legalbenchrag.benchmark_types import (
    Benchmark,
    QAGroundTruth,
    Snippet,
    sort_and_merge_spans,
)
from legalbenchrag.generate.utils import WRITE_TITLES, create_title, download_zip

save_path = "./data/raw_data/contractnli"


def download_contractnli() -> None:
    download_zip(
        name="ContractNLI",
        url="https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
        save_path=save_path,
        check_path="contract-nli",
    )


# JSON Parser
class AnnotationJSON(BaseModel):
    choice: str
    spans: list[int]


class AnnotationSetJSON(BaseModel):
    annotations: dict[str, AnnotationJSON]


class DocumentJSON(BaseModel):
    id: int
    file_name: str
    text: str
    spans: list[tuple[int, int]]
    annotation_sets: list[AnnotationSetJSON]


class DatasetJSON(BaseModel):
    documents: list[DocumentJSON]
    labels: dict[str, dict[str, str]] | None = None


def get_dataset_json() -> DatasetJSON:
    with open(f"{save_path}/contract-nli/train.json") as f:
        train_dataset = DatasetJSON.model_validate_json(f.read())
    with open(f"{save_path}/contract-nli/test.json") as f:
        test_dataset = DatasetJSON.model_validate_json(f.read())
    return DatasetJSON(
        documents=train_dataset.documents + test_dataset.documents,
    )


async def generate_contractnli() -> None:
    download_contractnli()
    label_mapping = {
        "nda-11": "Does the document mention whether or not the Receiving Party is allowed to reverse engineer any objects which embody the Disclosing Party's Confidential Information?",
        "nda-16": "Does the document specify whether the Receiving Party is required to destroy or return Confidential Information upon the termination of the Agreement?",
        "nda-15": "Does the document indicate that the Agreement does not grant the Receiving Party any rights to the Confidential Information?",
        "nda-10": "Does the document include a clause that prevents the Receiving Party from disclosing the fact that the Agreement was agreed upon or negotiated?",
        "nda-2": "Does the document state that Confidential Information shall only include technical information?",
        "nda-1": "Does the document require that all Confidential Information be expressly identified by the Disclosing Party?",
        "nda-19": "Does the document mention that some obligations of the Agreement may survive the termination of the Agreement?",
        "nda-12": "Does the document allow the Receiving Party to independently develop information that is similar to the Confidential Information?",
        "nda-20": "Does the document permit the Receiving Party to retain some Confidential Information even after its return or destruction?",
        "nda-3": "Does the document allow verbally conveyed information to be considered as Confidential Information?",
        "nda-18": "Does the document include a clause that prohibits the Receiving Party from soliciting some of the Disclosing Party's representatives?",
        "nda-7": "Does the document allow the Receiving Party to share some Confidential Information with third parties, including consultants, agents, and professional advisors?",
        "nda-17": "Does the document permit the Receiving Party to create a copy of some Confidential Information under certain circumstances?",
        "nda-8": "Does the document require the Receiving Party to notify the Disclosing Party if they are required by law, regulation, or judicial process to disclose any Confidential Information?",
        "nda-13": "Does the document allow the Receiving Party to acquire information similar to the Confidential Information from a third party?",
        "nda-5": "Does the document allow the Receiving Party to share some Confidential Information with their employees?",
        "nda-4": "Does the document restrict the use of Confidential Information to the purposes stated in the Agreement?",
    }

    # Process the dataset
    corpus_dir = "./data/corpus/contractnli"
    os.makedirs(corpus_dir, exist_ok=True)
    used_filename_to_text: dict[str, str] = {}
    qa_list: list[QAGroundTruth] = []

    dataset_json = get_dataset_json()
    for document in dataset_json.documents:
        filename = document.file_name.strip()
        filename = filename.replace(".pdf", ".txt")
        if ".txt" not in filename:
            continue
        for annotation_set in document.annotation_sets:
            for annotation_label, annotation in annotation_set.annotations.items():
                spans = [document.spans[span_index] for span_index in annotation.spans]
                spans = sort_and_merge_spans(spans, max_bridge_gap_len=1)
                if len(spans) == 0:
                    continue
                used_filename_to_text[filename] = document.text
                qa_list.append(
                    QAGroundTruth(
                        query=label_mapping[annotation_label],
                        snippets=[
                            Snippet(
                                file_path=f"contractnli/{filename}",
                                span=span,
                            )
                            for span in spans
                        ],
                    )
                )

    # Create a filename->title mapping
    extra_instructions = (
        "\n".join(
            [
                'First, try to list the names of all parties involved, in your thoughts. If the name of a party is a generic "Company A", then that counts as not knowing the name of that party.'
                'If there are two parties involved, use the title format: "Non-Disclosure Agreement between X and Y"',
                'If there is only one parties involved, use the title format: "X\'s Non-Disclosure Agreement"',
                'If you don\'t know the names of either party, say "Arbitrary Non-Disclosure Agreement"',
            ]
        )
        + "\n"
    )
    title_tasks: list[Coroutine[Any, Any, str]] = []
    for filename in used_filename_to_text:
        title_tasks.append(
            create_title(
                filename=filename,
                text=f"{save_path}/contract-nli/{filename}",
                extra_instructions=extra_instructions,
            )
        )
    titles = await asyncio.gather(*title_tasks)
    filename_to_title = dict(zip(used_filename_to_text.keys(), titles))

    # Replace queries, and filter bad titles
    valid_filenames: set[str] = set()
    new_qa_list = []
    for qa in qa_list:
        filename = qa.snippets[0].file_path.split("/")[-1]
        title = filename_to_title[filename]
        # NOTE: Manually verify that this correctly inserts definite article "the".
        if "'s" in title.lower() and not title.lower().startswith("non-"):
            # "Consider X's Non-Disclosure agreement"
            qa.query = f"Consider {title}; {qa.query}"
        else:
            # "Consider the Non-Disclosure agreement between X and Y"
            qa.query = f"Consider the {title}; {qa.query}"
        if title.lower().count("arbitrary") == 0:
            valid_filenames.add(filename)
            new_qa_list.append(qa)
    qa_list = new_qa_list

    # Write out all the files
    for filename in valid_filenames:
        text = used_filename_to_text[filename]
        with open(f"{corpus_dir}/{filename}", "w") as f:
            f.write(text)

    if WRITE_TITLES:
        with open("./tmp/contractnli_titles.txt", "w") as f:
            for filename in valid_filenames:
                title = filename_to_title[filename]
                f.write(f"{filename} -> {title}\n")

    benchmark_dir = "./data/benchmarks"
    os.makedirs(benchmark_dir, exist_ok=True)
    with open(f"{benchmark_dir}/contractnli.json", "w") as f:
        f.write(Benchmark(tests=qa_list).model_dump_json(indent=4))


if __name__ == "__main__":

    async def main() -> None:
        await generate_contractnli()

    asyncio.run(main())
