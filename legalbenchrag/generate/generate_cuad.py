import ast
import asyncio
import os
import shutil
from collections.abc import Coroutine
from typing import Any, cast

import pandas as pd

from legalbenchrag.benchmark_types import (
    Benchmark,
    QAGroundTruth,
    Snippet,
    sort_and_merge_spans,
)
from legalbenchrag.generate.utils import WRITE_TITLES, create_title, download_zip

save_path = "./data/raw_data/cuad"


def download_cuad() -> None:
    download_zip(
        name="CUAD",
        url="https://zenodo.org/record/4595826/files/CUAD_v1.zip?download=1",
        save_path=save_path,
        check_path="CUAD_v1",
    )


def filename_pdf_to_text(pdf_filename: str) -> str:
    return pdf_filename.replace(".pdf", ".txt").replace(".PDF", ".txt")


def extract_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    start_idx = text.find(quote)
    if start_idx == -1:
        return None

    # If the quote occurs more than once, return None
    next_occurrence = text.find(quote, start_idx + len(quote))
    if next_occurrence != -1:
        return None

    # Return the index span
    return start_idx, start_idx + len(quote)


async def generate_cuad() -> None:
    download_cuad()

    df = pd.read_csv(f"{save_path}/CUAD_v1/master_clauses.csv")
    # Agreement Date - The snippet is too exact
    # Effective Date - The snippet is too exact

    tasks: list[tuple[int, Coroutine[Any, Any, str]]] = []
    for i, row in df.iterrows():
        assert isinstance(i, int)
        filename = filename_pdf_to_text(cast(str, row["Filename"]))
        if ".txt" not in filename:
            continue
        filepath = f"{save_path}/CUAD_v1/full_contract_txt/{filename}"
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            text = f.read()
        tasks.append((i, create_title(filename, text)))
    extracted_rows = list(
        zip(
            [task[0] for task in tasks],
            await asyncio.gather(*[task[1] for task in tasks]),
        )
    )
    filtered_extracted_rows: list[tuple[int, str]] = []
    for extracted_row in extracted_rows:
        filename = cast(str, df.loc[extracted_row[0], "Filename"]).lower()
        title = extracted_row[1].lower()

        # Unless the title says it's an amendment, then it's ambiguous
        if (
            any(
                s in filename
                for s in [
                    "agreement2",
                    "agreement3",
                    "agreement4",
                ]
            )
            and "amendment" not in title
        ):
            continue

        # Ambiguous
        if "part1" in filename or "part2" in filename:
            continue

        filtered_extracted_rows.append(extracted_row)

    if WRITE_TITLES:
        with open("./tmp/cuad_titles.txt", "w") as f:
            for i, title in filtered_extracted_rows:
                f.write(f"{i}: {df.loc[i, "Filename"]} -> {title}\n")

    column_queries = {
        "Expiration Date": "What is the expiration date of this contract?",
        "Renewal Term": "What is the renewal term for this contract?",
        "Notice Period To Terminate Renewal": "What is the notice period required to terminate the renewal?",
        "Governing Law": "What is the governing law for this contract?",
        "Most Favored Nation": "Is there a most favored nation clause in this contract?",
        "Competitive Restriction Exception": "Are there any exceptions to competitive restrictions in this contract?",
        "Non-Compete": "Is there a non-compete clause in this contract?",
        "Exclusivity": "Does this contract include an exclusivity agreement?",
        "No-Solicit Of Customers": "Is there a clause preventing the solicitation of customers in this contract?",
        "No-Solicit Of Employees": "Is there a clause preventing the solicitation of employees in this contract?",
        "Non-Disparagement": "Is there a non-disparagement clause in this contract?",
        "Termination For Convenience": "Can this contract be terminated for convenience, and under what conditions?",
        "Rofr/Rofo/Rofn": "Does this contract include any right of first refusal, right of first offer, or right of first negotiation?",
        "Change Of Control": "What happens in the event of a change of control of one of the parties in this contract?",
        "Anti-Assignment": "Is there an anti-assignment clause in this contract?",
        "Revenue/Profit Sharing": "Does this contract include any revenue or profit-sharing arrangements?",
        "Price Restrictions": "Are there any price restrictions or controls specified in this contract?",
        "Minimum Commitment": "Is there a minimum commitment required under this contract?",
        "Volume Restriction": "Does this contract include any volume restrictions?",
        "Ip Ownership Assignment": "How is intellectual property ownership assigned in this contract?",
        "Joint Ip Ownership": "Does this contract provide for joint intellectual property ownership?",
        "License Grant": "What licenses are granted under this contract?",
        "Non-Transferable License": "Are the licenses granted under this contract non-transferable?",
        "Affiliate License-Licensor": "Does the licensor's affiliates have any licensing rights under this contract?",
        "Affiliate License-Licensee": "Does the licensee's affiliates have any licensing rights under this contract?",
        "Unlimited/All-You-Can-Eat-License": "Does this contract include an unlimited or all-you-can-eat license?",
        "Irrevocable Or Perpetual License": "Are any of the licenses granted under this contract irrevocable or perpetual?",
        "Post-Termination Services": "Are there any services to be provided after the termination of this contract?",
        "Audit Rights": "What are the audit rights under this contract?",
        "Uncapped Liability": "Is there uncapped liability under this contract?",
        "Cap On Liability": "Is there a cap on liability under this contract?",
        "Warranty Duration": "What is the duration of any warranties provided in this contract?",
        "Insurance": "What are the insurance requirements under this contract?",
        "Covenant Not To Sue": "Is there a covenant not to sue included in this contract?",
        "Third Party Beneficiary": "Are there any third-party beneficiaries designated in this contract?",
    }

    qa_list: list[QAGroundTruth] = []
    used_filenames: set[str] = set()

    for i, generated_title in filtered_extracted_rows:
        row = df.iloc[i]
        filename = filename_pdf_to_text(cast(str, row["Filename"]))
        with open(f"{save_path}/CUAD_v1/full_contract_txt/{filename}") as f:
            text = f.read()
        for column_name, column_query in column_queries.items():
            # Parse the quotes
            raw_query_quotes = cast(str, row[column_name])
            any_quotes = ast.literal_eval(raw_query_quotes)
            assert isinstance(any_quotes, list)
            quotes: list[str] = []
            for any_quote in any_quotes:
                assert isinstance(any_quote, str)
                quotes.append(any_quote)

            # Save the query
            spans: list[tuple[int, int]] = []
            failed = False
            for quote in quotes:
                index_span = extract_quote_span(text, quote)
                if index_span is not None:
                    spans.append(index_span)
                else:
                    failed = True
                    break
            spans = sort_and_merge_spans(spans, max_bridge_gap_len=1)

            if not failed and len(spans) > 0:
                used_filenames.add(filename)
                qa_list.append(
                    QAGroundTruth(
                        query=f"Consider the {generated_title}; {column_query}",
                        snippets=[
                            Snippet(
                                file_path=f"cuad/{filename}",
                                span=span,
                            )
                            for span in spans
                        ],
                    )
                )

    corpus_dir = "./data/corpus/cuad"
    os.makedirs(corpus_dir, exist_ok=True)
    for used_filename in used_filenames:
        shutil.copy(
            f"{save_path}/CUAD_v1/full_contract_txt/{used_filename}",
            f"{corpus_dir}/{used_filename}",
        )

    benchmark_dir = "./data/benchmarks"
    os.makedirs(benchmark_dir, exist_ok=True)
    with open(f"{benchmark_dir}/cuad.json", "w") as f:
        f.write(Benchmark(tests=qa_list).model_dump_json(indent=4))


if __name__ == "__main__":

    async def main() -> None:
        await generate_cuad()

    asyncio.run(main())
