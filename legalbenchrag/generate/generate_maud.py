import asyncio
import json
import os
import re
import shutil
import zipfile
from collections.abc import Coroutine
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel
from unidecode import unidecode

from legalbenchrag.benchmark_types import (
    Benchmark,
    QAGroundTruth,
    Snippet,
    sort_and_merge_spans,
)
from legalbenchrag.generate.utils import WRITE_TITLES, create_title, download_zip

"""
Processing the MAUD dataset is a bit confusing / complicated.
maud_analysis.py helps in the analysis of the below issues.

main.csv's "Filename (anon)" column does NOT match the actual filename.txt
- But, MAUD_{dev,test,train}.csv's "contract_name" column does match the actual filename.txt
- So, we use get_contract_name_from_mae to deduce it, which uses MAUD_{dev,test,train}.csv.

Some of the columns' annotations aren't high-quality, so column_to_queries sets those to "None".
- For quality we want to ensure that the annotation matches all relevant text, and no irrelevant text.

The main.csv's annotations often aren't substrings of the original text, for many reasons:
- <omitted> tags can imply skipping over regions of the original text. But, often that creates ambiguity
  - E.g. "X <omitted> Y <omitted> Z", if Y occurs multiple times between X and Z, it isn't clear which Y it should it be
- The annotation column sometimes repeats the same annotation multiple times, at different levels of succinctness.
- The PDF->txt OCR appears to have output a lot of rare unicode characters (e.g. Greek Question Mark instead of semicolon), and these oddities sometimes aren't present in the annotation text.
- The annotations often have extra or missing whitespace.

We handle these issues by:
- For the "Y ambiguity" and repeated annotation issues, we mark those as failures.
- For the unicode oddities and whitespace inconsistencies, we use unidecode on and remove whitespace from both annotation and original text.
  - We use a sourcemap to keep track the original span indices of the modified original text.

main.csv has two rows, "Soliton, Inc._AbbVie Inc..pdf" and "Soliton_Inc_Abbvie_Inc.pdf", both of which point to contract_127.
- The second one, is good. The first one, is kinda broken. The first one is also the only one where MAE isn't a perfect match to the column in MAUD_{dev,test,train}.csv.
- So, we explicitly drop the first version ("Soliton, Inc._AbbVie Inc..pdf").
"""

SHOW_COLUMN_NAME_TO_QTY = False

save_path = "./data/raw_data/maud"
corpus_path = "./data/corpus/maud"


def download_maud() -> None:
    download_zip(
        name="MAUD",
        url="https://github.com/TheAtticusProject/maud/archive/refs/heads/main.zip",
        save_path=save_path,
        check_path="maud-main",
    )

    # Unzip data.zip
    if not os.path.exists(f"{save_path}/maud-main/data"):
        with zipfile.ZipFile(f"{save_path}/maud-main/data.zip", "r") as zip_ref:
            zip_ref.extractall(f"{save_path}/maud-main")
        print("Unzipped MAUD repo's data.zip")


def get_contract_text(contract_name: str) -> str:
    with open(f"{save_path}/maud-main/data/contracts/{contract_name}.txt") as f:
        return f.read()


column_to_queries: dict[str, str | None] = {
    # General Information
    "Type of Consideration": "What is the Type of Consideration",
    # Conditions to Closing
    "Accuracy of Target R&W Closing Condition": "Information about the Closing Condition: Accuracy of Target's Representations and Warranties",
    "Compliance with Covenant Closing Condition": "Information about the Closing Condition: Compliance with Covenants",
    "Absence of Litigation Closing Condition": "Information about the Closing Condition: No Litigation clause",
    # Material Adverse Effect
    'Agreement includes a "Back-Door" MAE': None,  # These annotations are low-quality for some reason.
    '"No MAE" R&W Made as of a Specified Date': "What is the Target's Representation & Warranty of No Material Adverse Effect, with regards to some specified date",
    "MAE Definition": 'What is the Definition of "Material Adverse Effect"',
    # Knowledge
    "Knowledge Definition": 'What is the Definition of "Knowledge"',
    # Deal Protection and Related Provisions
    "No-Shop": "Where is the No-Shop Clause",
    "Fiduciary exception:  Board determination (no-shop)": "What about the Fiduciary exception to the No-Shop Clause",  # Good Enough
    "Fiduciary exception to COR covenant": None,  # These annotations are medium-quality
    "Agreement provides for matching rights in connection with COR": None,  # These annotations are medium-quality
    "Superior Offer Definition": 'What is the Definition of "Superior Proposal"',
    "Intervening Event Definition": 'What is the Definition of "Interveining Event"',
    "FTR Triggers": "Information about the Fiduciary Termination Right Triggers for termination",  # Often only talks about a superior proposal
    "Limitations on FTR Exercise": None,  # Not great / succinct
    "Agreement provides for matching rights in connection with FTR": None,  # These annotations are medium-quality
    "Tail Period & Acquisition Proposal Details": "Is there a Tail provision for acquisition proposals",  # Good annotations, but the name "tail provision" is unclear in my research.
    "Breach of No Shop": "What happens during a Breach of No-Shop clause",
    "Breach of Meeting Covenant": "What happens during a Breach of Shareholder Meeting Covenant",
    # Operating and Efforts Covenant
    "Ordinary course covenant": "What are the Ordinary course of business covenants",
    "Negative interim operating covenant": None,  # These annotations aren't really focused on just "Negative"
    # - https://www.bloomberglaw.com/external/document/XNAGA80000000/m-a-clause-description-closing-conditions-regulatory-approvals
    "General Antitrust Efforts Standard": "Where is the Closing Conditions: Regulatory Approvals clause",
    "Limitations on Antitrust Efforts": "I want information about the Limitations on Antitrust Efforts",  # These annotations are succinct, but are often missing section titles / context.
    # Remedies
    "Specific Performance": "Where is the Specific Performance clause",
}


def get_contract_name_from_mae(
    df_testcases: pd.DataFrame, mae_definition: str, *, contract_127_fix: bool = False
) -> str:
    # only maud_analysis.py sets contract_127_fix
    if contract_127_fix and mae_definition.endswith(
        "terial Adverse Effect).  (Pages 74-75)"
    ):
        mae_definition = mae_definition.replace(
            "ternal or public projections,   70\n\n\nforecasts, gu",
            "ternal or public projections, \xa0 70\n\n\nforecasts, gu",
        )
        mae_definition = mae_definition.replace(
            "terial Adverse Effect).  (Pages 74-75)",
            "terial Adverse Effect). (Pages 74-75)",
        )
    # Find the match
    mae_matches = cast(
        list[str],
        df_testcases[
            (df_testcases["text_type"] == "MAE Definition")
            & (df_testcases["text"] == mae_definition)
            & (df_testcases["contract_name"] != "<RARE_ANSWERS>")
        ]["contract_name"]
        .unique()
        .tolist(),
    )
    if len(mae_matches) != 1:
        raise RuntimeError(f"Bad # Matches! {len(mae_matches)}")
    return cast(str, mae_matches[0]).replace(".pdf", "")


async def generate_maud() -> None:
    download_maud()
    df_testcases = pd.concat(
        [
            pd.read_csv(f"{save_path}/maud-main/data/MAUD_dev.csv"),
            pd.read_csv(f"{save_path}/maud-main/data/MAUD_test.csv"),
            pd.read_csv(f"{save_path}/maud-main/data/MAUD_train.csv"),
        ]
    )

    # Mapping from contract name to filename,
    # Only for the ones that we've used.
    used_contract_name_to_filename: dict[str, str] = {}
    qa_list: list[QAGroundTruth] = []
    failures = 0
    successes = 0
    column_name_to_qty = {}

    class RowInformation(BaseModel):
        filename: str
        contract_name: str
        title: str
        total_text: str
        total_text_sourcemap: list[int]
        column_values: dict[str, str]

    async def process_row(row: pd.Series) -> RowInformation:
        filename = cast(str, row["Filename"])
        filename = filename.replace("\n", "|")
        assert filename.endswith(".pdf")
        filename = filename[:-4]
        # row["Filename (anon)"] is Nonsense
        contract_name = get_contract_name_from_mae(
            df_testcases, cast(str, row["MAE Definition"])
        )

        # Get the text and sourcemap
        with open(f"{save_path}/maud-main/data/contracts/{contract_name}.txt") as f:
            total_text_raw = f.read()

        title = await create_title(
            filename=filename + ".txt",
            text=total_text_raw[:15000],
            extra_instructions="\n".join(
                [
                    'First, in your thoughts, try think about and list the names of ALL parties involved. Think a lot, keep listing entities, before ending with "That\'s all of the entities that I could find"',
                    'Think about whether or not the listed parent is just a shell, and the true parent is some larger entity. If so, list the "True Parent"',
                    'For the Mergers & Acquisitions agreement, think about which one is labelled the "Parent", and which one is not.',
                    'If there is a parent / acquiring body: say "There is a parent / acquiring body, and I will use the parent / acquiring body title format". If there is not, say "There is no parent / acquiring body, and I will use the merger title format."',
                    # Sometimes acquisitions are marked as mergers, that's ok.
                    'If there is a parent / acquiring body, use the title format: "Acquisition Agreement between Parent "X" and Target "Y"", replacing X and Y with the actual entity names, but keeping the company name wrapped in double-quotes.',
                    'ONLY if there is no clear and obvious "Parent", use the title format: "Merger Agreement between "X" and "Y""',
                    'If you don\'t know the names of the parties, say "Arbitrary M&A Agreement"',
                ]
            )
            + "\n",
        )

        # Construct without spaces, but original mapping to ranges
        total_text = ""
        total_text_sourcemap: list[int] = []
        for i, c in enumerate(total_text_raw):
            if ord(c) >= 0x80:
                c = unidecode(c)
                for c0 in c:
                    if not c0.isspace():
                        total_text += c0
                        total_text_sourcemap.append(i)
            elif not c.isspace():
                total_text += c
                total_text_sourcemap.append(i)

        column_values: dict[str, str] = {}
        for column_name in column_to_queries:
            column_values[column_name] = cast(str, row[column_name])

        return RowInformation(
            filename=filename,
            contract_name=contract_name,
            title=title,
            total_text=total_text,
            total_text_sourcemap=total_text_sourcemap,
            column_values=column_values,
        )

    tasks: list[Coroutine[Any, Any, RowInformation]] = []

    # Read the CSV
    df = pd.read_csv(f"{save_path}/maud-main/data/raw/main.csv")
    for _, row in df.iterrows():
        if cast(str, row["Filename"]) == "Soliton, Inc._AbbVie Inc..pdf":
            # This one is broken and duplicated
            continue
        tasks.append(process_row(row))

    # Iterate over the gathered row infos
    row_infos = await asyncio.gather(*tasks)
    if WRITE_TITLES:
        with open("./tmp/maud_titles.txt", "w") as f:
            for i, row_info in enumerate(row_infos):
                f.write(f"{i}: {row_info.filename}.pdf -> {row_info.title}\n")
    for row_info in row_infos:
        # Iterate over queries
        for column_name, query in column_to_queries.items():
            if query is None:
                continue
            column_value = row_info.column_values[column_name]
            column_value = unidecode(column_value)
            column_value = re.sub(r"\s+", "", column_value)
            matching_texts = cast(list[str], re.split(r"\s*<omitted>\s*", column_value))
            matching_texts[-1] = re.sub(
                r"\(Pages?\s*[\d-]+\)\s*$", "", matching_texts[-1]
            )

            # Parse through the document, accumulating matching spans
            spans: list[tuple[int, int]] = []
            current_index = 0
            did_fail = False
            for _, matching_text in enumerate(matching_texts):
                index = row_info.total_text.find(matching_text, current_index)
                if index == -1:
                    did_fail = True
                    break
                current_index = index + len(matching_text)
                # Don't include if it could've matched in multiple places, because then it's ambiguous
                if row_info.total_text.find(matching_text, current_index) != -1:
                    did_fail = True
                    break
                # Keep the span based on the original text
                spans.append(
                    (
                        row_info.total_text_sourcemap[index],
                        row_info.total_text_sourcemap[index + len(matching_text)],
                    )
                )
            if did_fail:
                failures += 1
                continue
            successes += 1
            spans = sort_and_merge_spans(spans, max_bridge_gap_len=0)

            # Handle success
            if column_name not in column_name_to_qty:
                column_name_to_qty[column_name] = 0
            column_name_to_qty[column_name] += 1

            contract_name = row_info.contract_name
            filename = row_info.filename
            if contract_name in used_contract_name_to_filename:
                assert used_contract_name_to_filename[contract_name] == filename
            used_contract_name_to_filename[contract_name] = filename
            qa_list.append(
                QAGroundTruth(
                    query=f"Consider the {row_info.title}; {query}",
                    snippets=[
                        Snippet(
                            file_path=f"maud/{filename}.txt",
                            span=span,
                        )
                        for span in spans
                    ],
                )
            )
    # Success Rate: 58%
    # print(f"Success Rate: {successes/(successes+failures)*100:.0f}%")
    if SHOW_COLUMN_NAME_TO_QTY:
        print(json.dumps(column_name_to_qty, indent=4))

    if os.path.exists(corpus_path):
        shutil.rmtree(corpus_path)
    os.makedirs(corpus_path, exist_ok=True)
    for contract_name, filename in used_contract_name_to_filename.items():
        shutil.copy(
            f"{save_path}/maud-main/data/contracts/{contract_name}.txt",
            f"{corpus_path}/{filename}.txt",
        )

    with open("./data/benchmarks/maud.json", "w") as f:
        f.write(Benchmark(tests=qa_list).model_dump_json(indent=4))


if __name__ == "__main__":

    async def main() -> None:
        await generate_maud()

    asyncio.run(main())
