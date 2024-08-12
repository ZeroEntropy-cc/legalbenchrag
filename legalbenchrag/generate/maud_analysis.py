import asyncio
import os
import re
import shutil
from typing import cast

import pandas as pd
from unidecode import unidecode

from legalbenchrag.generate.generate_maud import (
    download_maud,
    get_contract_name_from_mae,
)

"""
This file writes out a lot of information into ./tmp/maud,
Which helps in analyzing the annotations of each document and along each column (columns = Annotation Type).
"""

save_path = "./data/raw_data/maud"
tmp_save_dir_contracts = "./tmp/maud/contracts"
tmp_save_dir_columns = "./tmp/maud/columns"
if os.path.exists("./tmp/maud"):
    shutil.rmtree("./tmp/maud")


def is_substring(total_text: str, matching_text: str) -> bool:
    logg = False
    if False:
        logg = True
        print("HERE!")
    matching_text = unidecode(matching_text)
    matching_text = re.sub(r"\s+", "", matching_text)
    matching_texts = cast(list[str], re.split(r"\s*<omitted>\s*", matching_text))
    matching_texts[-1] = re.sub(r"\(Pages?\s*[\d-]+\)\s*$", "", matching_texts[-1])
    if logg:
        print(matching_texts)
        with open("test", "w") as f:
            f.write(total_text)

    current_index = 0
    did_fail = False
    for matching_text in matching_texts:
        index = total_text.find(matching_text, current_index)
        if index == -1:
            did_fail = True
            break
        current_index = index + len(matching_text)
        # Don't include if it could've matched in multiple places, because then it's ambiguous
        if total_text.find(matching_text, current_index) != -1:
            did_fail = True
            break
    return not did_fail


def save_main_csv() -> None:
    df_testcases = pd.concat(
        [
            pd.read_csv(f"{save_path}/maud-main/data/MAUD_dev.csv"),
            pd.read_csv(f"{save_path}/maud-main/data/MAUD_test.csv"),
            pd.read_csv(f"{save_path}/maud-main/data/MAUD_train.csv"),
        ]
    )

    column_categories = {
        "General Information": [
            "Type of Consideration",
        ],
        "Conditions to Closing": [
            "Accuracy of Target R&W Closing Condition",
            "Compliance with Covenant Closing Condition",
            "Absence of Litigation Closing Condition",
        ],
        "Material Adverse Effect": [
            'Agreement includes a "Back-Door" MAE',
            '"No MAE" R&W Made as of a Specified Date',
            "MAE Definition",
        ],
        "Knowledge": [
            "Knowledge Definition",
        ],
        "Deal Protection and Related Provisions": [
            "No-Shop",
            "Fiduciary exception:  Board determination (no-shop)",
            "Fiduciary exception to COR covenant",
            "Agreement provides for matching rights in connection with COR",
            "Superior Offer Definition",
            "Intervening Event Definition",
            "FTR Triggers",
            "Limitations on FTR Exercise",
            "Agreement provides for matching rights in connection with FTR",
            "Tail Period & Acquisition Proposal Details",
            "Breach of No Shop",
            "Breach of Meeting Covenant",
        ],
        "Operating and Efforts Covenant": [
            "Ordinary course covenant",
            "Negative interim operating covenant",
            "General Antitrust Efforts Standard",
            "Limitations on Antitrust Efforts",
        ],
        "Remedies": [
            "Specific Performance",
        ],
    }

    os.makedirs(tmp_save_dir_contracts, exist_ok=True)
    os.makedirs(tmp_save_dir_columns, exist_ok=True)

    # column names
    column_names_to_values: dict[str, list[tuple[str, str, str, bool]]] = {
        column_name: []
        for column_names in column_categories.values()
        for column_name in column_names
    }

    # Read the CSV
    used_contract_names: set[str] = set()
    df = pd.read_csv(f"{save_path}/maud-main/data/raw/main.csv")
    for _, row in df.iterrows():
        filename = cast(str, row["Filename"])
        # dumb_contract_name = cast(str, row["Filename (anon)"]).replace(".pdf", "")
        contract_name = get_contract_name_from_mae(
            df_testcases,
            cast(str, row["MAE Definition"]),
            contract_127_fix=True,
        )
        try:
            with open(f"{save_path}/maud-main/data/contracts/{contract_name}.txt") as f:
                total_text_raw = f.read()
        except FileNotFoundError:
            continue
        total_text = unidecode(total_text_raw)
        total_text = re.sub(r"\s+", "", total_text)

        # "Soliton, Inc._AbbVie Inc..pdf" and "Soliton_Inc_Abbvie_Inc.pdf" are both contract_127
        if contract_name in used_contract_names:
            contract_name += "|Second_Match"
        assert contract_name not in used_contract_names
        used_contract_names.add(contract_name)

        with open(f"{tmp_save_dir_contracts}/{contract_name}.md", "w") as f:
            f.write(f"# {filename}\n\n")
            for column_category, column_names in column_categories.items():
                f.write(f"## CATEGORY: {column_category}\n\n")
                for column_name in column_names:
                    column_value = cast(str, row[column_name])
                    is_substr = is_substring(total_text, column_value)
                    column_names_to_values[column_name].append(
                        (filename, contract_name, column_value, is_substr)
                    )
                    f.write(f"### {column_name}\n\n")
                    f.write(column_value + "\n\n")

    for column_name, column_values in column_names_to_values.items():
        column_name_clean = (
            column_name.replace(" ", "_").replace("/", "-").replace("\\", "-")
        )
        with open(f"{tmp_save_dir_columns}/{column_name_clean}.md", "w") as f:
            f.write(f"# {column_name}\n\n")
            for filename, contract_name, column_value, is_substr in column_values:
                f.write(
                    f"## Filename: {filename.replace('\n', '|')} ({contract_name}) ({'SUCCESS!' if is_substr else 'Failed...'})\n\n"
                )
                f.write(column_value + "\n\n")


if __name__ == "__main__":

    async def main() -> None:
        download_maud()
        save_main_csv()

    asyncio.run(main())
