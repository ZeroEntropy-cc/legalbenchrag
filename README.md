# LegalBench-RAG

This repository contains the LegalBench-RAG benchmark, which can test any retrieval system over the task of identifying the correct snippets that answer a given query.

# Download

To download the existing benchmark and corpus, please visit [this link](https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0).

# Usage

1. Install your venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies

```bash
pip install pip-tools
pip-sync && pip install -e .
```

3. Create your credentials.toml and set your API keys

```bash
cp ./credentials/credentials.example.toml ./credentials/credentials.toml
vim ./credentials/credentials.toml
```

4. Download or Generate the dataset

You can download the data using the download link provided above. The directory structure from the root should have a `./data/corpus` folder and a `./data/benchmarks` folder. The corpus folder should be a set of raw text files, potentially with a directory hierarchy within itself. The benchmarks folder should be a set of benchmark json files. Each benchmark json has a set of test cases. Each test case has a query, and a ground truth array of snippets. Each snippet references a text file in the corpus via its file path within the corpus folder, and a character index range of that file.

If instead you would like to re-generate the benchmark from the source datasets, the entire code to do so is also provided in this repository. Please ensure you agree to the usage policies of ContractNLI, CUAD, MAUD, and PrivacyQA, before running this script. Once you have done that, simply execute the following:

```bash
python ./legalbenchrag/generate
```

Please note that LLMs are used in the process of creating the LegalBench-RAG benchmark. So, running this generate script will not generate exactly the same benchmark as was provided in the download link. However, the data in the download link itself was generated from the exact same process.


5. Run the benchmark script

```bash
python ./legalbenchrag/benchmark.py
```

