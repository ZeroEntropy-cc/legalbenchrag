# LegalBenchRAG

This repository contains the LegalBenchRAG benchmark, which can test any retrieval system over the task of identifying the correct snippets that answer a given query.

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

4. Run the generate script

```bash
python ./legalbenchrag/generate
```

5. Run the benchmark script

```bash
python ./legalbenchrag/benchmark.py
```
