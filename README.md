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
pip-sync
```

3. Create your credentials.toml and set your API keys

```bash
cp ./credentials/credentials.example.toml ./credentials/credentials.toml
vim ./credentials/credentials.toml
```

4. Run the test script

```bash
python test.py
```
