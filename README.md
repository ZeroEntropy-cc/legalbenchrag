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

3. Specify your API Keys in `./utils/ai.py`

```python
OPENAI_API_KEY = "YOUR_API_KEY"
COHERE_API_KEY = "YOUR_API_KEY"
ANTHROPIC_API_KEY = "YOUR_API_KEY"
VOYAGE_API_KEY = "YOUR_API_KEY"
```

4. Create the cache directory

```bash
mkdir -p ./data/cache
```

5. Run the test script

```bash
python test.py
```
