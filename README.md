*This project has been created as part of the 42 curriculum by <login>.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a
codebase: the [vLLM](https://github.com/vllm-project/vllm) repository (v0.10.1).

The system:

1. **Indexes** the vLLM sources (Python code and Markdown/text documentation)
   into a lexical BM25 index.
2. **Retrieves** the top-k most relevant snippets for a question, each returned
   as a source location: a file path and a character range.
3. **Generates** a natural-language answer grounded in those snippets with the
   `Qwen/Qwen3-0.6B` model, running locally on CPU.
4. **Evaluates** retrieval quality with recall@k against ground-truth datasets.

Everything is driven by a command-line interface built with Python Fire.

## Instructions

### Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/)
- The vLLM corpus placed in `data/raw/vllm-0.10.1/` (not versioned, provided
  with the subject)
- ~2 GB of free RAM to run the model (CPU only, no GPU needed)

### Installation

```bash
make install        # or: uv sync
```

PyTorch is installed from the CPU-only index (configured in `pyproject.toml`)
to avoid downloading several GB of CUDA libraries. The model weights
(~1.2 GB) are downloaded from the Hugging Face Hub on first use and cached in
`~/.cache/huggingface/`.

### Commands

Every command is run as `uv run python -m src <command> [options]`
(or `make run ARGS="<command> [options]"`).

| Command | Description |
|---|---|
| `index --max_chunk_size <int>` | Chunk `data/raw/` and build the index under `data/processed/` (default chunk size: 2000) |
| `search <query> --k <int>` | Return the top-k sources for a single query |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Search a whole dataset and write a `StudentSearchResults` JSON file |
| `answer <query> --k <int>` | Answer a single query from the retrieved context |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Answer every question of a search results file and write a `StudentSearchResultsAndAnswer` JSON file |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Report recall@k against a ground-truth dataset |

### Other Makefile rules

- `make lint`: flake8 + mypy
- `make lint-strict`: flake8 + `mypy --strict`
- `make clean`: remove Python caches

## Example usage

```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 2000

# 2. Search a single question
uv run python -m src search "How do I enable LoRA adapters in vLLM?" --k 5

# 3. Search a whole dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 4. Answer a single question
uv run python -m src answer "How do I enable LoRA adapters in vLLM?" --k 5

# 5. Answer a whole dataset from its search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Example `search` output:

```json
{
  "question_id": "f78cf761-59c1-41b6-957b-30c2c938f348",
  "question": "how to configure the openai server",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md",
      "first_character_index": 1940,
      "last_character_index": 3169
    }
  ]
}
```

## System architecture

```
                 ┌──────────────── index ────────────────┐
data/raw/ ──► chunking (.py / .md, .txt) ──► BM25 index ──► data/processed/
                                                              │
question ──► tokenization ──► BM25 retrieval (top-k) ◄────────┘
                                   │
                                   ▼
                     MinimalSearchResults (file path + char range)
                                   │
                                   ▼
          re-read snippets from disk ──► truncate to token budget
                                   │
                                   ▼
               prompt (system rules + context + question)
                                   │
                                   ▼
                 Qwen/Qwen3-0.6B (transformers, CPU) ──► MinimalAnswer
```

| Component | Location | Role |
|---|---|---|
| CLI | `src/__main__.py` | Fire entry point, error handling |
| `Index` | `src/indexing/index.py` | Chunking, BM25 indexing, persistence, search |
| `AnswerGeneration` | `src/answer/answering.py` | Context building, prompting, generation |
| Data models | `src/models/` | Pydantic models exchanged between stages |

All data exchanged between stages is validated with pydantic models
(`MinimalSource`, `MinimalSearchResults`, `MinimalAnswer`,
`StudentSearchResults`, ...).

## Chunking strategy

Two distinct strategies, both based on LangChain's
`RecursiveCharacterTextSplitter`:

- **Markdown / text** (`.md`, `.txt`): recursive splitting on paragraphs, then
  lines, then words, so that chunks follow the natural structure of the
  document.
- **Python code** (`.py`): `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`,
  which splits first on class and function definitions, keeping logical code
  units together as much as possible.

Common settings:

- **Maximum chunk size**: configurable with `--max_chunk_size`, default 2000
  characters (the maximum accepted by the evaluator).
- **Overlap**: 20% of the chunk size, so that information sitting on a chunk
  boundary is still found in at least one chunk.
- **Character offsets**: `add_start_index=True` records where each chunk starts
  in its file; the end index is `start + len(chunk)`. These offsets are the
  `first_character_index` / `last_character_index` returned by the search.

## Retrieval method

Retrieval uses **BM25** through the [`bm25s`](https://github.com/xhluca/bm25s)
library.

- **Tokenization**: lowercasing, regex word tokenization, English stopwords
  removal; the same tokenizer is applied to chunks and queries.
- **Ranking**: each chunk is scored with BM25, which combines term frequency
  (with saturation, parameter `k1`), inverse document frequency (rare terms
  weigh more) and length normalization (parameter `b`). The top-k chunks are
  returned, best first.
- **Persistence**: the sparse index and the corpus (chunk text + source
  location) are saved under `data/processed/` and reloaded for each search,
  so queries take milliseconds.
- **Batch search**: `search_dataset` tokenizes and retrieves all questions in
  a single call.

## Answer generation

- **Model**: `Qwen/Qwen3-0.6B`, loaded with Hugging Face `transformers` on CPU.
  The model is loaded lazily, only by the `answer` commands, so `index` and
  `search` stay fast.
- **Context**: each retrieved snippet is re-read from disk using its character
  range, prefixed with `[Source n: <file path>]`, and truncated to 500 tokens
  (counted with the model's tokenizer).
- **Prompt**: a system message instructs the model to answer only from the
  provided context and to say when the answer is not in it; the context comes
  first and the question last.
- **Decoding**: greedy decoding (`do_sample=False`) for reproducible answers,
  at most 300 new tokens, Qwen3 thinking mode disabled
  (`enable_thinking=False`).

## Performance analysis

> TODO: fill in after running the evaluation.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Docs | | | | |
| Code | | | | |

| Metric | Requirement | Measured |
|---|---|---|
| Indexing time (whole corpus) | ≤ 5 min | |
| Retrieval (200 questions) | ≤ 90 s | |
| Recall@5 docs | ≥ 80% | |
| Recall@5 code | ≥ 50% | |

## Design decisions

- **BM25 rather than TF-IDF**: BM25 saturates term frequency and normalizes
  by document length, which suits chunks of uneven size and generally ranks
  better than raw TF-IDF.
- **`bm25s`**: fast sparse implementation, with built-in saving/loading of
  the index together with its corpus.
- **LangChain text splitters**: proven recursive splitting with a dedicated
  Python mode and character offset tracking.
- **Hugging Face `transformers` instead of vLLM for inference**: vLLM requires
  an NVIDIA GPU, while the project must run on CPU-only machines.
  `transformers` with CPU-only PyTorch keeps installation light and works
  everywhere.
- **`generate` instead of `pipeline`**: explicit control over the chat
  template (thinking disabled), token budget and decoding.
- **Snippets re-read from disk**: `answer_dataset` only receives source
  locations, so the same context-building code serves both `answer` and
  `answer_dataset`.

## Challenges faced

- **No GPU available**: vLLM could not run locally; switched to
  `transformers` with CPU-only PyTorch.
- **Heavy dependencies**: the default PyTorch wheels pull CUDA libraries
  (several GB); solved with a dedicated CPU index in `pyproject.toml`.
- **Token budget**: truncating snippets by characters does not guarantee a
  token limit; snippets are truncated on token ids instead.
- **Qwen3 thinking mode**: the model emits a `<think>` block by default;
  disabled through the chat template, with a safety strip on the output.
- **Large generated files in Git**: the index and corpus are excluded through
  `.gitignore` and rebuilt with `index`.

## Resources

- [vLLM documentation](https://docs.vllm.ai/)
- [BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [bm25s](https://github.com/xhluca/bm25s)
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [Hugging Face Transformers — text generation](https://huggingface.co/docs/transformers/llm_tutorial)
- [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)](https://arxiv.org/abs/2005.11401)
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [uv](https://docs.astral.sh/uv/)

### AI usage

Claude (Anthropic) was used as an assistant during the project, for:

- **Explanations**: the role of Hugging Face vs vLLM, `pipeline` vs
  `generate`, how to load and prompt Qwen3 with `transformers`.
- **Environment**: configuring CPU-only PyTorch with uv, cleaning up an
  interrupted vLLM installation, `.gitignore` and Git history cleanup.
- **Answer generation** (`src/answer/answering.py`): help structuring the
  pipeline, writing the prompt builder, token-based truncation and output
  decoding, and reviewing the code for bugs.
- **Documentation**: drafting this README.

Indexing and retrieval design, integration and testing were done by the
author.
