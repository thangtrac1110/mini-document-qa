# Mini Document Q&A

Ask questions about your PDF documents and get answers that cite the exact file and
page they came from.

Mini Document Q&A is a small retrieval-augmented generation (RAG) service. It works in
two stages:

1. It extracts text from uploaded PDFs, splits it into page-bounded passages and indexes
   them with BM25.
2. To answer a question, it retrieves the most relevant passages and asks an LLM (Claude,
   via the Anthropic API) to answer **only from those passages**.

The backend validates every citation the model returns. Filenames and page numbers
always come from the backend's own metadata, never from model output.

**Stack:** Python 3.10+ · PyMuPDF · rank-bm25 · FastAPI · Pydantic · Anthropic SDK · pytest

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Demo walkthrough](#demo-walkthrough)
- [Design decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Debugging notes](#debugging-notes)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Project layout](#project-layout)

## Features

- **PDF ingestion:** checks the file type, size and `%PDF-` header, then extracts text
  page by page with 1-based page numbers.
- **Page-bounded chunking:** fixed 100-word chunks that never cross a page, so every
  passage maps to exactly one source page.
- **BM25 retrieval:** lexical ranking with a tokenizer shared by indexing and querying.
  It keeps diacritics (e.g. Vietnamese).
- **Grounded answers:**
  - Only the retrieved passages and the question are sent to the LLM.
  - Answers come back as schema-validated JSON.
  - Citations are checked against the passages actually sent.
- **Honest abstention:** when retrieval finds nothing, the LLM is not called. When the
  passages are not enough, the answer is flagged `insufficient_information`.
- **Controlled failures:** provider timeouts, rate limits, refusals and malformed output
  become clear HTTP errors (502/503/504), and messages never contain credentials.
- **Search without an LLM:** `POST /search` returns ranked passages without calling the
  LLM. It works even when no API key is configured.
- **Evaluation:**
  - A labeled query set with Recall@k, Precision@k and MRR.
  - A controlled chunk-size experiment.
  - A manual answer-review workflow with a rubric.
- **148 automated tests:** they need no API key and make no network calls.

## How it works

```text
Ingestion                                  Question answering (POST /ask)
---------                                  ------------------------------
PDF upload                                 question
  -> validate (extension, size, header)      -> validate (1-1000 chars, top_k 1-20)
  -> extract text per page                   -> BM25: top-k passages (same as /search)
  -> normalize (NFC, collapse whitespace)    -> none found? -> "insufficient", no LLM call
  -> split into 100-word chunks per page     -> label passages S1..Sk, build prompt
  -> rebuild BM25 index over all chunks      -> LLM -> JSON {answer, supporting_source_ids,
  -> commit document + index together                     insufficient_information}
                                             -> drop IDs not in S1..Sk
                                             -> map IDs to chunk / file / page (backend)
```

| Module | Responsibility |
|---|---|
| `app/ingestion/pdf_reader.py` | PDF -> `PageText(page_number, text)`; explicit errors for missing, invalid, encrypted or text-less PDFs |
| `app/ingestion/chunking.py` | Normalization and page-bounded chunks with character offsets |
| `app/retrieval/tokenizer.py` | One tokenizer for both indexing and querying |
| `app/retrieval/bm25_index.py` | BM25 ranking over chunks |
| `app/pipeline.py` | `DocumentService`: all-or-nothing ingestion, in-memory state, search |
| `app/generation/llm_client.py` | `LLMClient` protocol and the Anthropic wrapper (the only provider-specific code) |
| `app/generation/prompts.py` | System prompt, context formatting, answer JSON schema |
| `app/generation/answer_generator.py` | Calls the LLM and validates its output |
| `app/services/qa_service.py` | `QAService`: retrieval -> generation -> citation validation |
| `app/api/` | FastAPI routes, request/response schemas, error-to-status mapping |
| `evaluation/` | Corpus, labeled queries, retrieval metrics, answer-review tooling |

## Quick start

Requirements: Python 3.10 or newer. An Anthropic API key is needed only for `/ask`.

```bash
git clone https://github.com/thangtrac1110/mini-document-qa.git
cd mini-document-qa

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"

pytest                                  # 148 tests, no API key needed
```

Enable answer generation by copying the environment template and setting your key:

```bash
cp .env.example .env                    # Windows: copy .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

Start the server and open the interactive docs at http://127.0.0.1:8000/docs:

```bash
uvicorn app.api.main:app --env-file .env
```

Without a key the server still starts. `/search` works, `/ask` returns 503, and
`GET /health` reports `"llm_configured": false`.

## Configuration

Settings are read from environment variables (or `.env` with `--env-file`). `.env` is
ignored by Git; only `.env.example` is committed.

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (unset) | Enables `/ask`. Read only where the LLM client is built and never logged. |
| `LLM_MODEL` | `claude-opus-5-5` | Model used for answers |
| `LLM_EFFORT` | `medium` | Reasoning effort: `low`, `medium`, `high`, `xhigh`, `max` |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout. The SDK retries 429, 5xx and connection errors twice. |

Fixed defaults in `app/config.py`:

- chunk size: 100 words
- maximum upload size: 20 MB
- LLM `max_tokens`: 16000

> **Data handling:** `/ask` sends the question and the retrieved passages to the
> Anthropic API. Use only documents you are allowed to share with a third-party service,
> and check current API pricing before running evaluations.

## API reference

| Method | Endpoint | Success | Expected errors |
|---|---|---|---|
| GET | `/health` | 200 `{"status": "ok", "llm_configured": bool}` | - |
| POST | `/documents` (multipart field `file`) | 201 document | 400 empty/corrupt PDF · 409 duplicate · 413 > 20 MB · 415 not `.pdf` or no `%PDF-` header · 422 no extractable text |
| GET | `/documents` | 200 list of documents | - |
| GET | `/documents/{document_id}` | 200 document | 404 unknown ID |
| POST | `/search` | 200 ranked passages (no LLM call) | 404 unknown `document_id` · 422 invalid body or query without words |
| POST | `/ask` | 200 answer + citations + sources | 404 / 422 as `/search` · 502 provider error, refusal or malformed output · 503 no key, rejected credentials or rate limited · 504 provider timeout |

Application errors return `{"detail": "<message>"}`. Request-validation errors use
FastAPI's standard 422 format, where `detail` is a list of field errors.

### `POST /documents`

```bash
curl -F "file=@samples/lighthouses.pdf" http://127.0.0.1:8000/documents
```

```json
{"document_id": "1ec9c8cf136d4af29a9779d1cbc070c9", "filename": "lighthouses.pdf", "page_count": 3, "chunk_count": 9}
```

Document IDs are random and change on every run. Uploading byte-identical content again
returns 409 with the existing ID.

### `POST /search`

| Field | Type | Rules |
|---|---|---|
| `query` | string | 1-1000 characters after trimming; must contain at least one word |
| `top_k` | integer | 1-50, default 5 |
| `document_id` | string or null | Optional; restricts results to one document |

```bash
curl -X POST http://127.0.0.1:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "when was Boston Light automated", "top_k": 2}'
```

```json
{
  "query": "when was Boston Light automated",
  "top_k": 2,
  "document_id": null,
  "results": [
    {
      "chunk_id": "1ec9c8cf136d4af29a9779d1cbc070c9:p3:c1",
      "document_id": "1ec9c8cf136d4af29a9779d1cbc070c9",
      "filename": "lighthouses.pdf",
      "page_number": 3,
      "text": "services in many countries automated their stations step by step. In the United States, Boston Light was the last station to be automated, in 1998. ...",
      "score": 7.63
    },
    {"chunk_id": "...:p2:c0", "filename": "lighthouses.pdf", "page_number": 2, "score": 3.95, "...": "..."}
  ]
}
```

Scores are BM25 ranking values for the current corpus and query. They are not
probabilities, and they cannot be compared across corpora or settings.

### `POST /ask`

| Field | Type | Rules |
|---|---|---|
| `question` | string | 1-1000 characters after trimming |
| `top_k` | integer | 1-20, default 5: the number of passages sent to the LLM |
| `document_id` | string or null | Optional; restricts retrieval to one document |

```bash
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "When was Boston Light automated?"}'
```

| Response field | Meaning |
|---|---|
| `question` | The question as received (trimmed) |
| `answer` | The model's answer, or a fixed message when no passage was retrieved |
| `insufficient_information` | The model, or the empty retrieval, reports too little evidence |
| `grounded` | Backend check: sufficient information claimed **and** at least one valid citation |
| `llm_called` | `false` when retrieval found nothing and the LLM was skipped |
| `citations` | Validated citations: `source_id`, `chunk_id`, `document_id`, `filename`, `page_number` |
| `rejected_source_ids` | IDs the model returned that were not in its context |
| `sources` | Every passage sent to the model (the `/search` result fields plus `source_id`) |

## Demo walkthrough

1. **Generate sample PDFs.** These are the three evaluation documents, rendered from
   `evaluation/data/corpus/`:

   ```bash
   python -m evaluation.corpus samples
   ```

2. **Start the server** and upload all three files, either from `/docs` or with
   `curl -F "file=@samples/<name>.pdf" http://127.0.0.1:8000/documents`.

3. **Search without an LLM.** Run `{"query": "chain wear checker"}` against `/search`.
   The top result is page 2 of `bicycle_maintenance.pdf`.

4. **Ask a question.** For example: "How often should I turn the compost pile?" The
   response contains the answer, the citations (file and page) and every passage the
   model saw.

5. **Show a retrieval limitation.** Send the question `cleaning` to `/ask`.
   - The documents say "clean" and "polishing", and there is no stemming, so BM25
     retrieves no passage.
   - The response has `insufficient_information: true` and `llm_called: false`, even
     though the documents do describe cleaning.

   This is a retrieval failure that no prompt can fix.

6. **Show abstention.** Send `electric car battery range` to `/search` first.
   - BM25 returns a bicycle passage, because it contains "pressure range".
   - Asking the same question via `/ask` shows whether the LLM correctly reports
     insufficient information instead of answering from unrelated text. The answer
     review checks exactly this.

## Design decisions

- **Retrieval before generation.** BM25 retrieval was built and evaluated first, and
  `/search` remains independent of the LLM. If the right passage never reaches the
  model, the failure can be attributed to retrieval rather than to the prompt.
- **Page-bounded chunks.** A chunk never spans two pages, so each citation points to
  exactly one page. Chunk size is measured in words and set to 100 before the evaluation
  was run.
- **All-or-nothing ingestion.** The new BM25 index is built from the existing chunks
  plus the new ones *before* anything is committed. A failure at any step leaves the
  previous documents searchable and the failed one invisible.
- **In-memory state.** There is no database, which keeps the MVP small. The cost is that
  restarting the server loses all documents.
- **Stable source labels.**
  - The model sees passages labeled `S1`..`Sk` and returns those labels.
  - The backend maps each label to chunk ID, filename and page number.
  - Labels the model invents are rejected and reported, not silently dropped.
- **No evidence, no LLM call.** If retrieval returns nothing, the service answers
  "insufficient information" itself. This avoids an API call and avoids implying the
  model saw evidence.
- **Structured output plus validation.** The answer is requested as JSON through the
  API's schema-constrained output and then validated with Pydantic. Truncated, non-JSON
  or schema-violating output becomes a 502, never a half-parsed answer.
- **Provider isolation.** All Anthropic-specific code lives in `llm_client.py` behind a
  one-method `LLMClient` protocol. Tests use a fake client, and switching provider means
  replacing a single class.
- **Refusal fallback.** Requests opt into the API's server-side fallback
  (`fallbacks: "default"`). If a safety classifier declines, the request is retried on a
  fallback model instead of failing.
- **Prompt-injection mitigation.** Document text is HTML-escaped so it cannot close its
  `<source>` tag, and the system prompt says source text is untrusted data. This reduces
  the risk but does not eliminate it.

## Evaluation

Retrieval and answer generation are evaluated separately.

### Retrieval

```bash
python -m evaluation.run_eval
```

The command writes [`evaluation/results/report.md`](evaluation/results/report.md),
including per-query results.

**Dataset.** There are three short documents written for this project: home composting,
lighthouses and bicycle maintenance, 3 pages each (9 pages in total). They use
overlapping vocabulary on purpose ("light", "oil", "clean"). There are 30 queries:
6 keyword, 8 paraphrase, 9 specific detail, 3 ambiguous and 4 no-answer.

**Labels** are in [`evaluation/data/queries.json`](evaluation/data/queries.json). Each
relevant passage is stored as an exact evidence string on a given page. A retrieved
chunk counts as relevant if it contains at least half of that passage's words. Labeling
passages instead of chunk IDs keeps the labels valid when the chunk size changes.

**Metrics** are averaged over the 26 answerable queries:

- Recall@k: the fraction of labeled passages found in the top k.
- Precision@5: always divided by 5.
- MRR@10: the mean of 1/rank of the first relevant chunk.

The metric functions are tested against a hand-calculated example.

#### Chunk-size experiment

Only the chunk size changes. Corpus, queries, tokenizer and BM25 parameters
(k1 = 1.5, b = 0.75) are fixed.

| chunk size (words) | chunks | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@10 |
|---|---|---|---|---|---|---|
| 50 | 49 | 0.667 | 0.776 | 0.814 | 0.177 | 0.757 |
| **100 (default)** | 27 | 0.609 | 0.891 | 0.891 | 0.192 | 0.775 |
| 200 | 18 | 0.628 | 0.949 | 0.949 | 0.208 | 0.801 |

**Interpretation.**

- Recall@5 rises with chunk size, but Recall@1 does not. Larger chunks are more likely
  to contain the evidence somewhere in the top results; they do not make the first result
  more accurate.
- Part of the gain is mechanical. With 200-word chunks, the top 5 results cover 5 of 18
  chunks (28% of the corpus, about 1,000 words). With 50-word chunks they cover 5 of 49
  (10%, about 250 words).
- For `/ask`, larger chunks also mean more text sent to the LLM per request.
- Precision@5 is low by construction. Most queries have one relevant passage, so the
  best possible Precision@5 for those queries is 0.2.
- With 26 answerable queries, one query moves Recall by about 0.04, so none of these
  differences is statistically meaningful.
- The 100-word default was fixed before the experiment and was not tuned on this query
  set.

**By query type** (chunk size 100):

| Query type | Recall@5 | MRR@10 |
|---|---|---|
| keyword | 1.000 | 1.000 |
| paraphrase | 1.000 | 0.708 |
| ambiguous | 0.389 | 0.667 |

For paraphrases the relevant chunk is usually found, but often ranked below an
irrelevant one.

#### Observed retrieval failures

- **No stemming.** `cleaning` retrieves nothing, because the text says "clean" and
  "polishing". `oil` misses the "oily foods" passage.
- **BM25 cannot abstain.** Three of the four no-answer queries still retrieve passages.
  For example, `electric car battery range` matches "pressure range" on the bicycle page,
  with a higher score than some correct answers. A raw score threshold would not separate
  these reliably.
- **Spelling variants.** The corpus uses "tyre". "flat tire" still reaches the right
  page only through "flat" ("pinch flat").

### Answer quality (manual review)

```bash
python -m evaluation.answer_review run          # calls the real API: needs ANTHROPIC_API_KEY, costs money
# score the rubric columns in evaluation/results/answer_review.csv, then:
python -m evaluation.answer_review summarize evaluation/results/answer_review.csv
```

`run` asks 16 of the labeled queries (every query type, including 3 no-answer) through
the same `QAService` that `/ask` uses, with `top_k=5`. It writes a CSV with these
automatic signals:

| Signal | Meaning |
|---|---|
| `retrieval_hit` | The labeled evidence was among the passages sent to the model |
| `cited_evidence` | A validated citation points at the labeled evidence |
| `failure_hint` | `retrieval_failure`, `wrongly_abstained` or `answered_without_evidence` |

A reviewer then scores each row 1, 0.5 or 0, leaving a cell blank when it doesn't apply:

| Criterion | Question |
|---|---|
| correctness | Does the answer match the source documents? |
| groundedness | Is every important claim supported by the retrieved text? |
| citation_accuracy | Does every cited source support the claim it is cited for? |
| abstention | No-answer query: did it admit insufficient evidence? Answerable query with evidence retrieved: did it answer? |

**Status:** the review has not been run and scored yet, so no answer-quality numbers are
reported. Results will be added with the model name, the effort level and the sample
size.

## Testing

```bash
pytest
```

The suite has 148 tests. None needs an API key or a network connection. The LLM is
replaced by a fake client, and two tests run the real Anthropic SDK over a local mock
HTTP transport to verify request serialization and response parsing.

| Test file | Tests | Covers |
|---|---|---|
| `test_pdf_reader.py` | 9 | Page order and numbering, blank pages, missing/invalid/text-less PDFs |
| `test_chunking.py` | 12 | Normalization, chunk boundaries, short final chunk, content preservation, metadata |
| `test_tokenizer.py` | 4 | Case folding, punctuation, Vietnamese diacritics, Unicode normalization |
| `test_bm25.py` | 16 | Ranking on a known corpus, `top_k`, filters, empty query/index, tiny-corpus regression |
| `test_pipeline.py` | 8 | Multi-document ingestion, duplicates, failed ingestion leaves state unchanged |
| `test_api.py` | 22 | Upload/list/get/search endpoints and their validation and error codes |
| `test_prompts.py` | 6 | Source labeling, context format, tag-escaping against injection, schema |
| `test_llm_client.py` | 16 | Request shape, provider-error mapping, refusals, malformed output, real-SDK round trip |
| `test_qa_service.py` | 13 | Context sent to the model, citation validation, abstention, no-evidence path |
| `test_api_ask.py` | 16 | `/ask` responses, error status codes, `/search` never calling the LLM |
| `test_metrics.py` | 14 | Recall@k, Precision@k and MRR against a hand-calculated example |
| `test_evaluation.py` | 7 | Evidence-to-chunk mapping; every label resolves at each chunk size |
| `test_answer_review.py` | 5 | Review rows, failure hints, rubric summary |

## Debugging notes

1. **`rank_bm25` IDF on tiny corpora.** `BM25Okapi` computes
   `idf = log((N - n + 0.5) / (n + 0.5))`. For a 2-chunk corpus, a term found in one chunk
   gets `log(1) = 0`, so an exact match scores 0. For a 1-chunk corpus the IDF is
   negative, which produced a score of -0.27. A freshly uploaded one-page PDF would
   therefore rank nothing. The fix is a subclass that uses Lucene's always-positive
   `log(1 + (N - n + 0.5) / (n + 0.5))`. It is covered by
   `test_single_and_two_chunk_corpora_still_score_matches`.
2. **Evidence split across chunk boundaries.** The first relevance rule counted a chunk
   as relevant if it held at least 50% of the evidence *characters*. A unit test showed
   that evidence split evenly across two chunks matched neither one, because the
   separating space belongs to neither chunk. Counting *words*, the unit chunks are built
   from, guarantees that one of the two chunks qualifies.
3. **Missing credentials fail late.** The Anthropic SDK client can be constructed
   without credentials and only fails at request time, with a generic `TypeError`.
   Catching `TypeError` would hide real bugs. Instead, the app checks `ANTHROPIC_API_KEY`
   explicitly and turns a missing key into a clear 503, while `/search` keeps working.

## Limitations

- **Memory only.** Restarting the server loses every document.
- **Text-based PDFs only.** Scanned or image-only PDFs are rejected; there is no OCR.
  Multi-column layouts and tables follow PyMuPDF's reading order, which may be wrong.
- **Lexical retrieval.** There is no stemming, synonym or spelling handling. Answer
  quality is capped by retrieval quality.
- **Citations are checked for existence, not support.** `grounded` means the answer
  cites passages that were really retrieved. It does not prove that they support the
  claim, which is why the manual review exists.
- **Prompt injection is mitigated, not prevented.**
- **Vietnamese is tokenized into syllables** ("học sinh" -> `học`, `sinh`), not
  segmented words. Neither retrieval nor answers have been evaluated on Vietnamese text.
- **No incremental indexing.** The index is rebuilt over all chunks on every upload.
  This is fine for a handful of documents and does not scale.
- **Small evaluation set.** There are 26 answerable queries over 9 pages, written and
  labeled by the same person, so results describe this setup rather than general
  quality.
- **Worst-case `/ask` latency is about 3 minutes** (a 60 s timeout with two SDK retries)
  when the provider is unavailable.

## Roadmap

- Run and score the answer review, then publish the results alongside their limitations.
- Add dense embeddings and compare dense and hybrid retrieval against the BM25 baseline
  on the same query set.
- Add English stemming or lemmatization and Vietnamese word segmentation, and measure
  their effect.
- Persist documents and chunks (e.g. SQLite) so they survive restarts.
- Add a small web UI for upload, search and Q&A.

## Project layout

```text
app/
  api/main.py                   FastAPI app factory and routes
  api/schemas.py                request/response models
  ingestion/pdf_reader.py       PDF -> pages
  ingestion/chunking.py         pages -> chunks
  retrieval/tokenizer.py        shared tokenizer
  retrieval/bm25_index.py       BM25 ranking
  generation/llm_client.py      LLMClient protocol + Anthropic wrapper
  generation/prompts.py         system prompt, context format, answer schema
  generation/answer_generator.py
  services/qa_service.py        retrieval -> generation -> citation validation
  pipeline.py                   DocumentService (state, ingestion, search)
  config.py, errors.py
evaluation/
  data/corpus/*.txt             evaluation documents (rendered to PDF at run time)
  data/queries.json             labeled queries
  corpus.py                     corpus loader / sample PDF writer
  metrics.py                    Recall@k, Precision@k, MRR
  run_eval.py                   retrieval evaluation + chunk-size experiment
  answer_review.py              answer-review CSV and rubric summary
  results/report.md             generated retrieval report
tests/                          unit, integration and API tests
.env.example                    environment variable template
pyproject.toml                  dependencies and tool configuration
```
