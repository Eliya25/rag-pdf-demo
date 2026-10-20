# RAG PDF Demo

A Streamlit demo for asking questions about text-based PDFs. It keeps PDF parsing, page-aware chunking, MiniLM embeddings, and custom vector search in the app, then uses Gemini to generate an answer from retrieved excerpts.

## Public demo controls and limits

The snapshot is locked to Gemini mode and `gemini-3.5-flash-lite`. `GEMINI_DEMO_ENABLED` is **off unless the owner explicitly sets it to `true`** in Streamlit Secrets. The owner must privately verify the exact project's billing tier, model/region availability, and quota before enabling requests. Quota exhaustion must not be handled by enabling billing or changing providers/models.

The demo applies these per-process or per-session limits: 5 MiB PDF, 25 pages, 100,000 extracted characters, 100 chunks, 1,000 characters per question, four retrieved chunks, four recent history messages, 10 admitted questions per session, a 20-second interval per session, 20 Gemini generation attempts per rolling process hour, one concurrent indexing operation, one concurrent generation, and 256 output tokens.

Provider failures do not trigger automatic retries or a different provider/model.

Counters and concurrency guards are held in process memory. They reset on app restart and do not coordinate multiple processes or replicas. They are best-effort abuse controls, not a durable quota or a guarantee of zero cost or availability. An open app can have its shared quota exhausted by visitors. Prefer invite-only access for an initial pilot.

## Data handling

PDFs are processed in the running app. The original PDF bytes are not sent to Gemini. Gemini receives the current question, selected recent chat, system instructions, and retrieved document excerpts. Do not upload confidential or personal documents. Document text can contain prompt injection; instructions and quoting do not guarantee the model will ignore it.

The app uses a shared Gemini API credential configured by the operator. Public use can exhaust available quota or make the app unavailable. Do not enable billing to work around quota limits.

## Run locally

Use Python 3.12. Install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run main.py
```

Configure `GEMINI_API_KEY` and `GEMINI_DEMO_ENABLED = false` in Streamlit Secrets. Only after the owner has completed the account and quota checks should the owner set the flag to `true`. The snapshot marker enforces public mode; no provider/model selection is needed or accepted. Never commit credentials. The first indexed PDF downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face.

## Architecture

```text
PDF -> local extraction/chunking -> MiniLM embeddings -> custom in-memory VectorIndex
question -> local retrieval -> question + selected history + excerpts -> Gemini
```

The index and chat are in memory and are not persistent. The app is intended as a small demo, not a high-volume or confidential-document service. PDF parser time and peak memory cannot be hard-bounded by these limits. Prompt instructions do not guarantee resistance to malicious document text.

## Attribution

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; see `THIRD_PARTY_NOTICES.md` and the model card. The weights are downloaded at runtime and are not included in this snapshot.

This public snapshot is licensed under MIT; see `LICENSE`.
