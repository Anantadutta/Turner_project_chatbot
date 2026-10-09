# Spiderman Company - Document-Grounded Chatbot (RAG)

An internal chatbot that answers employee questions **only from company documents**, including text inside images, charts, tables and scanned pages. If the answer is not in the documents, it says so instead of guessing.

Test document: `data/superx.pdf` (131 pages, ~247,000 characters, 522 embedded images).

## How it works

    PDF --+-- TEXT:   extract -> clean -> chunk (800 chars, 150 overlap) --+
          |                                                               +--> index (embeddings + keyword index)
          +-- VISUAL: find picture pages -> render PNG -> Gemini describes -+

    Question -> hybrid search (keyword BM25 + semantic embeddings) -> top 6 chunks
             -> Gemini answers ONLY from those chunks (+ real page images) -> answer with page citations

| Step | File | What it does |
|---|---|---|
| 1 | `ingest_text.py` | Extracts text page by page, removes repeated headers, splits into 427 overlapping chunks |
| 2 | `ingest_visuals.py` | Detects pages that are mainly visual (scanned pages, charts, vector Gantt charts, large images), renders them as PNG and asks Gemini to transcribe all text, numbers and tables |
| 3 | `build_index.py` | Creates Gemini embeddings (cached) for all text and image-description chunks |
| 4 | `app.py` | Streamlit chat UI with hybrid retrieval, grounded answers, page citations and source thumbnails |

## Key design decisions
- **Two pipelines (text + visual):** about 50 of the 131 pages contain almost no extractable text; their content exists only as pictures. Charts drawn as vector graphics (e.g. the manning chart on page 48) are not images at all, so whole pages are rendered instead of only extracting embedded images.
- **Grounding:** strict system prompt (answer only from context, otherwise say not found), temperature 0, mandatory page citations, and a retrieval check that refuses before calling the model when nothing relevant is found.
- **Hybrid search:** keyword search (BM25, runs locally) catches exact terms like "L1" or "FWT"; semantic embeddings catch paraphrased questions. If the embedding API is unavailable, the bot falls back to keyword search automatically.
- **Images at answer time:** for the top visual results, the actual page image is sent with the question so exact numbers can be read.
- **Free-tier friendly:** embeddings and image descriptions are cached, re-runs skip completed work, and there is automatic model fallback when one model is busy.

## How to run
1. `pip install -r requirements.txt`
2. Create a file named `.env` containing: `GEMINI_API_KEY=your_key_here`
3. `streamlit run app.py`

The `index/`, `images/` and `*_chunks.json` files are included, so the chatbot runs directly. To rebuild everything from the PDF: `python ingest_text.py`, then `python ingest_visuals.py`, then `python build_index.py`.

## Example questions
- Who is the proposal addressed to?
- What are the L1 to L5 commissioning levels? (answer comes from a diagram on page 17)
- What is the total project fee? (answer comes from a chart on page 48)
- What is the leave policy? (should reply: not found in the company documents)

## Limitations and future improvements
- Very small or blurry text inside photos may be missed by the vision model.
- Tables lose some structure when extracted as plain text.
- Gemini free-tier limits restrict how many pages can be described per day.
- Improvements: crop and describe each image separately, add a re-ranking model, use a vector database (FAISS/Chroma) for larger document sets, and add an automated evaluation set.
