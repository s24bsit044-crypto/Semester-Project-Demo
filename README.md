# 📘 BTBB Textbook Tutor (ED-05)

Retrieval-augmented tutor for Balochistan Textbook Board books (Physics, Chemistry, Biology, Mathematics; Grades 9–12).
Every answer cites subject, grade, chapter and page. Weak matches are refused instead of guessed.

## How it works
`data/*.csv` → unified knowledge base (`src/kb.py`) → hybrid index: BM25 + word TF-IDF + char TF-IDF fused with RRF (`src/rag.py`)
→ extractive answer (default) or optional Claude answer grounded only in retrieved pages.
The index is built automatically at app start (~10 s, cached) — there is no model to download and no GPU needed.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Docker
```bash
docker compose up --build        # or: docker build -t btbb-tutor . && docker run -p 8501:8501 btbb-tutor
```
Open http://localhost:8501

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. share.streamlit.io → **New app** → pick the repo, branch `main`, main file `app.py`.
3. (Optional AI answers) App → Settings → Secrets: `ANTHROPIC_API_KEY = "sk-ant-..."`

## Evaluate
`python scripts/evaluate.py` — recall@k on self-supervised queries and abstention calibration (writes `eval_report.json`).

## Notes
- `data/qa_dataset.csv` is used for the example-question buttons. The two QA files you uploaded were identical, so only one is included.
- Textbook text is OCR-noisy (e.g. "Scanned with CamScanner" artefacts). Source passages are shown so students can verify.
- Board books are copyrighted: keep the repo **private** unless you have permission to publish the text.
