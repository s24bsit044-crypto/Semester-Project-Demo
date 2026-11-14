"""Load the four BTBB textbook datasets into one unified knowledge base."""
import re
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
NOISE = re.compile(r"scanned\s+with\s+camscanner|\${2,}", re.I)


def clean(t: str) -> str:
    t = NOISE.sub(" ", str(t))
    return re.sub(r"\s+", " ", t).strip()


def _grade(v) -> int:
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else 0


def load_kb() -> pd.DataFrame:
    rows = []
    p = pd.read_csv(DATA / "physics.csv")
    for r in p.itertuples():
        rows.append(("Physics", _grade(r.grade_level), r.unit_title, int(r.pdf_page_number),
                     r.content_type, r.text, r.sample_id, r.book_source))
    b = pd.read_csv(DATA / "biology.csv")
    for r in b.itertuples():
        rows.append(("Biology", int(r.grade_num), r.chapter_or_unit, int(r.page_number),
                     r.section_type, r.clean_text, r.sample_id, r.book_title))
    c = pd.read_csv(DATA / "chemistry.csv")
    for r in c.itertuples():
        rows.append(("Chemistry", _grade(r.grade_level), r.chapter_detected, int(r.page_number),
                     r.content_type, r.text, r.sample_id, r.source_file))
    m = pd.read_csv(DATA / "mathematics.csv")
    # Math rows are small blocks: merge blocks of the same page into one chunk.
    for (unit, page), g in m.groupby(["unit_name", "page_number"], sort=False):
        g = g.sort_values("block_index")
        rows.append(("Mathematics", 9, unit, int(page), g.content_type.iloc[0],
                     " ".join(g.cleaned_text.astype(str)), g.sample_id.iloc[0], g.source_document.iloc[0]))
    kb = pd.DataFrame(rows, columns=["subject", "grade", "chapter", "page", "kind",
                                     "text", "sample_id", "book"])
    kb["text"] = kb["text"].map(clean)
    kb = kb[kb.text.str.len() >= 80].reset_index(drop=True)
    kb["chunk_id"] = kb.index
    return kb


def load_qa() -> pd.DataFrame:
    return pd.read_csv(DATA / "qa_dataset.csv")
