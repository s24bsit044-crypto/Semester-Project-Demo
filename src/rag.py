"""Hybrid retriever (BM25 + word TF-IDF + char TF-IDF, fused with RRF) with
citation and abstention. Pure CPU, no model downloads -> deploys anywhere."""
import os
import re
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RRF_K = 60
STOP = set("a an the of in on to is are was were be and or for with by as at from that this it its what which who how why "
           "define explain describe state write give difference between".split())


def tok(t: str):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP]


class Retriever:
    def __init__(self, kb):
        self.kb = kb
        docs = (kb.subject + " " + kb.chapter.astype(str) + " " + kb.text).tolist()
        self.cv = CountVectorizer(tokenizer=tok, lowercase=False, token_pattern=None)
        tf = self.cv.fit_transform(docs).tocsc().astype(np.float32)
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.tf = tf.tocsr()
        self.dl = np.asarray(self.tf.sum(1)).ravel()
        self.avgdl = self.dl.mean()
        self.w = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None,
                                 ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.W = self.w.fit_transform(docs)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        self.C = self.c.fit_transform(docs)

    def _bm25(self, q, k1=1.5, b=0.75):
        ids = [self.cv.vocabulary_[t] for t in tok(q) if t in self.cv.vocabulary_]
        s = np.zeros(self.tf.shape[0], np.float32)
        if not ids:
            return s
        sub = self.tf[:, ids].toarray()
        norm = k1 * (1 - b + b * self.dl / self.avgdl)[:, None]
        return ((sub * (k1 + 1) / (sub + norm)) * self.idf[ids]).sum(1)

    def search(self, q, k=5, subject=None, grade=None):
        mask = np.ones(len(self.kb), bool)
        if subject and subject != "All":
            mask &= (self.kb.subject == subject).to_numpy()
        if grade and grade != "All":
            mask &= (self.kb.grade == int(grade)).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return [], 0.0
        s_bm = self._bm25(q)[idx]
        s_w = (self.W[idx] @ self.w.transform([q]).T).toarray().ravel()
        s_c = (self.C[idx] @ self.c.transform([q]).T).toarray().ravel()
        fused = np.zeros(len(idx))
        for s in (s_bm, s_w, s_c):
            fused[np.argsort(-s)] += 1 / (RRF_K + np.arange(1, len(idx) + 1))
        order = np.argsort(-fused)[:k]
        # confidence: share of query terms found in the best chunk (0..1) blended with char cosine
        qt = set(tok(q))
        best = " ".join(self.kb.text.iloc[idx[j]] for j in order[:3]).lower()
        cover = (sum(t in best for t in qt) / len(qt)) if qt else 0.0
        conf = 0.6 * cover + 0.4 * min(1.0, float(s_c[order[0]]) * 3)
        hits = [dict(rank=i + 1, score=float(fused[j]), **self.kb.iloc[idx[j]].to_dict())
                for i, j in enumerate(order)]
        return hits, float(conf)


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", t) if len(s.strip()) > 25]


LENGTHS = {
    "Short": dict(k=3, sents=3, tokens=400, style="Give a short, exam-ready answer in 2-4 sentences."),
    "Detailed": dict(k=5, sents=7, tokens=1000,
                     style="Give a clear, detailed answer in 1-3 paragraphs with the key definition, explanation and any formula."),
    "Long": dict(k=8, sents=16, tokens=2200,
                 style=("Write a LONG, complete exam answer. Use these headed sections where relevant: Introduction / "
                        "Definition, Explanation, Formula or Laws (with symbols explained), Worked example, Applications, "
                        "Key points to remember. If the question has several parts (a, b, c...), answer every part in order.")),
}


def extractive(q, hits, n=3):
    """Pick the best n sentences from the top passages, kept in textbook order, grouped by page."""
    qt = set(tok(q))
    top_hits = hits[: 2 if n <= 3 else (3 if n <= 7 else 5)]
    cand = [(s, h) for h in top_hits for s in sentences(h["text"])]
    if not cand:
        return hits[0]["text"][:400], hits[0]
    sc = [len(qt & set(tok(s))) / (1 + 0.02 * len(s.split())) for s, _ in cand]
    chosen = sorted(np.argsort(sc)[::-1][:n])
    if n <= 3:
        return " ".join(cand[i][0] for i in chosen), cand[int(np.argmax(sc))][1]
    out, last = [], None
    for i in chosen:
        h = cand[i][1]
        if h["chunk_id"] != last:
            out.append(f"\n**{h['chapter']} (page {h['page']})**\n")
            last = h["chunk_id"]
        out.append("- " + cand[i][0])
    return "\n".join(out).strip(), cand[int(np.argmax(sc))][1]


def cite(h):
    return f"{h['subject']} · Grade {h['grade']} · {h['chapter']} · p. {h['page']}"


def llm_answer(q, hits, lang="English", length="Detailed"):
    """Optional grounded answer via the Anthropic API (set ANTHROPIC_API_KEY)."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    cfg = LENGTHS[length]
    ctx = "\n\n".join(f"[{i+1}] ({cite(h)}) {h['text'][:1800]}" for i, h in enumerate(hits))
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=cfg["tokens"],
        system=("You are a revision tutor for Balochistan Textbook Board (BTBB) books. Use ONLY the numbered passages "
                "(they are noisy OCR; fix obvious OCR errors) and do not add facts that are not in them. "
                f"{cfg['style']} Cite passages like [1]. Format with Markdown. Answer in {lang}. "
                "If the passages do not contain the answer, reply exactly NOT_IN_BOOK."),
        messages=[{"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q}"}])
    return msg.content[0].text.strip()


@dataclass
class Result:
    answer: str
    abstained: bool
    hits: list
    conf: float
    mode: str


def ask(ret, q, subject=None, grade=None, k=None, thr=0.50, use_llm=True, lang="English", length="Detailed"):
    cfg = LENGTHS[length]
    hits, conf = ret.search(q, k=k or cfg["k"], subject=subject, grade=grade)
    if not hits or conf < thr:
        return Result("I could not find this in the selected textbooks. Try other keywords, or ask your teacher.",
                      True, hits, conf, "abstain")
    if use_llm:
        try:
            out = llm_answer(q, hits, lang, length)
            if out:
                if "NOT_IN_BOOK" in out:
                    return Result("I could not find this in the selected textbooks.", True, hits, conf, "llm")
                return Result(out, False, hits, conf, "llm")
        except Exception as e:  # fall back silently to extractive
            print("LLM error:", e)
    ans, _ = extractive(q, hits, n=cfg["sents"])
    return Result(ans, False, hits, conf, "extractive")
