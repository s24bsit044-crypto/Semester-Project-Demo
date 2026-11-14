import streamlit as st
from src.kb import load_kb, load_qa
from src.rag import Retriever, ask, cite
import os

st.set_page_config(page_title="BTBB Textbook Tutor", page_icon="📘", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1100px}
.hero{background:linear-gradient(120deg,#0F766E,#115E59);color:#fff;padding:1.4rem 1.6rem;border-radius:16px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.7rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.9}
.cite{display:inline-block;background:#E6F4F1;color:#0F766E;border-radius:999px;padding:.15rem .7rem;font-size:.8rem;margin:.15rem .25rem .15rem 0}
.ans{background:#F1F5F4;border-left:5px solid #0F766E;padding:1rem 1.2rem;border-radius:10px}
.warn{background:#FFF7E6;border-left:5px solid #D97706;padding:1rem 1.2rem;border-radius:10px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building the textbook index (first run only)…")
def get_index():
    kb = load_kb()
    return kb, Retriever(kb)


@st.cache_data
def sample_questions():
    qa = load_qa()
    qa = qa[qa.qa_type == "concept_explanation"]
    return qa.groupby("subject").question.apply(lambda s: s.drop_duplicates().head(4).tolist()).to_dict()


kb, ret = get_index()
llm_on = bool(os.getenv("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("Filters")
    subject = st.selectbox("Subject", ["All"] + sorted(kb.subject.unique()))
    sub = kb if subject == "All" else kb[kb.subject == subject]
    grade = st.selectbox("Grade", ["All"] + [str(g) for g in sorted(sub.grade.unique())])
    length = st.radio("Answer length", ["Short", "Detailed", "Long"], index=1,
                      help="Long = full exam-style answer with sections, built from up to 8 textbook passages.")
    thr = st.slider("Strictness (higher = refuse more)", 0.0, 0.9, 0.5, 0.05,
                    help="If the best passage matches the question weakly, the tutor says it could not find it.")
    lang = st.selectbox("Answer language", ["English", "Urdu", "Hindi"], disabled=not llm_on,
                        help="Needs an ANTHROPIC_API_KEY. The books themselves are in English.")
    use_llm = st.toggle("AI-written answer", value=llm_on, disabled=not llm_on)
    st.caption("AI mode: on" if llm_on else "AI mode off — showing extracted textbook sentences. "
               "Set ANTHROPIC_API_KEY to enable.")
    st.divider()
    st.caption(f"📚 {len(kb):,} textbook pages indexed · BTBB Grades 9–12")

st.markdown('<div class="hero"><h1>📘 BTBB Textbook Tutor</h1>'
            '<p>Ask a question from your Balochistan Board textbook — every answer shows the exact page it came from.</p></div>',
            unsafe_allow_html=True)

if "q" not in st.session_state:
    st.session_state.q = ""
ex = sample_questions()
if subject != "All" and subject in ex:
    st.caption("Try one:")
    cols = st.columns(2)
    for i, q in enumerate(ex[subject]):
        if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.q = q

q = st.text_area("Your question (long, multi-part questions are fine)", key="q", height=100,
                 placeholder="e.g. Explain Newton's second law of motion with formula, example and applications.")
if q.strip():
    with st.spinner("Searching the textbooks…"):
        res = ask(ret, q, subject, grade, thr=thr, use_llm=use_llm, lang=lang, length=length)
    if res.abstained:
        st.markdown(f'<div class="warn">🔎 {res.answer}</div>', unsafe_allow_html=True)
    else:
        st.subheader("Answer")
        with st.container(border=True):
            st.markdown(res.answer)
        if res.mode == "extractive":
            st.caption("Extracted directly from the textbook (OCR text may contain small errors). "
                       "Enable AI mode for a fully written long answer.")
        st.markdown("".join(f'<span class="cite">[{h["rank"]}] {cite(h)}</span>' for h in res.hits),
                    unsafe_allow_html=True)
    st.caption(f"Match confidence: {res.conf:.0%}")
    if res.hits:
        st.subheader("Source passages")
        for h in res.hits:
            with st.expander(f"[{h['rank']}] {cite(h)}"):
                st.write(h["text"])
                st.caption(f"Book: {h['book']} · ID: {h['sample_id']}")
