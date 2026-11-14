"""Evaluate retrieval + calibrate the abstain threshold.
In-book queries: a 10-word span from a random sentence of a random chunk (self-supervised, so it is an
optimistic proxy). Out-of-book queries: general-knowledge questions that are not in the textbooks."""
import random, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kb import load_kb
from src.rag import Retriever, sentences, tok

OOB = ["Who was the first Mughal emperor?", "What is the capital of France?", "Who won the 2018 football world cup?",
 "How do I bake a chocolate cake?", "What is the price of bitcoin?", "Who wrote the play Hamlet?",
 "Which is the longest river in Africa?", "What is the population of Japan?", "Who is the prime minister of India?",
 "How to learn the guitar quickly?", "What is the best smartphone in 2026?", "When did the Second World War end?",
 "Who painted the Mona Lisa?", "How do I apply for a passport?", "What is the tallest building in Dubai?",
 "Which team won the cricket world cup in 2011?", "Translate good morning into French", "Write a poem about the sea",
 "What is the stock market?", "Who invented the telephone company AT&T?"]

random.seed(0)
kb = load_kb(); ret = Retriever(kb)
qs = []
for i in random.sample(range(len(kb)), 300):
    ss = sentences(kb.text[i])
    if not ss: continue
    w = random.choice(ss).split()
    if len(w) < 12: continue
    s = random.randint(0, len(w) - 10)
    qs.append((" ".join(w[s:s + 10]), i))
hit1 = hit5 = 0; in_conf = []
for q, i in qs:
    hits, conf = ret.search(q, k=5)
    ids = [h["chunk_id"] for h in hits]
    hit1 += ids[:1] == [i]; hit5 += i in ids; in_conf.append(conf)
out_conf = [ret.search(q, k=5)[1] for q in OOB]
res = dict(n_in=len(qs), recall_at_1=round(hit1 / len(qs), 3), recall_at_5=round(hit5 / len(qs), 3), thresholds={})
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    res["thresholds"][t] = dict(refuse_out_of_book=round(sum(c < t for c in out_conf) / len(out_conf), 2),
                                over_refuse_in_book=round(sum(c < t for c in in_conf) / len(in_conf), 2))
print(json.dumps(res, indent=1)); Path("eval_report.json").write_text(json.dumps(res, indent=1))
