"""Chunking strategy benchmark: fixed vs sentence-aware vs recursive.

Corpus: SQuAD v1.1 dev articles, each article's paragraphs concatenated
into one document. For every question we embed the query, retrieve
top-k chunks by cosine similarity (all-MiniLM-L6-v2), and score
answer recall@k: whether any gold answer string appears in a
retrieved chunk. Budgets match the visualizer defaults: 200-token
chunks (approximated as 800 chars) with 30-token (120-char) overlap.
"""
import json, re, random, sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

import os
SIZE_C = 800                                  # ~200 tokens
OVER_C = int(os.environ.get("OVER", 120))     # ~30 tokens; OVER=0 for the no-overlap run
N_ARTICLES, Q_PER_ARTICLE, SEED = 20, 40, 7


def fixed_chunks(text, size=SIZE_C, over=OVER_C):
    out, step, i = [], max(1, size - over), 0
    while i < len(text):
        out.append((i, min(i + size, len(text))))
        if i + size >= len(text):
            break
        i += step
    return out


def _spans(text, a, b, pattern):
    spans, start = [], a
    for m in re.finditer(pattern, text[a:b]):
        end = a + m.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < b:
        spans.append((start, b))
    return spans


def sentence_spans(text, a=0, b=None):
    return _spans(text, a, len(text) if b is None else b, r'[.!?]+(?=\s|$)|\n+')


def paragraph_spans(text):
    return _spans(text, 0, len(text), r'\n{2,}')


def word_spans(text, a, b):
    return _spans(text, a, b, r'\s+')


def pack(text, spans, size=SIZE_C, over=OVER_C):
    chunks, cur, cur_len, seeded = [], [], 0, 0

    def emit():
        nonlocal cur, cur_len, seeded
        if len(cur) <= seeded:
            cur, cur_len, seeded = [], 0, 0
            return
        chunks.append((cur[0][0], cur[-1][1]))
        keep, kl = [], 0
        for sp in reversed(cur):
            if kl >= over:
                break
            keep.insert(0, sp)
            kl += sp[1] - sp[0]
        if over > 0 and kl < size:
            cur, cur_len, seeded = keep, kl, len(keep)
        else:
            cur, cur_len, seeded = [], 0, 0

    for sp in spans:
        ln = sp[1] - sp[0]
        if ln > size:
            emit()
            cur, cur_len, seeded = [], 0, 0
            for c in fixed_chunks(text[sp[0]:sp[1]], size, over):
                chunks.append((sp[0] + c[0], sp[0] + c[1]))
            continue
        if cur_len + ln > size and cur:
            emit()
        cur.append(sp)
        cur_len += ln
    emit()
    return chunks


def recursive_spans(text, size=SIZE_C):
    out = []
    for p in paragraph_spans(text):
        if p[1] - p[0] <= size:
            out.append(p)
            continue
        for s in sentence_spans(text, p[0], p[1]):
            if s[1] - s[0] <= size:
                out.append(s)
            else:
                out.extend(word_spans(text, s[0], s[1]))
    return out


def norm(s):
    return re.sub(r'\s+', ' ', s.lower()).strip()


def main():
    data = json.load(open(sys.argv[1]))['data']
    random.seed(SEED)
    articles = data[:N_ARTICLES]

    docs, questions = [], []
    for art in articles:
        paras = [p['context'] for p in art['paragraphs']]
        doc = '\n\n'.join(paras)
        di = len(docs)
        docs.append(doc)
        qs = [(q['question'], [a['text'] for a in q['answers']], di)
              for p in art['paragraphs'] for q in p['qas']]
        random.shuffle(qs)
        questions.extend(qs[:Q_PER_ARTICLE])

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([q[0] for q in questions], batch_size=64,
                         normalize_embeddings=True, show_progress_bar=False)

    strategies = {
        'fixed': lambda t: fixed_chunks(t),
        'sentence': lambda t: pack(t, sentence_spans(t)),
        'recursive': lambda t: pack(t, recursive_spans(t)),
    }
    results = {}
    for name, fn in strategies.items():
        texts, owner = [], []
        dup_chars = total_chars = 0
        sizes = []
        for di, doc in enumerate(docs):
            chunks = fn(doc)
            prev_end = 0
            for (a, b) in chunks:
                texts.append(doc[a:b])
                owner.append(di)
                sizes.append((b - a) / 4)
                dup_chars += max(0, min(prev_end, b) - a) if a < prev_end else 0
                prev_end = b
            total_chars += len(doc)
        c_emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False)
        owner_arr = np.array(owner)
        hits = {k: 0 for k in (1, 3, 5)}
        for qi, (q, answers, di) in enumerate(questions):
            idx = np.where(owner_arr == di)[0]
            sims = c_emb[idx] @ q_emb[qi]
            top = idx[np.argsort(-sims)[:5]]
            na = [norm(a) for a in answers]
            for k in (1, 3, 5):
                if any(a in norm(texts[t]) for t in top[:k] for a in na):
                    hits[k] += 1
        n = len(questions)
        results[name] = {
            'recall@1': round(hits[1] / n, 4),
            'recall@3': round(hits[3] / n, 4),
            'recall@5': round(hits[5] / n, 4),
            'chunks': len(texts),
            'avg_tokens': round(float(np.mean(sizes)), 1),
            'duplication_pct': round(dup_chars / total_chars * 100, 1),
            'embedded_tokens': int(sum(sizes)),
        }
        print(name, results[name], flush=True)

    out = {
        'config': {'articles': len(docs), 'questions': len(questions),
                   'chunk_tokens': SIZE_C // 4, 'overlap_tokens': OVER_C // 4,
                   'model': 'all-MiniLM-L6-v2', 'dataset': 'SQuAD v1.1 dev',
                   'seed': SEED},
        'results': results,
    }
    json.dump(out, open('benchmark_results.json', 'w'), indent=2)
    print('DONE')


if __name__ == '__main__':
    main()
