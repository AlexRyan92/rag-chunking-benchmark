# rag-chunking-benchmark

Most chunking advice is vibes. This is measured.

800 questions from SQuAD v1.1 against three chunking strategies, embedded with all-MiniLM-L6-v2, scored on answer recall@k. Seeded and deterministic: re-running the script reproduces every number below, byte for byte. This repo holds the benchmark script, the raw results, and the chunkers as a tested, zero-dependency JS package.

## The results

200-token chunks, zero overlap (identical embedding cost all three ways):

| Strategy | recall@1 | recall@3 | recall@5 | chunks | dup % |
| --- | --- | --- | --- | --- | --- |
| fixed | 61.1% | 80.1% | 87.1% | 818 | 0 |
| sentence | 66.9% | 85.6% | 90.5% | 925 | 0 |
| recursive | **70.8%** | **88.1%** | **91.4%** | 1010 | 0 |

Same chunks with the common "30 tokens of overlap" request:

| Strategy | recall@1 | dup % actual | embedded tokens |
| --- | --- | --- | --- |
| fixed | 65.8% | 17.5% | 189,596 |
| sentence | 70.3% | **44.6%** | 233,473 |
| recursive | 73.3% | **78.3%** | 287,882 |

Two findings worth your attention:

1. **Structure is free.** Recursive splitting beats fixed windows by 9.6 points of recall@1 at identical embedding cost. Respecting paragraph and sentence boundaries costs nothing and buys recall.
2. **Overlap is not what you asked for.** Requesting 30 tokens of overlap produced 17.5% duplication on fixed chunks but 78.3% on recursive, because overlap is re-seeded as whole sentences and paragraphs. The number hides inside a splitter default most people have never printed.

Full write-up: [Chunking Strategies, Measured](https://alexryan.tech/blog/chunking-strategy-benchmark.html). Interactive version: [Chunking Benchmark Lab](https://alexryan.tech/tools/chunking-benchmark-lab.html).

## Repo layout

```
bench/chunk_benchmark.py     the benchmark (python, sentence-transformers)
data/results-overlap0.json   raw results, zero overlap
data/results-overlap30.json  raw results, 30-token overlap
js/                          the chunkers as a zero-dependency npm package
```

## Reproduce the benchmark

```sh
pip install sentence-transformers numpy
curl -LO https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
python bench/chunk_benchmark.py dev-v1.1.json          # overlap 0
OVER=120 python bench/chunk_benchmark.py dev-v1.1.json # overlap 30 tokens
```

Runs on a laptop CPU in a few minutes. Seed is pinned (7); the SQuAD dataset itself is CC BY-SA 4.0 and is not redistributed here.

## Use the chunkers

```sh
cd js && npm test                 # 10 tests, node:test, no deps
npx chunk-lab corpus.txt --size 200 --overlap 30
npx chunk-lab corpus.txt --strategy recursive --jsonl > chunks.jsonl
```

```js
import { chunk, metrics } from 'rag-chunking-benchmark';

const spans = chunk(text, 'recursive', { sizeTokens: 200, overlapTokens: 0 });
metrics(text, spans);  // { count, avgTokens, duplicationPct, embeddedTokens }
```

Chunkers are char-span based (`[start, end]` pairs into your original text), so nothing is copied, and the duplication metric measures what overlap actually costs at the embedding API.

## Honest limits

One corpus of Wikipedia-style prose, one small embedding model, a substring-match recall metric, and ~4 chars/token estimates. Tables, code, and legal numbering will behave differently and generally favor format-aware splitters. This tells you where to start tuning, not where to stop: swap your own corpus into the script in about ten lines.

## License

Code MIT © [Alexander Ryan](https://alexryan.tech). Results data CC BY 4.0. SQuAD v1.1 belongs to its authors (CC BY-SA 4.0) and is not included.
