# rag-chunking-benchmark

The three chunkers from a real 800-question RAG chunking benchmark, as a zero-dependency library and CLI. The published results ride along in `data/`.

The headline from the benchmark: on 800 SQuAD questions, structure-aware recursive splitting found the right chunk at rank one 70.8% of the time against 61.1% for fixed-size, at identical embedding cost. And a requested 15% overlap duplicated 78% of the corpus under span-granularity packing. Full numbers, method, and caveats: [the benchmark essay](https://alexryan.tech/blog/chunking-strategy-benchmark.html).

## Install

```bash
npm i rag-chunking-benchmark      # library
npx rag-chunking-benchmark --help # CLI is chunk-lab
```

## CLI

```bash
chunk-lab corpus.txt                               # all three strategies, stats table
chunk-lab corpus.txt --size 300 --overlap 30       # your budget
chunk-lab corpus.txt --strategy recursive --jsonl  # emit chunks as JSONL
```

The stats include a duplication percentage. If it surprises you, that is the point: span-granularity overlap overshoots its budget, and this number is how you catch it before paying to embed a second copy of your index.

## Library

```js
import { chunk, metrics, STRATEGIES } from 'rag-chunking-benchmark';

const spans = chunk(text, 'recursive', { sizeTokens: 200, overlapTokens: 0 });
// spans are [start, end] offsets into text, so chunks never drift from the source
const { chunks, avgTokens, duplicationPct } = metrics(text, spans);
```

Strategies: `fixed`, `sentence`, `recursive` (paragraphs first, sentences when a paragraph is too big, raw splits last). Token counts use the same 4-characters-per-token estimate everywhere, stated rather than hidden.

## The data

`data/results-overlap0.json` and `data/results-overlap30.json` are the published benchmark results (CC BY 4.0, attribution: alexryan.tech). The Python benchmark script, method, and corpus notes live in the [repo root](https://github.com/AlexRyan92/rag-chunking-benchmark). SQuAD itself is not redistributed.

## Provenance

Extracted from the chunkers behind the [Chunking Benchmark Lab](https://alexryan.tech/tools/chunking-benchmark-lab.html) and the [RAG Chunking Visualizer](https://alexryan.tech/tools/chunk-visualizer.html), byte-identical logic, with tests pinning the published numbers. By [Alexander Ryan](https://alexryan.tech). MIT.
