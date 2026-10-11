#!/usr/bin/env node
/**
 * chunk-lab CLI: run the benchmark's chunkers on your own document.
 *
 *   chunk-lab corpus.txt                          # all three strategies, stats
 *   chunk-lab corpus.txt --size 300 --overlap 30
 *   chunk-lab corpus.txt --strategy recursive --jsonl > chunks.jsonl
 */
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { chunk, metrics, STRATEGIES } from './chunkers.mjs';

const { values: v, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    size: { type: 'string', default: '200' },
    overlap: { type: 'string', default: '0' },
    strategy: { type: 'string' },
    jsonl: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
});

if (v.help || positionals.length === 0) {
  console.log(`usage: chunk-lab <file> [options]

  --size      chunk size in tokens          (default: 200)
  --overlap   overlap in tokens             (default: 0)
  --strategy  fixed | sentence | recursive  (default: all three, stats only)
  --jsonl     emit chunks as JSONL (requires --strategy)`);
  process.exit(positionals.length === 0 && !v.help ? 1 : 0);
}

const text = readFileSync(positionals[0], 'utf8');
const opts = { sizeTokens: Number(v.size), overlapTokens: Number(v.overlap) };

if (v.jsonl) {
  if (!v.strategy) { console.error('--jsonl requires --strategy'); process.exit(1); }
  const spans = chunk(text, v.strategy, opts);
  for (let i = 0; i < spans.length; i++) {
    const [a, b] = spans[i];
    process.stdout.write(JSON.stringify({
      id: `${v.strategy}-${i}`, start: a, end: b, text: text.slice(a, b),
    }) + '\n');
  }
  process.exit(0);
}

const strategies = v.strategy ? [v.strategy] : STRATEGIES;
console.log(`${positionals[0]}: ${text.length.toLocaleString()} chars, size ${v.size} tokens, overlap ${v.overlap} tokens\n`);
console.log('strategy    chunks   avg tokens   duplication   embedded tokens');
for (const s of strategies) {
  const m = metrics(text, chunk(text, s, opts));
  console.log(
    s.padEnd(12) +
    String(m.count).padStart(6) +
    String(m.avgTokens).padStart(13) +
    (m.duplicationPct + '%').padStart(14) +
    m.embeddedTokens.toLocaleString().padStart(18),
  );
}
