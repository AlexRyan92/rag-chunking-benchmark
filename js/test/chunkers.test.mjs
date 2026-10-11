import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fixedChunks, sentenceSpans, paragraphSpans, pack, recursiveSpans,
  chunk, metrics, CHARS_PER_TOKEN,
} from '../src/chunkers.mjs';

const PARA = 'The quick brown fox jumps over the lazy dog. It was not amused! Was the dog even awake?';
const DOC = `${PARA}\n\n${PARA}\n\n${PARA}`;

function assertCoverage(text, spans) {
  // Spans must tile the text in order with no gaps.
  assert.equal(spans[0][0], 0);
  assert.equal(spans[spans.length - 1][1], text.length);
  for (let i = 1; i < spans.length; i++) {
    assert.equal(spans[i][0], spans[i - 1][1], `gap before span ${i}`);
  }
}

test('sentenceSpans and paragraphSpans tile the text exactly', () => {
  assertCoverage(DOC, sentenceSpans(DOC));
  assertCoverage(DOC, paragraphSpans(DOC));
});

test('fixedChunks: step = size - overlap, full coverage', () => {
  const out = fixedChunks('x'.repeat(1000), 400, 100);
  // Steps of 300; the window starting at 600 already reaches the end,
  // so no redundant tail chunk is emitted.
  assert.deepEqual(out.map((c) => c[0]), [0, 300, 600]);
  assert.equal(out[out.length - 1][1], 1000);
});

test('fixedChunks with zero overlap never duplicates', () => {
  const text = 'y'.repeat(997);
  const m = metrics(text, fixedChunks(text, 200, 0));
  assert.equal(m.duplicationPct, 0);
});

test('chunk() respects the 4 chars/token convention', () => {
  const spans = chunk('z'.repeat(8000), 'fixed', { sizeTokens: 200, overlapTokens: 0 });
  assert.equal(spans[0][1] - spans[0][0], 200 * CHARS_PER_TOKEN);
});

test('pack keeps chunks within size for packable spans', () => {
  const spans = chunk(DOC, 'sentence', { sizeTokens: 20, overlapTokens: 0 });
  for (const [a, b] of spans) {
    assert.ok(b - a <= 20 * CHARS_PER_TOKEN, `chunk of ${b - a} chars exceeds budget`);
  }
});

test('overlap produces measurable duplication, and more than requested (the benchmark finding)', () => {
  const noOver = metrics(DOC, chunk(DOC, 'sentence', { sizeTokens: 20, overlapTokens: 0 }));
  const withOver = metrics(DOC, chunk(DOC, 'sentence', { sizeTokens: 20, overlapTokens: 5 }));
  assert.equal(noOver.duplicationPct, 0);
  assert.ok(withOver.duplicationPct > 0);
  // Overlap is re-seeded as WHOLE sentences, so actual duplication
  // exceeds the naive 5/20 = 25% request. This is the published result
  // in miniature (17.5% requested -> up to 78.3% actual on SQuAD).
  assert.ok(withOver.duplicationPct > 25, `got ${withOver.duplicationPct}%`);
});

test('recursiveSpans splits oversized paragraphs by sentence, oversized sentences by word', () => {
  const longSentence = 'word '.repeat(300).trim() + '.';
  const text = `${PARA}\n\n${longSentence}`;
  const spans = recursiveSpans(text, 200);
  assert.ok(spans.length > 10, 'long sentence must shatter into words');
  assertCoverage(text, spans);
});

test('metrics: exact arithmetic on a crafted case', () => {
  // Two chunks of 400 chars over a 600-char text: 200 chars re-emitted.
  const text = 'a'.repeat(600);
  const m = metrics(text, [[0, 400], [200, 600]]);
  assert.equal(m.count, 2);
  assert.equal(m.avgTokens, 100);
  assert.equal(m.duplicationPct, 25); // 200 / 800
  assert.equal(m.embeddedTokens, 200);
});

test('published results files parse and carry the known findings', async () => {
  const { readFile } = await import('node:fs/promises');
  const r0 = JSON.parse(await readFile(new URL('../../data/results-overlap0.json', import.meta.url)));
  const r30 = JSON.parse(await readFile(new URL('../../data/results-overlap30.json', import.meta.url)));
  assert.equal(r0.config.questions, 800);
  assert.equal(r0.results.recursive['recall@1'], 0.7075);
  assert.equal(r0.results.fixed['recall@1'], 0.6112);
  assert.equal(r30.results.fixed.duplication_pct, 17.5);
  assert.equal(r30.results.recursive.duplication_pct, 78.3);
});

test('unknown strategy throws', () => {
  assert.throws(() => chunk('text', 'semantic'), RangeError);
});
