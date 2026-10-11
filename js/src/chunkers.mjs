/**
 * The three chunking strategies from the benchmark, as a dependency-free
 * ES module. Char-span based: every chunker returns [start, end] pairs
 * into the original text, so nothing is ever copied or mutated.
 *
 * Tokens are approximated at 4 characters throughout, matching the
 * benchmark script, so results are comparable across the repo.
 */

export const CHARS_PER_TOKEN = 4;
export const estimateTokens = (chars) => Math.max(1, Math.round(chars / CHARS_PER_TOKEN));

/** Sliding fixed-size windows with overlap. */
export function fixedChunks(text, sizeC, overC) {
  const out = [];
  const step = Math.max(1, sizeC - overC);
  for (let i = 0; i < text.length; i += step) {
    out.push([i, Math.min(i + sizeC, text.length)]);
    if (i + sizeC >= text.length) break;
  }
  return out;
}

function spansBy(text, a, b, re) {
  const spans = [];
  let start = a, m;
  re.lastIndex = a;
  while ((m = re.exec(text)) && m.index < b) {
    const end = Math.min(m.index + m[0].length, b);
    if (end > start) spans.push([start, end]);
    start = end;
  }
  if (start < b) spans.push([start, b]);
  return spans;
}

export const sentenceSpans = (text, a = 0, b = text.length) => spansBy(text, a, b, /[.!?]+(?=\s|$)|\n+/g);
export const paragraphSpans = (text) => spansBy(text, 0, text.length, /\n{2,}/g);
export const wordSpans = (text, a, b) => spansBy(text, a, b, /\s+/g);

/**
 * Pack structural spans into chunks up to sizeC chars, re-seeding each
 * new chunk with trailing spans worth ~overC chars of overlap.
 * Spans larger than sizeC fall back to fixed windows.
 */
export function pack(text, spans, sizeC, overC) {
  const chunks = [];
  let cur = [], curLen = 0, seeded = 0;
  function emit() {
    if (cur.length <= seeded) { cur = []; curLen = 0; seeded = 0; return; }
    chunks.push([cur[0][0], cur[cur.length - 1][1]]);
    const keep = [];
    let kl = 0;
    for (let i = cur.length - 1; i >= 0 && kl < overC; i--) {
      keep.unshift(cur[i]);
      kl += cur[i][1] - cur[i][0];
    }
    if (overC > 0 && kl < sizeC) { cur = keep; curLen = kl; seeded = keep.length; }
    else { cur = []; curLen = 0; seeded = 0; }
  }
  for (const sp of spans) {
    const len = sp[1] - sp[0];
    if (len > sizeC) {
      emit(); cur = []; curLen = 0; seeded = 0;
      for (const c of fixedChunks(text.slice(sp[0], sp[1]), sizeC, overC)) {
        chunks.push([sp[0] + c[0], sp[0] + c[1]]);
      }
      continue;
    }
    if (curLen + len > sizeC && cur.length) emit();
    cur.push(sp);
    curLen += len;
  }
  emit();
  return chunks;
}

/** Paragraphs first, oversized ones by sentence, oversized sentences by word. */
export function recursiveSpans(text, sizeC) {
  const out = [];
  for (const p of paragraphSpans(text)) {
    if (p[1] - p[0] <= sizeC) { out.push(p); continue; }
    for (const s of sentenceSpans(text, p[0], p[1])) {
      if (s[1] - s[0] <= sizeC) { out.push(s); continue; }
      out.push(...wordSpans(text, s[0], s[1]));
    }
  }
  return out;
}

export const STRATEGIES = ['fixed', 'sentence', 'recursive'];

/**
 * Chunk `text` with a named strategy.
 *
 * @param {string} text
 * @param {'fixed'|'sentence'|'recursive'} strategy
 * @param {object} [opts]
 * @param {number} [opts.sizeTokens=200]
 * @param {number} [opts.overlapTokens=0]
 * @returns {Array<[number, number]>} char spans into `text`.
 */
export function chunk(text, strategy, { sizeTokens = 200, overlapTokens = 0 } = {}) {
  const sizeC = sizeTokens * CHARS_PER_TOKEN;
  const overC = overlapTokens * CHARS_PER_TOKEN;
  if (strategy === 'fixed') return fixedChunks(text, sizeC, overC);
  if (strategy === 'sentence') return pack(text, sentenceSpans(text), sizeC, overC);
  if (strategy === 'recursive') return pack(text, recursiveSpans(text, sizeC), sizeC, overC);
  throw new RangeError(`unknown strategy: ${strategy} (expected ${STRATEGIES.join('|')})`);
}

/**
 * The benchmark's structural metrics for a set of chunks.
 * Duplication is the share of emitted characters that are re-emissions
 * of text already covered by another chunk: what overlap actually costs
 * at the embedding API.
 */
export function metrics(text, chunks) {
  let emitted = 0;
  const sizes = [];
  for (const [a, b] of chunks) {
    emitted += b - a;
    sizes.push(estimateTokens(b - a));
  }
  return {
    count: chunks.length,
    avgTokens: sizes.length ? +(sizes.reduce((s, x) => s + x, 0) / sizes.length).toFixed(1) : 0,
    duplicationPct: emitted > 0 ? +(Math.max(0, (emitted - text.length) / emitted) * 100).toFixed(1) : 0,
    embeddedTokens: estimateTokens(emitted),
  };
}
