// Horizon-aware forecast for pumpfun-top5-forecast-01.
// Node 22, standard library only. Deterministic: a fixed seed drives every draw.
//
// One forecast is scored at every answer read, so the file is the mixture of the
// predictive distributions at those reads. Parameters come from the snapshot
// history itself (recent transitions on a formed board, with weak priors), which
// keeps a short practice rehearsal and the long final window on the same model:
// uncertainty and the chance a coin we have never seen takes a place both grow
// with the gap between the last mark and each read.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const SEED = 0x5eed2026;
const N_SIMS = 3200;
const MAX_CANDIDATES = 48;
const MAX_NEW = 240;
const MIN_ORDER_P = 0.008;
const MIN_SIZE_P = 0.0035;
const SIZE_UNIFORM = 0.1;
const VEL_CARRY = 0.42;
const VEL_DECAY = 0.5;

interface Task {
  places: number;
  chart_size: number;
  cadence_minutes: number;
  output: string;
  reads: string[];
  size: { edges: number[] };
}

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  quote_reserves_lamports?: string | number;
}

interface Mark {
  at: string;
  t: number;
  ok?: boolean;
  rows: Row[];
}

interface Coin {
  mint: string;
  logcap: number;
  vel: number;
  pool: boolean;
  ageMin: number;
  persist: number;
  rank: number;
}

interface Shock {
  crashP: number;
  crashMean: number;
  crashSd: number;
  calmMean: number;
  calmSd: number;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const marks: Mark[] = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as { at: string; ok?: boolean; rows: Row[] | null })
  .filter((m) => m.ok !== false && Array.isArray(m.rows) && m.rows.length > 0)
  .map((m) => ({ at: m.at, t: Date.parse(m.at), rows: m.rows! }));

if (marks.length === 0) throw new Error('the history holds no ok mark');
const last = marks[marks.length - 1]!;
const stepMin = task.cadence_minutes > 0 ? task.cadence_minutes : 10;
const stepMs = stepMin * 60_000;
const places = task.places;
const edges = task.size.edges;

/** log10(cap in SOL) = log10(lamports) - 9, exact enough past 2^53. */
function logCap(lamports: string | number): number {
  const n = BigInt(lamports);
  if (n <= 0n) return -9;
  const s = n.toString();
  const take = Math.min(15, s.length);
  return Math.log10(Number(s.slice(0, take))) + (s.length - take) - 9;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const z = spare;
      spare = null;
      return z;
    }
    let u = rng();
    if (u < 1e-12) u = 1e-12;
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

function poisson(rng: () => number, lambda: number): number {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= rng();
  } while (p > L && k < 40);
  return k - 1;
}

interface Sample {
  w: number;
  x: number;
}

function fitMeanSd(samples: Sample[], priorMean: number, priorSd: number, priorN: number): { mean: number; sd: number } {
  let wsum = priorN;
  let acc = priorMean * priorN;
  for (const s of samples) {
    wsum += s.w;
    acc += s.w * s.x;
  }
  const mean = acc / wsum;
  let v = priorN * priorSd * priorSd;
  for (const s of samples) v += s.w * (s.x - mean) ** 2;
  return { mean, sd: Math.sqrt(Math.max(v / wsum, 1e-6)) };
}

function poolCount(rows: Row[], limit: number): number {
  let n = 0;
  for (const r of rows) {
    if (r.rank <= limit && (r.venue === 'pool' || r.venue === 'migrating')) n += 1;
  }
  return n;
}

const formed: Array<{ prev: Mark; cur: Mark; dt: number; w: number }> = [];
for (let i = 1; i < marks.length; i++) {
  const prev = marks[i - 1]!;
  const cur = marks[i]!;
  const dt = Math.max(1, Math.round((cur.t - prev.t) / stepMs));
  if (dt > 4) continue;
  if (poolCount(prev.rows, 15) < 4) continue;
  const ageH = (last.t - cur.t) / 3_600_000;
  const w = Math.exp(-ageH / 1.25) / dt;
  formed.push({ prev, cur, dt, w });
}
const transitions = formed.length > 0 ? formed : [];

const poolCalm: Sample[] = [];
const poolCrash: Sample[] = [];
const curveCalm: Sample[] = [];
const curveCrash: Sample[] = [];
const relSamples: Sample[] = [];
let lambdaW = 0;
let lambdaAcc = 0;
let crashPoolW = 0;
let crashPoolN = 0;
let crashCurveW = 0;
let crashCurveN = 0;

for (const tr of transitions) {
  const prevBy = new Map(tr.prev.rows.map((r) => [r.mint, r]));
  const prevTop = tr.prev.rows.find((r) => r.rank === 1) ?? tr.prev.rows[0];
  const top1 = prevTop ? logCap(prevTop.market_cap_lamports) : 3;
  let entrants = 0;
  for (const r of tr.cur.rows) {
    if (r.rank > 8) break;
    if (!prevBy.has(r.mint) || (prevBy.get(r.mint)!.rank > 30)) {
      entrants += 1;
      relSamples.push({ w: tr.w, x: logCap(r.market_cap_lamports) - top1 });
    }
  }
  lambdaW += tr.w * tr.dt;
  lambdaAcc += tr.w * entrants;

  for (const r of tr.prev.rows) {
    if (r.rank > 20) break;
    const nxt = tr.cur.rows.find((q) => q.mint === r.mint);
    if (!nxt) {
      // Left the published rows: treat as a crash if it had been near the top.
      if (r.rank <= 12) {
        const shock = r.venue === 'curve' ? curveCrash : poolCrash;
        shock.push({ w: tr.w, x: -0.7 });
        if (r.venue === 'curve') {
          crashCurveW += tr.w;
          crashCurveN += tr.w;
        } else {
          crashPoolW += tr.w;
          crashPoolN += tr.w;
        }
      }
      continue;
    }
    const d = (logCap(nxt.market_cap_lamports) - logCap(r.market_cap_lamports)) / tr.dt;
    const isCurve = r.venue === 'curve';
    const crashed = d < -0.28;
    if (isCurve) {
      crashCurveW += tr.w;
      if (crashed) crashCurveN += tr.w;
      (crashed ? curveCrash : curveCalm).push({ w: tr.w, x: d });
    } else {
      crashPoolW += tr.w;
      if (crashed) crashPoolN += tr.w;
      (crashed ? poolCrash : poolCalm).push({ w: tr.w, x: d });
    }
  }
}

function shockOf(
  calm: Sample[],
  crash: Sample[],
  crashW: number,
  crashN: number,
  priorCrash: number,
  priorCalmMean: number,
  priorCalmSd: number,
  priorCrashMean: number,
): Shock {
  const calmFit = fitMeanSd(calm, priorCalmMean, priorCalmSd, 10);
  const crashFit = fitMeanSd(crash, priorCrashMean, 0.22, 4);
  const crashP = (crashN + priorCrash * 6) / (crashW + 6);
  return {
    crashP: Math.min(0.62, Math.max(0.04, crashP)),
    crashMean: Math.max(-1.3, Math.min(-0.15, crashFit.mean)),
    crashSd: Math.min(0.55, Math.max(0.08, crashFit.sd)),
    calmMean: Math.max(-0.12, Math.min(0.18, calmFit.mean)),
    calmSd: Math.min(0.4, Math.max(0.05, calmFit.sd)),
  };
}

const poolShock = shockOf(poolCalm, poolCrash, crashPoolW, crashPoolN, 0.12, 0.01, 0.11, -0.55);
const curveShock = shockOf(curveCalm, curveCrash, crashCurveW, crashCurveN, 0.28, -0.08, 0.16, -0.6);

// Prior: about 0.7 serious new coins per step, some of them above the current leader.
const lambda = Math.min(3.5, Math.max(0.15, (lambdaAcc + 0.7 * 4) / (lambdaW + 4)));
const relFit = fitMeanSd(relSamples, -0.02, 0.28, 8);
const relMean = Math.max(-0.45, Math.min(0.35, relFit.mean));
const relSd = Math.min(0.55, Math.max(0.12, relFit.sd * 1.05));

const recentCut = marks[Math.max(0, marks.length - 12)]!.t;
const seen = new Map<string, { n: number; prevLog: number | null; prevT: number }>();
for (const m of marks) {
  if (m.t < recentCut) continue;
  for (const r of m.rows) {
    if (r.rank > 25) continue;
    const lc = logCap(r.market_cap_lamports);
    const slot = seen.get(r.mint) ?? { n: 0, prevLog: null, prevT: 0 };
    slot.n += 1;
    slot.prevLog = lc;
    slot.prevT = m.t;
    seen.set(r.mint, slot);
  }
}

const prevMark = marks.length >= 2 ? marks[marks.length - 2]! : null;
const prevByMint = new Map(prevMark ? prevMark.rows.map((r) => [r.mint, r]) : []);
const coins: Coin[] = [];
for (const r of last.rows) {
  if (coins.length >= MAX_CANDIDATES && r.rank > 20) break;
  if (r.rank > 60 && coins.length >= 20) break;
  const lc = logCap(r.market_cap_lamports);
  const prev = prevByMint.get(r.mint);
  let vel = 0;
  if (prev) {
    const dt = Math.max(1, Math.round((last.t - prevMark!.t) / stepMs));
    vel = (lc - logCap(prev.market_cap_lamports)) / dt;
  }
  const ageMin = Number.isFinite(Date.parse(r.created_at)) ? (last.t - Date.parse(r.created_at)) / 60_000 : 120;
  coins.push({
    mint: r.mint,
    logcap: lc,
    vel: Math.max(-0.8, Math.min(0.8, vel)),
    pool: r.venue === 'pool' || r.venue === 'migrating',
    ageMin,
    persist: seen.get(r.mint)?.n ?? 1,
    rank: r.rank,
  });
}
if (coins.length < places) throw new Error(`the last mark ranks ${coins.length} coins, fewer than ${places} places`);
coins.sort((a, b) => a.rank - b.rank || (a.mint < b.mint ? -1 : 1));

const readSteps: number[] = [];
for (const iso of task.reads) {
  const t = Date.parse(iso);
  const steps = Math.round((t - last.t) / stepMs);
  readSteps.push(Math.max(0, Math.min(96, steps)));
}
if (readSteps.length === 0) readSteps.push(1);
const maxStep = Math.max(...readSteps);

/** Place-k log cap on the last mark: the size model follows these levels, not one coin. */
const lastPlaceLc: number[] = [];
for (let k = 0; k < places; k++) {
  const row = last.rows.find((r) => r.rank === k + 1);
  const prev = lastPlaceLc[k - 1];
  lastPlaceLc.push(row ? logCap(row.market_cap_lamports) : (prev ?? 3) - 0.08);
}

const placeDrift: number[] = [];
const placeSd: number[] = [];
for (let k = 0; k < places; k++) {
  const deltas: Sample[] = [];
  for (const tr of transitions) {
    const a = tr.prev.rows.find((r) => r.rank === k + 1);
    const b = tr.cur.rows.find((r) => r.rank === k + 1);
    if (!a || !b) continue;
    deltas.push({ w: tr.w, x: (logCap(b.market_cap_lamports) - logCap(a.market_cap_lamports)) / tr.dt });
  }
  // Mild upward prior: a new board's top drifts up, then the saturation below stops it running away.
  const priorDrift = 0.045 - k * 0.005;
  const fit = fitMeanSd(deltas, priorDrift, 0.1, 3);
  placeDrift.push(Math.max(-0.08, Math.min(0.22, fit.mean)));
  placeSd.push(Math.min(0.28, Math.max(0.05, fit.sd)));
}

/** Integrated drift over `s` steps, saturating with time so a 10-hour window does not explode. */
function placeMean(k: number, s: number): number {
  const tau = 8;
  if (s <= 0) return lastPlaceLc[k]!;
  return lastPlaceLc[k]! + placeDrift[k]! * tau * (1 - Math.exp(-s / tau));
}
function placeSigma(k: number, s: number): number {
  const steps = Math.max(s, 1);
  return Math.sqrt(0.16 ** 2 + placeSd[k]! ** 2 * steps);
}

function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}
/** P(X < x) for a normal. */
const normalCdf = (x: number, mean: number, sd: number): number => 0.5 * erfc(-(x - mean) / (sd * Math.SQRT2));

const rng = mulberry32(SEED);
const gauss = makeGaussian(rng);
const uni = rng;

const orderHits: Array<Map<string, number>> = Array.from({ length: places }, () => new Map());
const sizeHits: number[][] = Array.from({ length: places }, () => new Array(edges.length + 1).fill(0));
let samples = 0;

function binOf(x: number): number {
  if (x < edges[0]!) return 0;
  if (x >= edges[edges.length - 1]!) return edges.length;
  // edges are a uniform 0.05 grid in this contest; fall back to a scan if not.
  const width = edges[1]! - edges[0]!;
  if (width > 0) {
    const i = Math.floor((x - edges[0]!) / width) + 1;
    if (i >= 1 && i <= edges.length - 1 && x >= edges[i - 1]! && x < edges[i]!) return i;
  }
  for (let i = 0; i < edges.length; i++) if (x < edges[i]!) return i;
  return edges.length;
}

function stepCoin(logcap: number, vel: number, pool: boolean, young: boolean, sticky: boolean): { logcap: number; vel: number } {
  const s = pool ? poolShock : curveShock;
  let pCrash = s.crashP;
  if (young) pCrash = Math.min(0.72, pCrash * 1.7);
  if (sticky) pCrash *= 0.7;
  const jump = uni() < pCrash;
  // A brand-new print that is not already running reverts; sustained coins keep their calm mean.
  const fade = young ? (pool ? -0.055 : -0.11) : 0;
  const shock = (jump ? s.crashMean + s.crashSd * gauss() : s.calmMean + s.calmSd * gauss()) + fade;
  const carried = VEL_CARRY * vel;
  const next = logcap + carried + shock;
  return { logcap: next, vel: vel * VEL_DECAY };
}

for (let sim = 0; sim < N_SIMS; sim++) {
  const logcaps = coins.map((c) => c.logcap);
  const vels = coins.map((c) => c.vel);
  const bornAt = coins.map(() => -1_000);
  const news: number[] = [];
  const newsBorn: number[] = [];
  let cursor = 0;
  const targets = new Map<number, number[]>();
  for (let r = 0; r < readSteps.length; r++) {
    const list = targets.get(readSteps[r]!) ?? [];
    list.push(r);
    targets.set(readSteps[r]!, list);
  }
  const record = () => {
    const ids: Array<{ id: number; lc: number; mint: string | null }> = [];
    for (let i = 0; i < coins.length; i++) ids.push({ id: i, lc: logcaps[i]!, mint: coins[i]!.mint });
    for (let j = 0; j < news.length; j++) ids.push({ id: 1000 + j, lc: news[j]!, mint: null });
    ids.sort((a, b) => b.lc - a.lc || (a.mint && b.mint ? (a.mint < b.mint ? -1 : 1) : a.id - b.id));
    for (let k = 0; k < places; k++) {
      const row = ids[k];
      sizeHits[k]![binOf(row ? row.lc : 0)] += 1;
      if (row?.mint) {
        const m = orderHits[k]!;
        m.set(row.mint, (m.get(row.mint) ?? 0) + 1);
      }
    }
    samples += 1;
  };
  if (targets.has(0)) record();
  for (let step = 1; step <= maxStep; step++) {
    let leader = -10;
    for (const lc of logcaps) if (lc > leader) leader = lc;
    for (const lc of news) if (lc > leader) leader = lc;
    for (let i = 0; i < coins.length; i++) {
      const c = coins[i]!;
      const age = c.ageMin + step * stepMin;
      const young = age < 14 && c.persist <= 2;
      const sticky = c.persist >= 4 && c.pool;
      const nxt = stepCoin(logcaps[i]!, vels[i]!, c.pool, young, sticky);
      logcaps[i] = nxt.logcap;
      vels[i] = nxt.vel;
      void bornAt;
    }
    const nNew = poisson(uni, lambda);
    for (let j = 0; j < nNew && news.length < MAX_NEW; j++) {
      // Modest right skew: most launches land near the leaders, a few clear the top.
      const rel = relMean + relSd * (uni() < 0.18 ? Math.abs(gauss()) + 0.2 : gauss());
      news.push(leader + rel);
      newsBorn.push(step);
    }
    for (let j = 0; j < news.length; j++) {
      const age = (step - newsBorn[j]!) * stepMin;
      const nxt = stepCoin(news[j]!, 0, true, age < 20, false);
      news[j] = nxt.logcap;
    }
    // Drop newcomers that can no longer reach the top of the board.
    if (news.length > 80) {
      let floor = leader;
      for (const lc of logcaps) if (lc > floor) floor = lc;
      floor -= 1.6;
      let w = 0;
      for (let j = 0; j < news.length; j++) {
        if (news[j]! >= floor) {
          news[w] = news[j]!;
          newsBorn[w] = newsBorn[j]!;
          w += 1;
        }
      }
      news.length = w;
      newsBorn.length = w;
    }
    if (targets.has(step)) record();
    void cursor;
  }
}

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  const rows: Array<{ mint: string; micro: number }> = [];
  for (const [mint, hits] of orderHits[k]!) {
    const micro = Math.floor((hits / samples) * 1_000_000 + 1e-9);
    if (micro >= MIN_ORDER_P * 1_000_000) rows.push({ mint, micro });
  }
  rows.sort((a, b) => b.micro - a.micro || (a.mint < b.mint ? -1 : 1));
  let sum = 0;
  const kept: Array<{ mint: string; micro: number }> = [];
  for (const row of rows) {
    if (kept.length >= 80) break;
    if (sum + row.micro > 1_000_000) break;
    kept.push(row);
    sum += row.micro;
  }
  for (const row of kept) probabilities.push({ mint: row.mint, place: k + 1, p: row.micro / 1_000_000 });
}

const score = new Map<string, number>();
for (let k = 0; k < places; k++) {
  for (const [mint, hits] of orderHits[k]!) {
    score.set(mint, (score.get(mint) ?? 0) + (hits / samples) * (places + 1 - (k + 1)));
  }
}
const ranked = [...score.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
const top5: string[] = [];
for (const [mint] of ranked) {
  if (top5.length >= places) break;
  top5.push(mint);
}
for (const c of coins) {
  if (top5.length >= places) break;
  if (!top5.includes(c.mint)) top5.push(c.mint);
}

const sizes: Array<{ place: number; bin: number; p: number }> = [];
const nBins = edges.length + 1;
const binLo = (b: number): number => (b <= 0 ? -Infinity : edges[b - 1]!);
const binHi = (b: number): number => (b >= edges.length ? Infinity : edges[b]!);
for (let k = 0; k < places; k++) {
  // Equal mixture over the answer reads of a drifting place-cap, plus a small
  // high-side component for a launch that resets the top. The coin simulation
  // is a poor size model: when the leader crashes the next coin takes the place,
  // so the place itself moves less than any one mint.
  const mix = new Array(nBins).fill(0);
  const skewW = 0.2 - k * 0.025;
  const nReads = readSteps.length;
  for (const s of readSteps) {
    const mean = placeMean(k, s);
    const sd = placeSigma(k, s);
    const parts = [
      { w: 1 - skewW, mean, sd },
      { w: skewW, mean: mean + 0.2 - k * 0.02, sd: sd * 1.12 },
    ];
    for (const part of parts) {
      for (let b = 0; b < nBins; b++) {
        const mass = normalCdf(binHi(b), part.mean, part.sd) - normalCdf(binLo(b), part.mean, part.sd);
        mix[b] += (part.w / nReads) * mass;
      }
    }
  }
  const rows: Array<{ bin: number; micro: number }> = [];
  let named = 0;
  for (let b = 0; b < nBins; b++) {
    const p = (1 - SIZE_UNIFORM) * mix[b]! + SIZE_UNIFORM / nBins;
    const micro = Math.floor(p * 1_000_000 + 1e-9);
    if (micro >= MIN_SIZE_P * 1_000_000) {
      rows.push({ bin: b, micro });
      named += micro;
    }
  }
  // Keep the named bins' probabilities (leftover mass stays unnamed and is shared).
  void named;
  for (const row of rows) sizes.push({ place: k + 1, bin: row.bin, p: row.micro / 1_000_000 });
}

const forecast = { probabilities, top5, sizes };
const text = `${JSON.stringify(forecast)}\n`;
if (text.length > 1_048_576) throw new Error(`forecast is ${text.length} bytes`);
writeFileSync(join(OUTPUT_DIR, task.output), text);

if (process.env.MAKEFUN_DEBUG === '1') {
  const named = (place: number) => probabilities.filter((r) => r.place === place);
  const sumP = (place: number) => named(place).reduce((a, r) => a + r.p, 0);
  const mintName = new Map(last.rows.map((r) => [r.mint, `${r.symbol ?? ''}#${r.rank}`]));
  console.error(
    JSON.stringify(
      {
        marks: marks.length,
        formed: transitions.length,
        lambda: Number(lambda.toFixed(3)),
        rel: [Number(relMean.toFixed(3)), Number(relSd.toFixed(3))],
        placeDrift: placeDrift.map((d) => Number(d.toFixed(4))),
        placeSd: placeSd.map((d) => Number(d.toFixed(4))),
        lastPlaceLc: lastPlaceLc.map((d) => Number(d.toFixed(3))),
        pool: poolShock,
        curve: curveShock,
        readSteps,
        samples,
        coins: coins.slice(0, 8).map((c) => ({
          mint: c.mint.slice(0, 6),
          rank: c.rank,
          lc: Number(c.logcap.toFixed(3)),
          vel: Number(c.vel.toFixed(3)),
          pool: c.pool,
          age: Math.round(c.ageMin),
          persist: c.persist,
        })),
        place1: named(1)
          .slice(0, 8)
          .map((r) => ({ who: mintName.get(r.mint) ?? r.mint.slice(0, 6), p: r.p })),
        unnamed: [1, 2, 3, 4, 5].map((p) => Number((1 - sumP(p)).toFixed(3))),
        sizePeak: [0, 1, 2, 3, 4].map((k) => {
          const rows = sizes.filter((s) => s.place === k + 1);
          const best = rows.reduce((a, b) => (b.p > a.p ? b : a));
          const mass = rows.reduce((a, b) => a + b.p, 0);
          return { bin: best.bin, p: best.p, mass: Number(mass.toFixed(3)), n: rows.length };
        }),
        top5: top5.map((m) => mintName.get(m) ?? m.slice(0, 6)),
        bytes: text.length,
      },
      null,
      2,
    ),
  );
}
