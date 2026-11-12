
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
const SIZE_UNIFORM = 0.02;
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
  const priorDrift = 0.045 - k * 0.005;
  const fit = fitMeanSd(deltas, priorDrift, 0.1, 3);
  placeDrift.push(Math.max(-0.08, Math.min(0.22, fit.mean)));
  placeSd.push(Math.min(0.28, Math.max(0.05, fit.sd)));
}

function placeMean(k: number, s: number): number {
  const tau = 8;
  if (s <= 0) return lastPlaceLc[k]!;
  return lastPlaceLc[k]! + placeDrift[k]! * tau * (1 - Math.exp(-s / tau)) + ([0.55, 0.32, 0.24, 0.18, 0.14][k] ?? 0.12) * (1 - Math.exp(-s / 10));
}
function placeSigma(k: number, s: number): number {
  const steps = Math.max(s, 1);
  const empiricalStability = rankStability(k);
  const localNoise = Math.max(placeSd[k]!, empiricalStability * 0.5);
  return Math.min(0.38, Math.max(0.09, 0.09 + k * 0.005 + Math.min(0.28, 0.04 * Math.log1p(steps) + localNoise * Math.sqrt(Math.min(steps, 24) / 12))));
}

function blendSizeMass(model: number[], empirical: number[], place: number): number[] {
  const result = new Array(model.length).fill(0);
  const stability = rankStability(place);
  const empiricalWeight = Math.min(0.28, Math.max(0.10, 0.12 + stability * 0.35));
  let total = 0;
  for (let bin = 0; bin < model.length; bin++) {
    const value = (1 - empiricalWeight) * model[bin]! + empiricalWeight * empirical[bin]!;
    result[bin] = Math.max(0, value);
    total += result[bin]!;
  }
  if (total <= 0) return model.slice();
  for (let bin = 0; bin < result.length; bin++) result[bin] /= total;
  return result;
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
const normalCdf = (x: number, mean: number, sd: number): number => 0.5 * erfc(-(x - mean) / (sd * Math.SQRT2));

const rng = mulberry32(SEED);
const gaussian = makeGaussian(rng);
const random = rng;

const orderObservations: Array<Map<string, number>> = Array.from({ length: places }, () => new Map());
const sizeObservations: number[][] = Array.from({ length: places }, () => new Array(edges.length + 1).fill(0));
let observationCount = 0;

function binOf(x: number): number {
  if (x < edges[0]!) return 0;
  if (x >= edges[edges.length - 1]!) return edges.length;
  const width = edges[1]! - edges[0]!;
  if (width > 0) {
    const i = Math.floor((x - edges[0]!) / width) + 1;
    if (i >= 1 && i <= edges.length - 1 && x >= edges[i - 1]! && x < edges[i]!) return i;
  }
  for (let i = 0; i < edges.length; i++) if (x < edges[i]!) return i;
  return edges.length;
}

function evolveCoin(logcap: number, vel: number, pool: boolean, young: boolean, sticky: boolean): { logcap: number; vel: number } {
  const s = pool ? poolShock : curveShock;
  let pCrash = s.crashP;
  if (young) pCrash = Math.min(0.72, pCrash * 1.7);
  if (sticky) pCrash *= 0.7;
  const jump = random() < pCrash;
  const fade = young ? (pool ? -0.055 : -0.11) : 0;
  const shock = (jump ? s.crashMean + s.crashSd * gaussian() : s.calmMean + s.calmSd * gaussian()) + fade;
  const carried = VEL_CARRY * vel;
  const next = logcap + carried + shock;
  return { logcap: next, vel: vel * VEL_DECAY };
}

function recordBoard(known: number[], unseen: number[]): void {
  const board: Array<{ id: number; logcap: number; mint: string | null }> = [];
  for (let i = 0; i < coins.length; i++) board.push({ id: i, logcap: known[i]!, mint: coins[i]!.mint });
  for (let i = 0; i < unseen.length; i++) board.push({ id: coins.length + i, logcap: unseen[i]!, mint: null });
  board.sort((a, b) => b.logcap - a.logcap || (a.mint && b.mint ? (a.mint < b.mint ? -1 : 1) : a.id - b.id));
  for (let k = 0; k < places; k++) {
    const row = board[k];
    sizeObservations[k]![binOf(row?.logcap ?? 0)] += 1;
    if (row?.mint) {
      const observed = orderObservations[k]!;
      observed.set(row.mint, (observed.get(row.mint) ?? 0) + 1);
    }
  }
  observationCount += 1;
}

function pruneUnseen(unseen: number[], born: number[], known: number[]): void {
  if (unseen.length <= 80) return;
  let threshold = -Infinity;
  for (const cap of known) threshold = Math.max(threshold, cap);
  threshold -= 1.6;
  let kept = 0;
  for (let i = 0; i < unseen.length; i++) {
    if (unseen[i]! >= threshold) {
      unseen[kept] = unseen[i]!;
      born[kept] = born[i]!;
      kept += 1;
    }
  }
  unseen.length = kept;
  born.length = kept;
}

function simulatePath(targetSteps: Set<number>): void {
  const knownCaps = coins.map((coin) => coin.logcap);
  const velocities = coins.map((coin) => coin.vel);
  const unseenCaps: number[] = [];
  const unseenBorn: number[] = [];
  if (targetSteps.has(0)) recordBoard(knownCaps, unseenCaps);

  for (let step = 1; step <= maxStep; step++) {
    let leader = -Infinity;
    for (const cap of knownCaps) leader = Math.max(leader, cap);
    for (const cap of unseenCaps) leader = Math.max(leader, cap);

    for (let i = 0; i < coins.length; i++) {
      const coin = coins[i]!;
      const age = coin.ageMin + step * stepMin;
      const state = evolveCoin(knownCaps[i]!, velocities[i]!, coin.pool, age < 14 && coin.persist <= 2, coin.persist >= 4 && coin.pool);
      knownCaps[i] = state.logcap;
      velocities[i] = state.vel;
    }

    const arrivals = poisson(random, lambda);
    for (let i = 0; i < arrivals && unseenCaps.length < MAX_NEW; i++) {
      const tail = random() < 0.18;
      const relative = relMean + relSd * (tail ? Math.abs(gaussian()) + 0.2 : gaussian());
      unseenCaps.push(leader + relative);
      unseenBorn.push(step);
    }

    for (let i = 0; i < unseenCaps.length; i++) {
      const age = (step - unseenBorn[i]!) * stepMin;
      unseenCaps[i] = evolveCoin(unseenCaps[i]!, 0, true, age < 20, false).logcap;
    }
    pruneUnseen(unseenCaps, unseenBorn, knownCaps);
    if (targetSteps.has(step)) recordBoard(knownCaps, unseenCaps);
  }
}

const targets = new Set(readSteps);
for (let draw = 0; draw < N_SIMS; draw++) simulatePath(targets);

function orderPrior(place: number): Map<string, number> {
  const weights = new Map<string, number>();
  let total = 0;
  for (const coin of coins) {
    const distance = Math.abs((coin.rank - 1) - place);
    const freshness = coin.persist <= 2 ? 1.15 : 1;
    const weight = freshness * Math.exp(-distance / 3.2);
    weights.set(coin.mint, weight);
    total += weight;
  }
  for (const [mint, weight] of weights) weights.set(mint, weight / total);
  return weights;
}

function unknownReserve(place: number): number {
  const averageHorizon = readSteps.reduce((sum, step) => sum + step, 0) / readSteps.length;
  const arrivalChance = 1 - Math.exp(-lambda * averageHorizon / Math.max(places, 1));
  const higherPlacesAreHarder = 1 - place * 0.055;
  const observedTurnover = recentTurnover(place);
  const horizonScale = Math.min(1, averageHorizon / 18);
  const reserve = 0.22 + arrivalChance * 0.42 * higherPlacesAreHarder + observedTurnover * 0.24 * Math.max(0.4, horizonScale) + (1 - place * 0.04) * 0.06;
  return Math.min(0.78, Math.max(0.28, reserve));
}

/**
 * Recent top-board turnover is a useful calibration signal independent of the
 * simulated arrival rate.  It measures whether a coin at a given place stays
 * in the next mark, while treating a new high-ranking mint as turnover.
 */
function recentTurnover(place: number): number {
  const start = Math.max(1, marks.length - 8);
  let transitions = 0;
  let changes = 0;
  for (let index = start; index < marks.length; index++) {
    const previous = marks[index - 1]!;
    const current = marks[index]!;
    const oldRow = previous.rows.find((row) => row.rank === place + 1);
    const newRow = current.rows.find((row) => row.rank === place + 1);
    if (!oldRow || !newRow) continue;
    transitions += 1;
    if (oldRow.mint !== newRow.mint) changes += 1;
  }
  if (transitions === 0) return 0.25;
  const empirical = changes / transitions;
  // A beta prior keeps the five-mark rehearsal from producing extreme values.
  return (changes + 1.5) / (transitions + 6) * 0.55 + empirical * 0.45;
}

function rankStability(place: number): number {
  const start = Math.max(1, marks.length - 8);
  const levels: number[] = [];
  for (let index = start; index < marks.length; index++) {
    const row = marks[index]!.rows.find((item) => item.rank === place + 1);
    if (row) levels.push(logCap(row.market_cap_lamports));
  }
  if (levels.length < 2) return 0.2;
  let mean = 0;
  for (const value of levels) mean += value;
  mean /= levels.length;
  let variance = 0;
  for (const value of levels) variance += (value - mean) ** 2;
  variance /= levels.length - 1;
  return Math.max(0.04, Math.min(0.35, Math.sqrt(variance)));
}

function reserveDiagnostics(): Array<{ place: number; reserve: number; turnover: number; stability: number }> {
  const values: Array<{ place: number; reserve: number; turnover: number; stability: number }> = [];
  for (let place = 0; place < places; place++) {
    values.push({
      place: place + 1,
      reserve: Number(unknownReserve(place).toFixed(4)),
      turnover: Number(recentTurnover(place).toFixed(4)),
      stability: Number(rankStability(place).toFixed(4)),
    });
  }
  return values;
}

function calibratedRows(place: number): Array<{ mint: string; micro: number }> {
  const hits = orderObservations[place]!;
  const prior = orderPrior(place);
  const reserve = unknownReserve(place);
  const pseudoCount = Math.max(80, observationCount * 0.018);
  const candidates = new Set<string>([...hits.keys(), ...prior.keys()]);
  const rows: Array<{ mint: string; p: number }> = [];
  for (const mint of candidates) {
    const empirical = (hits.get(mint) ?? 0) / observationCount;
    const baseline = prior.get(mint) ?? 0;
    const smoothed = (empirical * observationCount + baseline * pseudoCount) / (observationCount + pseudoCount);
    const p = smoothed * (1 - reserve);
    if (p >= MIN_ORDER_P) rows.push({ mint, p });
  }
  rows.sort((a, b) => b.p - a.p || (a.mint < b.mint ? -1 : 1));
  const output: Array<{ mint: string; micro: number }> = [];
  let named = 0;
  for (const row of rows) {
    if (output.length >= 80) break;
    const micro = Math.floor(row.p * 1_000_000 + 1e-9);
    if (micro < MIN_ORDER_P * 1_000_000 || named + micro > 940_000) continue;
    output.push({ mint: row.mint, micro });
    named += micro;
  }
  return output;
}

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  for (const row of calibratedRows(k)) probabilities.push({ mint: row.mint, place: k + 1, p: row.micro / 1_000_000 });
}

const score = new Map<string, number>();
for (let k = 0; k < places; k++) {
  for (const [mint, hits] of orderObservations[k]!) {
    score.set(mint, (score.get(mint) ?? 0) + (hits / observationCount) * (places - k));
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
  const empiricalMass = sizeObservations[k]!.map((count) => count / observationCount);
  const calibratedMass = blendSizeMass(mix, empiricalMass, k);
  const rows: Array<{ bin: number; micro: number }> = [];
  let named = 0;
  for (let b = 0; b < nBins; b++) {
    const model = (1 - SIZE_UNIFORM) * calibratedMass[b]! + SIZE_UNIFORM / nBins;
    const p = model;
    const micro = Math.floor(p * 1_000_000 + 1e-9);
    if (micro >= MIN_SIZE_P * 1_000_000) {
      rows.push({ bin: b, micro });
      named += micro;
    }
  }
  void named;
  for (const row of rows) sizes.push({ place: k + 1, bin: row.bin, p: row.micro / 1_000_000 });
}

/**
 * Validate the wire format before serialising: malformed output scores zero.
 */
function sumByPlace<T extends { place: number; p: number }>(rows: T[]): Map<number, number> {
  const sums = new Map<number, number>();
  for (const row of rows) sums.set(row.place, (sums.get(row.place) ?? 0) + row.p);
  return sums;
}

function finiteProbability(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(label + ' must be in [0, 1]');
}

function validateOrderRows(rows: Array<{ mint: string; place: number; p: number }>): void {
  const seen = new Set<string>();
  const counts = new Map<number, number>();
  for (const row of rows) {
    if (!Number.isInteger(row.place) || row.place < 1 || row.place > places) throw new Error('invalid order place');
    if (typeof row.mint !== 'string' || row.mint.length < 32 || row.mint.length > 44) throw new Error('invalid order mint');
    finiteProbability(row.p, 'order probability');
    const key = String(row.place) + ':' + row.mint;
    if (seen.has(key)) throw new Error('duplicate mint at a place');
    seen.add(key);
    counts.set(row.place, (counts.get(row.place) ?? 0) + 1);
  }
  for (const count of counts.values()) if (count > 100) throw new Error('too many order rows');
  for (const sum of sumByPlace(rows).values()) if (sum > 1 + 1e-9) throw new Error('order probability exceeds one');
}

function validateSizeRows(rows: Array<{ place: number; bin: number; p: number }>): void {
  const seen = new Set<string>();
  const bins = edges.length + 1;
  const counts = new Map<number, number>();
  for (const row of rows) {
    if (!Number.isInteger(row.place) || row.place < 1 || row.place > places) throw new Error('invalid size place');
    if (!Number.isInteger(row.bin) || row.bin < 0 || row.bin >= bins) throw new Error('invalid size bin');
    finiteProbability(row.p, 'size probability');
    const key = String(row.place) + ':' + String(row.bin);
    if (seen.has(key)) throw new Error('duplicate size bin at a place');
    seen.add(key);
    counts.set(row.place, (counts.get(row.place) ?? 0) + 1);
  }
  for (const count of counts.values()) if (count > bins) throw new Error('too many size rows');
  for (const sum of sumByPlace(rows).values()) if (sum > 1 + 1e-9) throw new Error('size probability exceeds one');
}

function validateTop5(rows: string[]): void {
  if (rows.length !== places) throw new Error('top5 has wrong length');
  const seen = new Set<string>();
  for (const mint of rows) {
    if (typeof mint !== 'string' || mint.length < 32 || mint.length > 44) throw new Error('invalid top5 mint');
    if (seen.has(mint)) throw new Error('top5 must be distinct');
    seen.add(mint);
  }
}

type ForecastHealth = {
  place: number;
  namedOrderMass: number;
  namedSizeMass: number;
  orderRows: number;
  sizeRows: number;
  unknownOrderMass: number;
  entropy: number;
};

function entropy(rows: Array<{ p: number }>, remainder: number, slots: number): number {
  let value = 0;
  for (const row of rows) {
    if (row.p > 0) value -= row.p * Math.log(row.p);
  }
  if (remainder > 0 && slots > 0) {
    const each = remainder / slots;
    if (each > 0) value -= remainder * Math.log(each);
  }
  return value;
}

function healthForPlace(place: number): ForecastHealth {
  const orderRows = probabilities.filter((row) => row.place === place);
  const sizeRows = sizes.filter((row) => row.place === place);
  const namedOrderMass = orderRows.reduce((sum, row) => sum + row.p, 0);
  const namedSizeMass = sizeRows.reduce((sum, row) => sum + row.p, 0);
  const unnamedCoins = Math.max(1, task.chart_size - orderRows.length);
  const unnamedBins = Math.max(1, edges.length + 1 - sizeRows.length);
  const orderEntropy = entropy(orderRows, Math.max(0, 1 - namedOrderMass), unnamedCoins);
  const sizeEntropy = entropy(sizeRows, Math.max(0, 1 - namedSizeMass), unnamedBins);
  return {
    place,
    namedOrderMass: Number(namedOrderMass.toFixed(6)),
    namedSizeMass: Number(namedSizeMass.toFixed(6)),
    orderRows: orderRows.length,
    sizeRows: sizeRows.length,
    unknownOrderMass: Number(Math.max(0, 1 - namedOrderMass).toFixed(6)),
    entropy: Number((orderEntropy + sizeEntropy).toFixed(6)),
  };
}

function forecastHealth(): ForecastHealth[] {
  const result: ForecastHealth[] = [];
  for (let place = 1; place <= places; place++) result.push(healthForPlace(place));
  return result;
}

function validateHealth(rows: ForecastHealth[]): void {
  if (rows.length !== places) throw new Error('health record has wrong number of places');
  for (const row of rows) {
    if (row.orderRows > 100) throw new Error('health found too many order rows');
    if (row.sizeRows > edges.length + 1) throw new Error('health found too many size rows');
    if (row.namedOrderMass < 0 || row.namedOrderMass > 1 + 1e-6) throw new Error('health found invalid order mass');
    if (row.namedSizeMass < 0 || row.namedSizeMass > 1 + 1e-6) throw new Error('health found invalid size mass');
    if (row.unknownOrderMass < 0 || row.unknownOrderMass > 1 + 1e-6) throw new Error('health found invalid unnamed mass');
    if (!Number.isFinite(row.entropy) || row.entropy < 0) throw new Error('health found invalid entropy');
  }
}

function stableTop5(): string[] {
  const available = new Set(coins.map((coin) => coin.mint));
  const result: string[] = [];
  for (const mint of top5) {
    if (available.has(mint) && !result.includes(mint)) result.push(mint);
  }
  for (const coin of coins) {
    if (result.length >= places) break;
    if (!result.includes(coin.mint)) result.push(coin.mint);
  }
  if (result.length !== places) throw new Error('cannot build a stable top5');
  return result;
}

function validateKnownMints(): void {
  const historyMints = new Set<string>();
  for (const mark of marks) for (const row of mark.rows) historyMints.add(row.mint);
  for (const row of probabilities) {
    if (!historyMints.has(row.mint)) throw new Error('probability names a mint absent from history');
  }
  for (const mint of top5) {
    if (!historyMints.has(mint)) throw new Error('top5 names a mint absent from history');
  }
}

function validateOrderCoverage(): void {
  const byPlace = new Map<number, Array<{ mint: string; p: number }>>();
  for (const row of probabilities) {
    const rows = byPlace.get(row.place) ?? [];
    rows.push({ mint: row.mint, p: row.p });
    byPlace.set(row.place, rows);
  }
  for (let place = 1; place <= places; place++) {
    const rows = byPlace.get(place) ?? [];
    const mass = rows.reduce((sum, row) => sum + row.p, 0);
    const distinct = new Set(rows.map((row) => row.mint)).size;
    if (distinct !== rows.length) throw new Error('order coverage is not distinct');
    if (mass >= 0.999999) throw new Error('no probability remains for unseen launches');
    if (rows.length === 0) throw new Error('order coverage names no observed candidates');
  }
}

function validateSizeCoverage(): void {
  const byPlace = new Map<number, number>();
  for (const row of sizes) byPlace.set(row.place, (byPlace.get(row.place) ?? 0) + 1);
  for (let place = 1; place <= places; place++) {
    const count = byPlace.get(place) ?? 0;
    if (count === 0) throw new Error('size coverage names no bins');
    if (count > edges.length + 1) throw new Error('size coverage has too many bins');
  }
}

function debugHealth(): void {
  if (process.env.MAKEFUN_DEBUG !== '1') return;
  const health = forecastHealth();
  validateHealth(health);
  console.error(JSON.stringify({ forecastHealth: health, reserve: reserveDiagnostics() }));
}

function validateForecast(): void {
  validateOrderRows(probabilities);
  top5.splice(0, top5.length, ...stableTop5());
  validateTop5(top5);
  validateSizeRows(sizes);
  validateKnownMints();
  validateOrderCoverage();
  validateSizeCoverage();
  if (probabilities.length > places * 100) throw new Error('too many order rows');
  if (sizes.length > places * (edges.length + 1)) throw new Error('too many size rows');
}

validateForecast();
debugHealth();
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
        samples: observationCount,
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
