// werewolf-agent-01 entry: a pure-heuristic Werewolf agent (no model calls).
// It plays by deduction, not by LLM: it tracks every public vote, death and
// claim, scores suspicion, and acts on that. Deterministic given the env's
// views and MAKEFUN_TRIAL_SEED: every tie-break and template choice comes
// from a seeded PRNG, never the wall clock or Math.random.
// Node 22 standard library only. Run by the grader as `node /bot/main.ts`.

import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

/** The most decisions in flight at once: the env's live games. */
export const CONCURRENCY = 10;

type Seat = number;
type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: number;
  target?: number | null;
  text?: string;
  died?: number | null;
  role?: string | null;
  tally?: { seat: number; votes: number }[];
  [k: string]: unknown;
}

interface Player {
  seat: Seat;
  name: string;
  alive: boolean;
  role: string | null;
}

interface View {
  day: number;
  phase: string;
  you: { seat: Seat; name: string; role: Role; alive: boolean; partner: { seat: Seat; name: string; alive: boolean } | null };
  players: Player[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat } | null;
  wolfVotes: { night: number; seat: Seat; target: number | null }[] | null;
  log: LogEntry[];
  outcome: { winner: string | null; reason: string; day: number; winners: number[] } | null;
  [k: string]: unknown;
}

export interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

/** FNV-1a hash of a string to a 32-bit unsigned int. */
function hash32(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** mulberry32: a small deterministic PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The PRNG for one decision: a function of the trial seed, the game, and how far its log has come. */
function rngFor(trialSeed: string, p: Pending): () => number {
  const key = `${trialSeed}:${p.game}:${p.decision}:${p.view.log.length}:${p.view.day}`;
  return mulberry32(hash32(key));
}

/** Pick one element of a non-empty array with a PRNG. */
function pick<T>(rng: () => number, arr: readonly T[]): T {
  return arr[Math.floor(rng() * arr.length)]!;
}

/** A statement folded to one line the env's way. */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

export function fallbackFor(p: Pick<Pending, 'game' | 'decision'>): Record<string, unknown> {
  return p.decision === 'speak' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: p.decision, target: null };
}

// ---------------------------------------------------------------------------
// Game analysis
// ---------------------------------------------------------------------------

interface Claim {
  seer?: boolean;
  doctor?: boolean;
  inspected?: Map<Seat, boolean>;
}

interface Analysis {
  me: Seat;
  role: Role;
  partner: Seat | null;
  alive: Seat[];
  revealed: Map<Seat, Role>;
  votes: Map<number, Map<Seat, Seat | null>>;
  speechCount: Map<Seat, number>;
  claims: Map<Seat, Claim>;
  knownWolves: Seat[];
  knownVillagers: Seat[];
  inspected: Set<Seat>;
  claimedWolf: Seat | null;
  seerClaimants: Seat[];
  doctorClaimants: Seat[];
}

const SEER_RE = /\bseer\b/i;
const DOCTOR_RE = /\bdoctor\b/i;
const IAM_RE = /\b(i am|i'm|im)\b/i;
const CHECKED_RE = /\b(check|checked|inspect|inspected|looked at|scry)\b[^0-9]{0,20}?(\d)/i;
const WOLF_WORD_RE = /\b(wolf|werewolf|wolves)\b/i;
const VILLAGER_WORD_RE = /\b(villager|villagers|innocent|town|not a wolf|not the wolf|clear)\b/i;

function parseSpeech(text: string): Claim {
  const c: Claim = {};
  const lower = text.toLowerCase();
  if (SEER_RE.test(lower) && IAM_RE.test(lower)) c.seer = true;
  if (DOCTOR_RE.test(lower) && IAM_RE.test(lower)) c.doctor = true;
  const m = CHECKED_RE.exec(text);
  if (m) {
    const seat = Number(m[2]);
    if (seat >= 1 && seat <= 8) {
      const after = text.slice(m.index + m[0].length - 2);
      let werewolf: boolean | null = null;
      if (WOLF_WORD_RE.test(after)) werewolf = true;
      else if (VILLAGER_WORD_RE.test(after)) werewolf = false;
      else if (WOLF_WORD_RE.test(lower)) werewolf = true;
      else if (VILLAGER_WORD_RE.test(lower)) werewolf = false;
      if (werewolf !== null) {
        c.inspected = new Map([[seat, werewolf]]);
      }
    }
  }
  return c;
}

function analyze(view: View): Analysis {
  const me = view.you.seat;
  const role = view.you.role;
  const partner = view.you.partner ? view.you.partner.seat : null;
  const alive: Seat[] = view.players.filter((pl) => pl.alive).map((pl) => pl.seat);
  const revealed = new Map<Seat, Role>();
  for (const pl of view.players) {
    if (!pl.alive && pl.role) revealed.set(pl.seat, pl.role as Role);
  }
  const votes = new Map<number, Map<Seat, Seat | null>>();
  const speechCount = new Map<Seat, number>();
  const claims = new Map<Seat, Claim>();
  for (const e of view.log) {
    if (e.kind === 'dawn' && e.died != null && e.role) {
      revealed.set(e.died, e.role as Role);
    } else if (e.kind === 'exile' && e.seat != null && e.role) {
      revealed.set(e.seat, e.role as Role);
    } else if (e.kind === 'vote' && e.seat != null) {
      let dayMap = votes.get(e.day);
      if (!dayMap) { dayMap = new Map(); votes.set(e.day, dayMap); }
      dayMap.set(e.seat, (e.target as Seat | null) ?? null);
    } else if (e.kind === 'speech' && e.seat != null && typeof e.text === 'string') {
      speechCount.set(e.seat, (speechCount.get(e.seat) ?? 0) + 1);
      const c = parseSpeech(e.text);
      if (c.seer || c.doctor || c.inspected) {
        const prev = claims.get(e.seat) ?? {};
        if (c.seer) prev.seer = true;
        if (c.doctor) prev.doctor = true;
        if (c.inspected) {
          prev.inspected = prev.inspected ?? new Map();
          for (const [s, w] of c.inspected) prev.inspected.set(s, w);
        }
        claims.set(e.seat, prev);
      }
    }
  }
  const knownWolves: Seat[] = [];
  const knownVillagers: Seat[] = [];
  const inspected = new Set<Seat>();
  if (view.seerResults) {
    for (const r of view.seerResults) {
      inspected.add(r.seat);
      if (r.werewolf) knownWolves.push(r.seat);
      else knownVillagers.push(r.seat);
    }
  }
  const seerClaimants: Seat[] = [];
  const doctorClaimants: Seat[] = [];
  for (const [seat, c] of claims) {
    if (c.seer && view.players[seat - 1]?.alive) seerClaimants.push(seat);
    if (c.doctor && view.players[seat - 1]?.alive) doctorClaimants.push(seat);
  }
  let claimedWolf: Seat | null = null;
  for (const seat of seerClaimants) {
    const c = claims.get(seat)!;
    if (c.inspected) {
      for (const [target, werewolf] of c.inspected) {
        if (werewolf && view.players[target - 1]?.alive && revealed.get(target) !== 'villager' && revealed.get(target) !== 'seer' && revealed.get(target) !== 'doctor') {
          claimedWolf = target;
        }
      }
    }
  }
  return { me, role, partner, alive, revealed, votes, speechCount, claims, knownWolves, knownVillagers, inspected, claimedWolf, seerClaimants, doctorClaimants };
}

/**
 * Suspicion score for a living seat (higher = more likely a werewolf).
 * Built from public evidence only: votes against revealed innocents/wolves,
 * seer-claim accuracy, and silence. My own private knowledge (seer results)
 * overrides with +-100.
 */
function suspicion(a: Analysis, seat: Seat): number {
  let s = 0;
  for (const [, dayMap] of a.votes) {
    const target = dayMap.get(seat);
    if (target == null) continue;
    const r = a.revealed.get(target);
    if (r === 'werewolf') s -= 3;
    else if (r) s += 2;
  }
  const c = a.claims.get(seat);
  if (c?.inspected) {
    for (const [target, werewolf] of c.inspected) {
      const r = a.revealed.get(target);
      if (r === 'werewolf' && werewolf) s -= 5;
      else if (r && r !== 'werewolf' && werewolf) s += 5;
      else if (r && r !== 'werewolf' && !werewolf) s -= 2;
    }
  }
  if ((a.speechCount.get(seat) ?? 0) === 0) s += 1;
  if (a.knownWolves.includes(seat)) s += 100;
  if (a.knownVillagers.includes(seat)) s -= 100;
  return s;
}

const othersAlive = (a: Analysis): Seat[] => a.alive.filter((s) => s !== a.me);

function mostSuspicious(a: Analysis, rng: () => number, candidates: Seat[]): Seat | null {
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const s of candidates) {
    const sc = suspicion(a, s) + rng() * 0.5;
    if (sc > bestScore) { bestScore = sc; best = s; }
  }
  return best;
}

function loudest(a: Analysis, candidates: Seat[]): Seat | null {
  let best: Seat | null = null;
  let bestN = -1;
  for (const s of candidates) {
    const n = a.speechCount.get(s) ?? 0;
    if (n > bestN) { bestN = n; best = s; }
  }
  return best;
}

function quietest(a: Analysis, candidates: Seat[]): Seat | null {
  let best: Seat | null = null;
  let bestN = Infinity;
  for (const s of candidates) {
    const n = a.speechCount.get(s) ?? 0;
    if (n < bestN) { bestN = n; best = s; }
  }
  return best;
}

/** The player(s) the village currently votes for most (from the log's latest voting round). */
function voteLeader(a: Analysis, view: View): Seat | null {
  const tally = new Map<Seat, number>();
  let latestDay = -1;
  for (const e of view.log) {
    if (e.kind === 'vote' && e.target != null) {
      if (e.day > latestDay) { latestDay = e.day; tally.clear(); }
      if (e.day === latestDay) tally.set(e.target, (tally.get(e.target) ?? 0) + 1);
    }
  }
  let best: Seat | null = null;
  let bestN = 0;
  for (const [seat, n] of tally) {
    if (n > bestN && a.alive.includes(seat) && seat !== a.me) { bestN = n; best = seat; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

/** A villager-side statement: name the top suspect and the reason. */
function villagerStatement(a: Analysis, rng: () => number, suspect: Seat | null): string | null {
  const others = othersAlive(a);
  if (others.length === 0) return null;
  if (suspect === null) {
    return pick(rng, [
      'No strong read yet. Let us hear from everyone before we vote.',
      'I am a villager. I want to hear more before I point a finger.',
      'Too early to tell. Watch the votes today, that is where the truth shows.',
    ]);
  }
  const n = a.speechCount.get(suspect) ?? 0;
  // Find a concrete reason: did the suspect vote for a revealed innocent?
  let reason: string | null = null;
  for (const [, dayMap] of a.votes) {
    const t = dayMap.get(suspect);
    if (t != null) {
      const r = a.revealed.get(t);
      if (r && r !== 'werewolf') { reason = `seat ${suspect} voted to exile seat ${t}, who turned out to be ${r}`; break; }
      if (r === 'werewolf') { reason = null; break; }
    }
  }
  if (reason === null && n === 0) reason = `seat ${suspect} has said nothing at all`;
  const lines = reason
    ? [
        `I am suspicious of seat ${suspect}: ${reason}. I think we should look hard at them today.`,
        `Seat ${suspect} worries me. ${reason.charAt(0).toUpperCase() + reason.slice(1)}.`,
        `My vote goes to seat ${suspect}. ${reason.charAt(0).toUpperCase() + reason.slice(1)}, and that is not how a villager plays.`,
      ]
    : [
        `I am suspicious of seat ${suspect}. Their play does not add up to me.`,
        `Seat ${suspect} is my strongest suspicion right now. Watch how they vote today.`,
      ];
  return pick(rng, lines);
}

/** A werewolf's statement: blend in, agree with the village, never defend the partner openly. */
function wolfStatement(a: Analysis, rng: () => number, suspect: Seat | null, leader: Seat | null): string | null {
  const others = othersAlive(a).filter((s) => s !== a.partner);
  if (others.length === 0) return null;
  // If the partner is the village's top suspect, gently deflect to someone else.
  const target = suspect !== null && suspect !== a.partner ? suspect : (leader !== null && leader !== a.partner ? leader : null);
  if (target === null) {
    return pick(rng, [
      'I am just a villager. I have no read yet, I want to hear more.',
      'Let us not rush. A wrong exile helps the wolves more than waiting does.',
      'I agree we should watch the votes. That is where wolves slip up.',
    ]);
  }
  return pick(rng, [
    `I agree with the doubt on seat ${target}. Their voting record looks off to me.`,
    `Seat ${target} is my suspicion too. Let us hear their defence before we vote.`,
    `I have been watching seat ${target}. I think they are worth a vote today.`,
    `Seat ${target} voted oddly yesterday. I am leaning that way.`,
  ]);
}

function decideSpeak(a: Analysis, rng: () => number, view: View, maxChars: number): Record<string, unknown> {
  const others = othersAlive(a);
  let text: string | null = null;
  if (a.role === 'werewolf') {
    // A wolf must speak exactly like a villager: push a suspect WITH A REASON.
    // Only deflect when the partner is the village's top suspect; otherwise
    // argue a case against a villager the same way a villager would. Pure
    // agreement ("I agree with the doubt") reads as evasive to the other
    // players and got wolves exiled early (round 1 wolf score 133).
    const leader = voteLeader(a, view);
    const suspect = mostSuspicious(a, rng, others.filter((s) => s !== a.partner));
    const partnerHot = (leader !== null && leader === a.partner) || (suspect !== null && suspect === a.partner);
    if (partnerHot) {
      text = wolfStatement(a, rng, suspect, leader);
    } else {
      text = villagerStatement(a, rng, suspect);
    }
  } else if (a.role === 'seer') {
    const wolf = a.knownWolves.find((s) => a.alive.includes(s));
    if (wolf !== undefined) {
      text = pick(rng, [
        `I am the seer. I checked seat ${wolf} last night and they are a werewolf. Vote seat ${wolf}.`,
        `I am the seer, and seat ${wolf} is a werewolf. I checked them last night. We must exile seat ${wolf} today.`,
      ]);
    } else {
      const suspect = mostSuspicious(a, rng, others);
      text = villagerStatement(a, rng, suspect);
    }
  } else {
    // Villager or doctor: speak as a villager. If a credible seer claim named a wolf, back it.
    if (a.claimedWolf !== null && a.alive.includes(a.claimedWolf)) {
      const w = a.claimedWolf;
      text = pick(rng, [
        `Seat ${w} was named a werewolf by the seer claim. I believe it. Vote seat ${w}.`,
        `If the seer claim is true, seat ${w} is a wolf. I am voting seat ${w}.`,
      ]);
    } else {
      const suspect = mostSuspicious(a, rng, others);
      text = villagerStatement(a, rng, suspect);
    }
  }
  if (text === null) return { decision: 'speak', pass: true };
  const folded = foldStatement(text);
  if (folded === '') return { decision: 'speak', pass: true };
  const cut = [...folded].length > maxChars ? [...folded].slice(0, maxChars).join('').trim() : folded;
  return { decision: 'speak', text: cut };
}

function decideVote(a: Analysis, rng: () => number, options: Seat[]): Record<string, unknown> {
  if (options.length === 0) return { decision: 'vote', target: null };
  // Seer with a confirmed living wolf: vote it.
  if (a.role === 'seer') {
    const wolf = a.knownWolves.find((s) => options.includes(s));
    if (wolf !== undefined) return { decision: 'vote', target: wolf };
  }
  // A credible seer claim naming a living wolf: back it.
  if (a.role !== 'werewolf' && a.claimedWolf !== null && options.includes(a.claimedWolf)) {
    return { decision: 'vote', target: a.claimedWolf };
  }
  let candidates = options;
  if (a.role === 'werewolf') {
    // Never vote the partner; vote the villager the village suspects most.
    candidates = options.filter((s) => s !== a.partner);
    if (candidates.length === 0) return { decision: 'vote', target: null };
  }
  const target = mostSuspicious(a, rng, candidates);
  // If suspicion is flat (day 1, no evidence), vote the quietest player.
  if (target !== null) {
    const sc = suspicion(a, target);
    if (sc <= 0 && a.day <= 1) {
      const q = quietest(a, candidates);
      if (q !== null && (a.speechCount.get(q) ?? 0) === 0) return { decision: 'vote', target: q };
    }
    return { decision: 'vote', target };
  }
  return { decision: 'vote', target: null };
}

function decideKill(a: Analysis, rng: () => number, view: View, options: Seat[], trialSeed: string, game: number): Record<string, unknown> {
  if (options.length === 0) return { decision: 'kill', target: null };
  // Coordinate: if my partner already named a target tonight, agree with them.
  if (view.wolfVotes) {
    const tonight = view.wolfVotes.filter((v) => v.night === view.day && v.seat === a.partner && v.target != null);
    if (tonight.length > 0) {
      const t = tonight[tonight.length - 1]!.target!;
      if (options.includes(t)) return { decision: 'kill', target: t };
    }
  }
  const villagers = options.filter((s) => s !== a.partner);
  if (villagers.length === 0) return { decision: 'kill', target: null };
  // Priority 1: a credible seer claimant.
  const seer = a.seerClaimants.find((s) => villagers.includes(s));
  if (seer !== undefined) return { decision: 'kill', target: seer };
  // Priority 2: someone who named a wolf correctly (likely the seer, unclaimed).
  for (const [seat, c] of a.claims) {
    if (!villagers.includes(seat)) continue;
    if (c.inspected) {
      for (const [t, w] of c.inspected) {
        if (w && a.revealed.get(t) === 'werewolf') return { decision: 'kill', target: seat };
      }
    }
  }
  // Priority 3: the loudest villager (they drive the village), but not always —
  // sometimes take a quiet one so the doctor cannot guess the target.
  // The 70/30 split is decided by a hash of (trial seed, game, day) so both
  // wolves compute the SAME choice even when they decide simultaneously and
  // have not yet seen each other's vote; an rng() call here could desync
  // them because their rng states diverge after different speak/vote calls.
  const splitHash = hash32(`${trialSeed}:${game}:${view.day}`);
  if (splitHash % 10 < 7) {
    const loud = loudest(a, villagers);
    if (loud !== null) return { decision: 'kill', target: loud };
  }
  return { decision: 'kill', target: pick(rng, villagers) };
}

function decideProtect(a: Analysis, rng: () => number, view: View, options: Seat[]): Record<string, unknown> {
  const last = view.lastProtect?.seat ?? null;
  const candidates = options.filter((s) => s !== last);
  if (candidates.length === 0) return { decision: 'protect', target: null };
  // Protect a credible seer claimant first.
  const seer = a.seerClaimants.find((s) => candidates.includes(s));
  if (seer !== undefined) return { decision: 'protect', target: seer };
  // Else the loudest villager (the wolves' likeliest target).
  const loud = loudest(a, candidates);
  if (loud !== null) return { decision: 'protect', target: loud };
  return { decision: 'protect', target: pick(rng, candidates) };
}

function decideInspect(a: Analysis, rng: () => number, options: Seat[]): Record<string, unknown> {
  const uninspected = options.filter((s) => !a.inspected.has(s));
  const pool = uninspected.length > 0 ? uninspected : options;
  if (pool.length === 0) return { decision: 'inspect', target: null };
  const target = mostSuspicious(a, rng, pool);
  if (target !== null) return { decision: 'inspect', target };
  return { decision: 'inspect', target: pick(rng, pool) };
}

/** The action for one pending decision, computed purely from its game's view. */
function decide(trialSeed: string, p: Pending): Record<string, unknown> {
  const view = p.view;
  const a = analyze(view);
  const rng = rngFor(trialSeed, p);
  const options = (p.options ?? []).filter((s) => a.alive.includes(s) && s !== a.me);
  switch (p.decision) {
    case 'speak': {
      const act = decideSpeak(a, rng, view, p.max_chars ?? 500);
      return { game: p.game, ...act };
    }
    case 'vote': return { game: p.game, ...decideVote(a, rng, options) };
    case 'kill': return { game: p.game, ...decideKill(a, rng, view, options, trialSeed, p.game) };
    case 'protect': return { game: p.game, ...decideProtect(a, rng, view, options) };
    case 'inspect': return { game: p.game, ...decideInspect(a, rng, options) };
  }
}

// ---------------------------------------------------------------------------
// Run loop (env protocol, as the starter)
// ---------------------------------------------------------------------------

interface Reply {
  status: number;
  body: unknown;
}

async function readReply(res: Response): Promise<Reply> {
  const text = await res.text();
  let body: unknown = null;
  try { body = JSON.parse(text); } catch { body = text; }
  return { status: res.status, body };
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export interface RunOptions {
  envUrl?: string;
  brief?: string;
  log?: (line: string) => void;
}

export interface RunSummary {
  decisions: number;
}

export async function run(o: RunOptions = {}): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (o.envUrl ?? env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const trialSeed = env.MAKEFUN_TRIAL_SEED ?? 'werewolf-agent-01';
  const log = o.log ?? ((line: string) => process.stdout.write(`${line}\n`));
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  // The brief is read for completeness (the rules live in it), but decisions
  // are computed from the views alone.
  if (o.brief === undefined) {
    const file = env.MAKEFUN_PROMPT_FILE ?? '';
    if (file !== '') {
      try { readFileSync(file, 'utf8'); } catch { /* the brief is not needed to play */ }
    }
  }
  const summary: RunSummary = { decisions: 0 };
  const envPost = async (path: string, body: unknown): Promise<Reply> =>
    readReply(await fetch(`${envUrl}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));

  const decideOne = async (p: Pending): Promise<void> => {
    summary.decisions += 1;
    let act: Record<string, unknown>;
    try {
      act = decide(trialSeed, p);
    } catch {
      act = fallbackFor(p);
    }
    // The decision must name the game and the kind exactly as /observe listed it.
    const body: Record<string, unknown> = { game: p.game, decision: p.decision };
    if (p.decision === 'speak') {
      if ('text' in act && typeof act.text === 'string') body.text = act.text;
      else body.pass = true;
    } else {
      body.target = ('target' in act ? act.target : null) as number | null;
    }
    let posted = await envPost('/act', body);
    if (posted.status === 400 && JSON.stringify(body) !== JSON.stringify({ game: p.game, decision: p.decision, ...(p.decision === 'speak' ? { pass: true } : { target: null }) })) {
      const fb = fallbackFor(p);
      const fbBody: Record<string, unknown> = { game: p.game, decision: p.decision };
      if (p.decision === 'speak') fbBody.pass = true;
      else fbBody.target = null;
      void fb;
      posted = await envPost('/act', fbBody);
    }
    if (posted.status !== 200) {
      // A refusal changes nothing; the decision will be listed again. Give the
      // env a moment rather than spinning.
      await sleep(100);
    }
  };

  const keyOf = (p: Pending): string => `${p.game}:${p.decision}:${p.view.log.length}`;
  const inFlight = new Map<string, Promise<void>>();
  const started = new Set<string>();
  const errors: string[] = [];
  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    for (const p of obs.pending) {
      const key = keyOf(p);
      if (started.has(key) || inFlight.size >= CONCURRENCY) continue;
      started.add(key);
      const task = decideOne(p)
        .catch((err: unknown) => { errors.push(`game ${p.game}: ${(err as Error).message}`); })
        .finally(() => inFlight.delete(key));
      inFlight.set(key, task);
    }
    const waiting = obs.pending.some((p) => !started.has(keyOf(p)));
    if (inFlight.size > 0 && (inFlight.size >= CONCURRENCY || !waiting)) await Promise.race(inFlight.values());
    else if (inFlight.size === 0 && obs.pending.length > 0 && !waiting) await sleep(250);
    if (errors.length > 0) break;
  }
  await Promise.all(inFlight.values());
  if (errors.length > 0) throw new Error(`the env failed: ${errors.join('; ')}`);
  log(`werewolf: ${summary.decisions} decisions, all heuristic (no model calls)`);
  return summary;
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
