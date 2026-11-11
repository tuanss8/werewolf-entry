# Werewolf agent (werewolf-agent-01)

A pure-heuristic Werewolf agent for make.fun's `werewolf-agent-01` bounty.
No model calls: it plays by deduction from the public game record.

## How it plays

- **Tracks everything public**: every vote, death, revealed role, speech and
  role claim in the game's log.
- **Suspicion scoring**: a seat that voted to exile a revealed innocent
  gains suspicion; one that voted out a revealed wolf loses it. A seer claim
  that named a revealed wolf is trusted; one that named a revealed innocent
  is treated as a lie. Silent seats gain a little suspicion.
- **Seer**: inspects the most suspicious uninspected seat; when it finds a
  werewolf it reveals itself in the debate and names the wolf.
- **Doctor**: protects a credible seer claimant first, else the loudest
  villager (the wolves' likeliest target), never the previous night's seat.
- **Werewolf**: coordinates the kill — it agrees with its partner's pick
  when the partner already named one; otherwise it kills a credible seer
  claimant, then the loudest villager. By day it blends in, backing the
  village's suspicion and never voting its partner.
- **Village vote**: backs a credible seer claim, votes a seer-confirmed wolf,
  otherwise votes the most suspicious living seat; on day 1 with no evidence
  it votes the quietest seat.

## Determinism

Every tie-break and statement template is chosen by a seeded PRNG
(mulberry32 over `MAKEFUN_TRIAL_SEED:game:decision:logLength:day`). No wall
clock, no `Math.random`, no cross-game state: each decision is a function of
its own game's view alone, so a replay sends the same requests and plays the
same games.

## Run

`node main.ts` with `MAKEFUN_ENV_URL` (and `MAKEFUN_PROMPT_FILE`) set.
Node 22 standard library only.
