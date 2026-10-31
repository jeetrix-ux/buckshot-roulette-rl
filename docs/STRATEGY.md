# Strategy and training

## 1. The edge: the gun is not shuffled uniformly
`game_API._setup_round` builds the shell order by flipping a fair coin for each position while both
shell types remain, then fills the rest with whatever is left. So, given the shells already removed:

| remaining shells | true P(next live) | naive lives/shells |
|---|---|---|
| both types left | **0.50** | anything from 0.03 to 0.97 |
| no blanks left | 1.00 | 1.00 |
| no lives left | 0.00 | 0.00 |

Checked over 20,000 seeded games: remaining (5 live, 1 blank) gives 0.485 live, not 0.83. Remaining
(1, 5) gives 0.49, not 0.17. A bot that believes lives/shells shoots itself when it shouldn't and
wastes saws on shells that are really coin flips.

Other players' coffee removes shells we never see (`ejections.unknowns_ejected`). We marginalise
over every possible split of those shells, weighting each split by its binomial count. Histories
where one type is already exhausted get an extra weight of 2.5 (fitted). Those histories include
forced, probability-1 steps, so they are likelier under the coin-flip generator.

Calibration (`train/test_belief.py`, about 130k predictions in coffee-heavy random games):

| model | Brier score |
|---|---|
| ours | **0.190** |
| naive lives/shells | 0.214 |

## 2. Policy structure (runtime: pure Python, about 0.2 ms per decision)
Play-turn priority ladder. Each call returns one action, and the bot is called again after every upgrade use.
1. Drink juice when health < `juice_below`.
2. Pool floatie on the highest-scoring opponent (next to act, threat, health) if the score > `f_use`.
3. If the next shell is uncertain: magnifying glass (gated by `mag_min_saws` / `mag_keep`); else coffee when at
   most `coffee_rounds` shells are left.
4. Certain blank: shoot self (keeps the turn).
4b. Uncertain shell, but our saws can kill a rival and P(live) ≥ `p_saw_lethal` (run 5: 0.38, so every
   50/50): stack saws to exactly lethal and shoot that rival. Half the time it is a kill.
5. Certain live: pick the target by linear score (lethal-with-saws counts strongly), stack hand saws up to
   exactly lethal (or up to `saw_nonlethal`), then shoot.
6. Uncertain: shoot self only if P(live) < `p_self`, otherwise shoot the best target.

Upgrade picks: argmax of `u_type + h_type·held + u_hurt·missing_health (juice) + u_deny·…`, plus the
saw-combo terms (run 5): `u_combo_mag` for a glass while holding a saw, `u_combo_saw` for a saw while
holding a glass, and `u_saw_reach` for a saw while our saws can't yet one-shot the weakest rival.

**Health-aware defence (added in run 4).** Most bots shoot whoever has the lowest health, so having
the lowest health at the table ("danger") is what gets us knocked out. In danger, the target score
gains `t_danger_health·health` and `t_divert` for a hit that drops the target below us (the focus
fire moves to them), the floatie score gains `f_danger`, and juice and floatie picks gain
`u_danger_juice` and `u_danger_floatie`.

Safety: every action is checked for legality before it is returned. Exceptions fall back to a trivial
legal shot. A compute-time guard and a lean mode drop optional upgrade actions after 100 decisions,
because Docker IPC costs about 2 ms per request against the 0.5 s match budget.

## 3. Training: Cross-Entropy Method in a league (`train/cem.py`)
- **Policy parameters:** 44 weights and thresholds (`DEFAULTS` in `my_bot/bot.py`): 28 at first, then five
  defence weights in run 4, six saw-combo weights in run 5 and five threat weights in run 6 (unused). New weights default to "off", so a run
  starts from exactly the previous policy (checked: 0 of 400 games differ).
- **Environment:** `train/sim.py`, an in-process copy of the tournament loop: the same `game_API`, the same
  fallback and strike rules, the same placement points. About 5,000 games/s on 22 cores.
- **Fitness:** mean placement points / (k−1), over mixed tables (4p ×4, 3p, 2p) with random seating.
- **League:**
  - baselines `random_bot`, `naive_prob`, `aggressive` and `cautious`
  - frozen snapshots of our own mean policy (one added every 3 generations, the last 6 kept)
  So the policy has to beat both simple opponents and itself.
- **Variance reduction:** common random numbers. All candidates in a generation play identical
  tables and seeds. Until run 5, opponents without a params file (`aggressive`) seeded their dice from
  the clock, so 'identical' tables still differed in about 1 game in 5; they now get a seed derived from
  the table seed.
- **Update:** keep the top 8 of 48, refit a diagonal Gaussian, and smooth sigma (0.7 old + 0.3 elite std).
- **Benchmark:** a fixed set of seeded tables (4-player vs baselines, vs the untrained bot, 1v1 duels, and
  from run 3 on, one opponent style filling the table), scored every generation. The training score
  itself drifts down as frozen copies of the bot join the league, so it is not a measure of progress.
- **Resuming:** `--resume` continues from `train/cem_history.json`. Weights added later are back-filled as
  constants in earlier generations. Windows power throttling (EcoQoS) slows long background runs about
  10×, so the trainer and its workers opt out of it at startup.

## 4. Results
Five runs, 190 generations in all. Each run continues the previous one's distribution.

| Run | Generations | Change | Outcome |
|---|---|---|---|
| 1 | 0–59 | 1,500 tables per candidate | 4p vs baselines +0.07 → **+0.52** points, win rate 26% → 67.5% |
| 2 | 60–99 | σ ×2, 3,000 tables | no measurable gain: converged |
| 3 | 100–129 | 40% single-style tables | tie on mixed tables; 3× `random_bot` 31% → 35% |
| 4 | 130–159 | five health-aware defence weights | **clear gain** (below) |
| 5 | 160–189 | six saw-combo weights, seeded opponents | **gain**, mostly head-to-head (below). **Shipped.** |
| 6 | 190–219 | five threat-model weights (who shoots us next) | slightly worse vs the league: **not shipped** |

Run 3 vs run 4 on identical tables (`train/compare.py`, 6,000 tables per setting):

| Setting | Run 3 | Run 4 |
|---|---|---|
| 4p vs league | +0.506 | **+0.552** |
| mixed 2/3/4p vs league | +0.495 | **+0.531** |
| 4p, one style ×3 | +0.326 | **+0.373** |
| 2v2 head-to-head, win per seat | 19.9% | **30.1%** |

Run 4 learned `t_danger_health` −6.1 and `t_divert` +1.5: when it has the lowest health, it shoots the
weakest rival. Over 900 four-player games, last places fell from 136 to 102.

Run 4 vs run 5, re-tested on a second, independent set of tables (`--seed 7`) to rule out noise:

| Setting | Run 4 | Run 5 (tables 99) | Run 4 | Run 5 (tables 7) |
|---|---|---|---|---|
| 4p vs league | +0.552 | **+0.556** | +0.530 | **+0.540** |
| mixed 2/3/4p vs league | +0.531 | **+0.542** | +0.544 | **+0.561** |
| 4p, one style ×3 | +0.373 | **+0.387** | +0.382 | **+0.392** |
| 2v2 head-to-head, win per seat | 22.7% | **27.3%** | 22.6% | **27.4%** |

Ablations (run 5 with one part switched off, tables 7):
- **Without the 50/50 all-in:** about the same vs the league, but the head-to-head drops from 28.6% to
  21.4% win per seat. Against a copy of itself, a lethal coin flip is the best shot on the table.
- **Without the combo picks:** slightly worse vs the league (−0.013 mixed), even head-to-head.
- **Glass hoarding** was learned off (`mag_keep` 0.18: one glass is enough to use it).

Run 6 tried a threat model: per rival, whether we are its lowest-health target (`t_aims_us`), how soon
it acts (`t_soon`), and whether its saws plus a live shell would kill us (`t_lethal_us`), plus the same
threats for the floatie. It learned to float the rivals who threaten us and shoot the ones who act
soonest. It beat run 5 head-to-head (26.0% / 26.9% vs 24.0% / 23.1% win per seat), but it trained
against copies of run 5, and against the independent baseline league it was slightly worse on both
table sets (mixed −0.015 / −0.010 pts). So run 5 stays shipped. The code stays in the bot with the
five weights at 0, which plays identically to run 5.

What the trained bot does (`train/replay.py`, 900 four-player games vs the baselines):
- **Wins:** 68% of games; 96% of wins end with its own final shot.
- **Picks:** juice first.
- **Shooting:** it almost never shoots itself on a 50/50.
- **Kills:** 48% on a shell it knows is live, 52% on 50/50 shots.
- **Saws (run 5):** 325 all-in saw shots on a 50/50 shell in 900 games; 102 of 843 kills (12%) used saws.

Where a naive bot would have shot itself, the shell was live 47% of the time.

Known limit: `random_bot` already shoots opponents on every uncertain shell, which is correct under the
coin-flip loading rule. So the bot's edge is small at a table of three random bots: about 35% wins,
against 25% for an even share.

Visualiser: `python train/replay.py && python viz/build.py` writes `viz/buckshot_lab.html`.
