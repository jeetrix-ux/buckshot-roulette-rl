"""Decide between two weight files: identical tables for both (common random numbers), then head-to-head.

    python train/compare.py A.json B.json [--games 6000]

Prints each one's score vs the baseline league (4-player and mixed tables) and the result of
4-player tables seating two copies of A against two copies of B.
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402
from eval import LEAGUE, REPO  # noqa: E402

MY_BOT = os.path.join(REPO, "my_bot")


def _chunk(args):
    params_list, tables = args
    out = []
    for seats, slots, seed in tables:
        specs = list(seats)
        for slot, which in slots:  # place our candidates into their seats
            specs.insert(slot, (MY_BOT, params_list[which]))
        r = sim.play_match(specs, seed, roundtrip=False)
        k = len(specs)
        # slots are inserted in increasing order, so each stored index is that seat's final position
        out.append([(which, r["points"][slot] / (k - 1), r["placements"][slot] == 1) for slot, which in slots])
    return out


def tables_vs_league(rng, games, k_choices, same=False):
    tables = []
    for _ in range(games):
        k = rng.choice(k_choices)
        one = rng.choice(LEAGUE)
        seats = [(one if same else rng.choice(LEAGUE), {"seed": rng.randrange(10**9)}) for _ in range(k - 1)]
        tables.append((seats, rng.randrange(k), rng.randrange(2**31)))
    return tables


def run(pool, params_list, tables_spec):
    jobs = []
    for i in range(0, len(tables_spec), 100):
        jobs.append((params_list, tables_spec[i:i + 100]))
    rows = [r for chunk in pool.map(_chunk, jobs) for r in chunk]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=99, help="table stream (change it for an independent re-test)")
    args = ap.parse_args()
    params = []
    for p in (args.a, args.b):
        with open(p) as f:
            params.append(json.load(f))
    rng = random.Random(args.seed)
    from cem import no_throttle
    no_throttle()
    with Pool(initializer=no_throttle) as pool:
        for label, kc, same in (("4p vs league", (4,), False), ("mixed 2/3/4p vs league", (4, 4, 3, 2), False),
                                ("4p vs one style x3", (4,), True)):
            base = tables_vs_league(rng, args.games, kc, same)
            for which, name in ((0, args.a), (1, args.b)):
                spec = [(seats, [(slot, which)], seed) for seats, slot, seed in base]
                rows = run(pool, params, spec)
                pts = sum(r[0][1] for r in rows) / len(rows)
                win = sum(r[0][2] for r in rows) / len(rows)
                print(f"{label:<24} {os.path.basename(name):<22} pts {pts:+.3f}  win {100 * win:.1f}%", flush=True)
        # head-to-head: two A and two B at a 4-player table, all seatings rotated by random draw
        h2h = []
        for _ in range(args.games // 2):
            order = [0, 0, 1, 1]
            rng.shuffle(order)
            h2h.append(([], sorted(enumerate(order)), rng.randrange(2**31)))
        rows = run(pool, params, h2h)
        tot = {0: [0.0, 0, 0], 1: [0.0, 0, 0]}
        for row in rows:
            for which, pts, win in row:
                tot[which][0] += pts
                tot[which][1] += win
                tot[which][2] += 1
        for which, name in ((0, args.a), (1, args.b)):
            t = tot[which]
            print(f"{'2v2 head-to-head':<24} {os.path.basename(name):<22} pts {t[0] / t[2]:+.3f}  "
                  f"win {100 * t[1] / t[2]:.1f}% (per seat)")


if __name__ == "__main__":
    main()
