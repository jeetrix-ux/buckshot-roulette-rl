"""Head-to-head evaluation with seat rotation.

    python train/eval.py my_bot train/opponents/naive_prob ... [--games 400] [--k 4]

Each game seats a random k-subset of the given bots (first bot always included) in a random order;
reports average placement points, win rate, faults and the worst per-match decision time.
"""
import argparse
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

REPO = sim.ROOT
LEAGUE = [os.path.join(REPO, "bots", "random_bot")] + [
    os.path.join(REPO, "train", "opponents", d) for d in ("naive_prob", "aggressive", "cautious")]


def _job(args):
    specs, names, seed = args
    r = sim.play_match(specs, seed, timing=True)
    return names, r


def evaluate(entries, games, k, seed=0, workers=None, fixed_k=True):
    """entries: list of (name, spec). entries[0] is the subject; returns per-name stats."""
    rng = random.Random(seed)
    jobs = []
    for g in range(games):
        kk = k if fixed_k else rng.choice((2, 3, 4))
        others = [rng.choice(entries[1:]) for _ in range(kk - 1)]
        table = [entries[0]] + others
        rng.shuffle(table)
        jobs.append(([s for _, s in table], [n for n, _ in table], rng.randrange(2**31)))
    stats = defaultdict(lambda: {"n": 0, "pts": 0.0, "wins": 0, "faults": 0, "max_t": 0.0, "max_dec": 0})
    with Pool(workers) as pool:
        for names, r in pool.imap_unordered(_job, jobs, chunksize=8):
            for seat, name in enumerate(names):
                s = stats[name]
                s["n"] += 1
                s["pts"] += r["points"][seat] / (len(names) - 1)  # normalised to [-1, 1]
                s["wins"] += r["placements"][seat] == 1
                s["faults"] += len(r["faults"][seat])
                s["max_t"] = max(s["max_t"], r["spent"][seat])
                s["max_dec"] = max(s["max_dec"], r["decisions"][seat])
                if r["faults"][seat] and s["faults"] <= 3:
                    print(f"  fault {name}: {r['faults'][seat][0]}")
    return stats


def print_stats(stats):
    print(f"{'bot':<16}{'games':>7}{'pts':>8}{'win%':>7}{'faults':>8}{'max_ms':>8}{'max_dec':>8}")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["pts"] / kv[1]["n"]):
        print(f"{name:<16}{s['n']:>7}{s['pts'] / s['n']:>8.3f}{100 * s['wins'] / s['n']:>7.1f}"
              f"{s['faults']:>8}{1000 * s['max_t']:>8.1f}{s['max_dec']:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="*", help="bot directories; first is the subject (default my_bot + league)")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--k", type=int, default=4, help="players per table (0 = mixed 2/3/4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = args.bots or [os.path.join(REPO, "my_bot")] + LEAGUE
    entries = [(os.path.basename(os.path.normpath(p)), (os.path.abspath(p), None)) for p in paths]
    stats = evaluate(entries, args.games, args.k or 4, args.seed, fixed_k=bool(args.k))
    print_stats(stats)


if __name__ == "__main__":
    main()
