"""Ablations: my_bot with parameter overrides vs the league.

    python train/ablate.py [--games 2000] [--k 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E  # noqa: E402

VARIANTS = [
    ("base", {}),
    ("juice-first picks", {"u_juice": 5, "h_juice": 0}),
    ("juice-first, no floatie", {"u_juice": 5, "h_juice": 0, "f_use": 99}),
    ("juice-first, juice only <=2hp", {"u_juice": 5, "h_juice": 0, "juice_below": 3}),
    ("juice-first, flat targeting", {"u_juice": 5, "h_juice": 0, "t_next": 0, "t_health": 0, "t_kill": 0,
                                     "t_saw": 0, "t_mag": 0, "t_upg": 0}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    for label, params in VARIANTS:
        entries = [("me", (os.path.join(E.REPO, "my_bot"), params))] + [
            (os.path.basename(p), (p, None)) for p in E.LEAGUE]
        st = E.evaluate(entries, args.games, args.k)
        me, rnd = st["me"], st["random_bot"]
        print(f"{label:<34} me {me['pts'] / me['n']:+.3f} (win {100 * me['wins'] / me['n']:.1f}%)"
              f"   random_bot {rnd['pts'] / rnd['n']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
