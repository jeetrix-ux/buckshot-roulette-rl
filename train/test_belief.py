"""Calibration of my_bot.next_live_prob against the real game_API shell generator.

    python train/test_belief.py

Plays random games where everyone drinks coffee often (so unknown ejections happen), and at every
play turn compares the predicted P(next live) with the actual next shell. Prints calibration by
predicted bin, for our model and for the naive lives/shells model.
"""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_bot"))
import game_API as api  # noqa: E402
from bot import next_live_prob  # noqa: E402
from schemas import *  # noqa: E402


def naive(obs):
    ej = obs.ejections
    lives = obs.lives_init - obs.lives_shot - ej.lives_ejected
    total = obs.rounds_remaining + ej.unknowns_ejected
    return lives / total if total else 0.5


def main(games=4000):
    rng = random.Random(0)
    bins = {"ours": defaultdict(lambda: [0, 0]), "naive": defaultdict(lambda: [0, 0])}
    brier = {"ours": 0.0, "naive": 0.0}
    n = 0
    for seed in range(games):
        s = api.initGame(seed, rng.choice((2, 3, 4)))
        steps = 0
        while not s.game_over and steps < 300:
            steps += 1
            if s.stage == Stage.PICK_UPGRADE:
                a = Action_PICK_UPGRADES()
                av = s.available_upgrades
                a.chosen_upgrade = UpgradeType.COFFEE if av.coffee else next(
                    t for t, k in ((UpgradeType.JUICE, "juice"), (UpgradeType.MAG_GLASS, "mag_glass"),
                                   (UpgradeType.POOL_FLOATIE, "pool_floatie"), (UpgradeType.HAND_SAW, "hand_saw"))
                    if getattr(av, k))
                api.step_PICK_UPGRADE(s, a)
                continue
            obs = api.play_observation(s)
            if obs.next_round == NextRound.UNKNOWN:
                actual = s.round_contents[len(s.round_contents) - s.rounds_remaining]
                for name, f in (("ours", next_live_prob), ("naive", naive)):
                    p = f(obs)
                    b = bins[name][round(p, 1)]
                    b[0] += actual
                    b[1] += 1
                    brier[name] += (p - actual) ** 2
                n += 1
            me = s.turn
            a = Action_PLAY()
            a.useUpgrade, a.target = None, None
            if s.players[me].upgrades.coffee and rng.random() < 0.6:
                a.action, a.useUpgrade = ActionType.USE_UPGRADE, UpgradeType.COFFEE
            else:
                a.action = ActionType.SHOOT
                a.target = rng.choice([i for i, p in enumerate(s.players) if p.health > 0])
            api.step_PLAY(s, a)
    out = {}
    for name in ("ours", "naive"):
        print(f"{name}: Brier {brier[name] / n:.4f} over {n} predictions")
        out[name] = {"brier": round(brier[name] / n, 4), "bins": []}
        for p in sorted(bins[name]):
            hit, cnt = bins[name][p]
            print(f"   predicted {p:.1f}  actual {hit / cnt:.3f}  (n={cnt})")
            out[name]["bins"].append([p, round(hit / cnt, 4), cnt])
    return out


if __name__ == "__main__":
    result = main()
    if len(sys.argv) > 1:  # optional output path for the visualiser
        import json
        with open(sys.argv[1], "w") as f:
            json.dump(result, f)
