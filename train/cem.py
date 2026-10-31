"""Cross-Entropy Method over my_bot's weights, trained in a league of baselines and self-snapshots.

    python train/cem.py [--gens 40] [--pop 48] [--games 600] [--out my_bot/params.json]

Each generation samples `pop` weight vectors from a diagonal Gaussian, scores every candidate on
the same tables and seeds (common random numbers), and refits the Gaussian to the top `elite`.
Opponents are the fixed league plus frozen snapshots of earlier generation means, so the policy
keeps improving against itself instead of overfitting the baselines.
Fitness = mean placement points normalised by (k-1), over mostly 4-player and some 2/3-player tables.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402
from eval import LEAGUE, REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "my_bot"))
from bot import DEFAULTS  # noqa: E402

MY_BOT = os.path.join(REPO, "my_bot")
KEYS = sorted(DEFAULTS)
# (low, high, initial sigma) per parameter; anything not listed is a free real weight.
BOUNDS = {
    "p_self": (0.0, 0.5, 0.1),
    "p_saw": (0.4, 1.01, 0.15),
    "saw_nonlethal": (0, 6, 1.5),
    "coffee_rounds": (0, 6, 1.5),
    "juice_below": (1, 5.5, 1.0),
    "mag_min_saws": (0, 4, 1.0),
    "mag_keep": (0, 3, 1.0),
    "p_saw_lethal": (0.3, 1.01, 0.2),
}
FREE_SIGMA = 1.0
TABLE_SIZES = (4, 4, 4, 4, 3, 2)  # mostly 4-player, as in the tournament default


def no_throttle():
    """Opt this process out of Windows power throttling (EcoQoS). Without it, a long run launched from a
    terminal that loses focus gets parked on the low-power cores and slows down by ~10x."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    execution_speed, process_power_throttling = 0x1, 4  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED, ProcessPowerThrottling
    state = State(1, execution_speed, 0)  # control execution speed, state off = never throttle
    k32 = ctypes.windll.kernel32
    k32.SetProcessInformation(k32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state),
                              ctypes.sizeof(state))


def clip(key, value):
    lo, hi, _ = BOUNDS.get(key, (-10.0, 10.0, FREE_SIGMA))
    return min(hi, max(lo, value))


def to_params(vec):
    return {k: round(clip(k, v), 4) for k, v in zip(KEYS, vec)}


def _seeded(seats, seed):
    """Opponents given no params get a seed derived from the table seed, so every candidate meets the
    same opponent dice on a table (aggressive.py would otherwise seed from the clock)."""
    return [(s[0], {"seed": seed + 7919 * j}) if s[1] is None else s for j, s in enumerate(seats)]


def _score(args):
    params, tables = args
    me = (MY_BOT, params)
    total = 0.0
    for seats, my_seat, seed in tables:
        specs = _seeded(seats, seed)
        specs.insert(my_seat, me)
        r = sim.play_match(specs, seed, roundtrip=False)
        total += r["points"][my_seat] / (len(specs) - 1)
    return total / len(tables)


def _bench_chunk(args):
    """(label, params, tables) -> (label, points sum, wins, games) for a fixed benchmark slice."""
    label, params, tables = args
    me = (MY_BOT, params)
    pts = wins = 0
    for seats, my_seat, seed in tables:
        specs = _seeded(seats, seed)
        specs.insert(my_seat, me)
        r = sim.play_match(specs, seed, roundtrip=False)
        pts += r["points"][my_seat] / (len(specs) - 1)
        wins += r["placements"][my_seat] == 1
    return label, pts, wins, len(tables)


def make_bench(league):
    """Fixed tables (same every generation) so progress is measured against a stationary target."""
    rng = random.Random(12345)
    v0 = (MY_BOT, dict(DEFAULTS))
    bench = {"4p vs baselines": [], "4p vs v0": []}
    for _ in range(600):
        bench["4p vs baselines"].append(([rng.choice(league) for _ in range(3)], rng.randrange(4),
                                         rng.randrange(2**31)))
        bench["4p vs v0"].append(([v0] * 3, rng.randrange(4), rng.randrange(2**31)))
    for spec in league + [v0]:
        name = "v0" if spec is v0 else os.path.basename(spec[0])
        bench["1v1 " + name] = [([spec], rng.randrange(2), rng.randrange(2**31)) for _ in range(200)]
    # one style filling the table (drawn from a separate stream so the tables above stay identical)
    rng2 = random.Random(54321)
    for spec in league:
        bench["4p vs 3x " + os.path.basename(spec[0])] = [
            ([(spec[0], {"seed": rng2.randrange(10**9)}) for _ in range(3)], rng2.randrange(4), rng2.randrange(2**31))
            for _ in range(300)]
    return bench


def run_bench(pool, bench, params, chunk=50):
    jobs = [(label, params, tables[i:i + chunk]) for label, tables in bench.items()
            for i in range(0, len(tables), chunk)]
    acc = {}
    for label, pts, wins, n in pool.map(_bench_chunk, jobs):
        a = acc.setdefault(label, [0.0, 0, 0])
        a[0] += pts
        a[1] += wins
        a[2] += n
    return {label: {"pts": round(a[0] / a[2], 4), "win": round(a[1] / a[2], 4)} for label, a in acc.items()}


def _backfill_new_keys(history, start):
    """Weights added to the bot after a run started: record them as constants (sigma 0) in earlier
    generations, at their starting value, so the history stays rectangular. Their sigma of 0 is
    replaced by the initial sigma when the run resumes."""
    old = history["keys"]
    if old == KEYS:
        return
    fill = {k: float(start[k]) for k in KEYS if k not in old}
    for g in [history["start"]] + history["gens"]:
        for part in ([g] if "mean" in g else [g["sampled_from"]]):
            part["mean"].update(fill)
            part["sigma"].update({k: 0.0 for k in fill})
        if "new_mean" in g:
            g["new_mean"].update(fill)
            g["new_sigma"].update({k: 0.0 for k in fill})
            g["population"] = [[dict(zip(old, vec)).get(k, fill.get(k)) for k in KEYS] for vec in g["population"]]
    history["keys"] = KEYS
    history["bounds"] = {k: BOUNDS.get(k, (-10.0, 10.0, FREE_SIGMA))[:2] for k in KEYS}


def make_tables(rng, games, pool_specs, homog=0.0):
    """Random tables; with probability `homog` every opponent seat is the same bot (one style fills the table)."""
    tables = []
    for _ in range(games):
        k = rng.choice(TABLE_SIZES)
        if rng.random() < homog:
            seats = [rng.choice(pool_specs)] * (k - 1)
        else:
            seats = [rng.choice(pool_specs) for _ in range(k - 1)]
        tables.append((seats, rng.randrange(k), rng.randrange(2**31)))
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--games", type=int, default=600, help="tables per candidate per generation")
    ap.add_argument("--snapshots", type=int, default=6, help="max self-snapshots kept in the league")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", help="params.json to start from (default: bot.py DEFAULTS)")
    ap.add_argument("--out", default=os.path.join(MY_BOT, "params.json"))
    ap.add_argument("--log", default=os.path.join(REPO, "train", "cem_log.csv"))
    ap.add_argument("--history", default=os.path.join(REPO, "train", "cem_history.json"),
                    help="full per-generation record (population, elites, mean/sigma, benchmark)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run stored in --history: its last mean, sigma and league snapshots")
    ap.add_argument("--sigma-boost", type=float, default=1.0, help="scale sigma when resuming (re-explore)")
    ap.add_argument("--workers", type=int, default=None, help="process pool size (default: all cores)")
    ap.add_argument("--homog", type=float, default=0.0, help="share of tables where one opponent type fills every seat")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start = dict(DEFAULTS)
    if args.init:
        with open(args.init) as f:
            start.update(json.load(f))
    mean = [float(start[k]) for k in KEYS]
    sigma = [BOUNDS.get(k, (0, 0, FREE_SIGMA))[2] for k in KEYS]
    league = [(p, None) for p in LEAGUE]
    snapshots = [(MY_BOT, to_params(mean))]
    history = {"keys": KEYS, "bounds": {k: BOUNDS.get(k, (-10.0, 10.0, FREE_SIGMA))[:2] for k in KEYS},
               "config": vars(args), "gens": []}
    first_gen = 0
    if args.resume:
        with open(args.history) as f:
            history = json.load(f)
        _backfill_new_keys(history, start)
        prev = history["gens"][-1]
        mean = [float(prev["new_mean"][k]) for k in KEYS]
        sigma = [args.sigma_boost * prev["new_sigma"][k] or BOUNDS.get(k, (0, 0, FREE_SIGMA))[2] for k in KEYS]
        snapshots = [(MY_BOT, g["new_mean"]) for g in history["gens"] if g["snapshot_added"]][-args.snapshots:]
        first_gen = prev["gen"] + 1
        history.setdefault("runs", []).append({"from_gen": first_gen, "config": vars(args)})
        print(f"resuming at generation {first_gen} with {len(snapshots)} snapshots", flush=True)
    bench = make_bench(league)

    no_throttle()
    with open(args.log, "a" if args.resume else "w", newline="") as logf, \
            Pool(args.workers, initializer=no_throttle) as pool:
        log = csv.writer(logf)
        if not args.resume:
            log.writerow(["gen", "best", "elite_mean", "pop_mean", "mean_sigma", "seconds"])
        if not args.resume:
            history["start"] = {"mean": to_params(mean), "sigma": dict(zip(KEYS, sigma)),
                                "bench": run_bench(pool, bench, to_params(mean))}
            print("bench at start:", history["start"]["bench"], flush=True)
        for gen in range(first_gen, first_gen + args.gens):
            t0 = time.time()
            sampled_from = {"mean": to_params(mean), "sigma": {k: round(s, 4) for k, s in zip(KEYS, sigma)}}
            # half the opponent seats are baselines, half are frozen copies of ourselves
            pool_specs = league * len(snapshots) + snapshots * len(league)
            tables = make_tables(rng, args.games, pool_specs, args.homog)
            pop = [mean[:]]  # keep the current mean in the population (elitism)
            for _ in range(args.pop - 1):
                pop.append([m + s * rng.gauss(0, 1) for m, s in zip(mean, sigma)])
            pop = [[clip(k, v) for k, v in zip(KEYS, vec)] for vec in pop]
            fits = pool.map(_score, [(to_params(v), tables) for v in pop])
            order = sorted(range(len(pop)), key=lambda i: -fits[i])
            elite = [pop[i] for i in order[:args.elite]]
            for j in range(len(KEYS)):
                vals = [e[j] for e in elite]
                m = sum(vals) / len(vals)
                var = sum((v - m) ** 2 for v in vals) / len(vals)
                mean[j] = m
                floor = 0.05 * BOUNDS.get(KEYS[j], (0, 0, FREE_SIGMA))[2]
                sigma[j] = max(floor, 0.7 * sigma[j] + 0.3 * var ** 0.5)
            elite_fit = sum(fits[i] for i in order[:args.elite]) / args.elite
            pop_fit = sum(fits) / len(fits)
            msig = sum(sigma) / len(sigma)
            secs = time.time() - t0
            log.writerow([gen, round(fits[order[0]], 4), round(elite_fit, 4), round(pop_fit, 4), round(msig, 4),
                          round(secs, 1)])
            logf.flush()
            print(f"gen {gen:3d}  best {fits[order[0]]:+.3f}  elite {elite_fit:+.3f}  pop {pop_fit:+.3f}"
                  f"  sigma {msig:.3f}  ({secs:.1f}s)", flush=True)
            snapshot = gen % 3 == 2
            if snapshot:
                snapshots.append((MY_BOT, to_params(mean)))
                snapshots = snapshots[-args.snapshots:]
            with open(args.out, "w") as f:
                json.dump(to_params(mean), f, indent=1, sort_keys=True)
            history["gens"].append({
                "gen": gen,
                "sampled_from": sampled_from,
                "population": [[round(clip(k, v), 3) for k, v in zip(KEYS, vec)] for vec in pop],
                "fitness": [round(f, 4) for f in fits],
                "elite": order[:args.elite],
                "new_mean": to_params(mean),
                "new_sigma": {k: round(s, 4) for k, s in zip(KEYS, sigma)},
                "elite_fit": round(elite_fit, 4), "pop_fit": round(pop_fit, 4), "best": round(fits[order[0]], 4),
                "league_snapshots": len(snapshots), "snapshot_added": snapshot,
                "bench": run_bench(pool, bench, to_params(mean)),
                "seconds": round(secs, 1),
            })
            with open(args.history, "w") as f:
                json.dump(history, f, separators=(",", ":"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
