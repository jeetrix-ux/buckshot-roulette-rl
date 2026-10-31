"""Record annotated games and win statistics for the training visualiser.

    python train/replay.py [--params my_bot/params.json] [--pool 1500] [--rollouts 64] [--out viz/replays.json]

1. Plays `pool` seeded games of the bot (frozen weights) against the baseline league and summarises
   each one: how the bot won or lost, who killed whom and how, and what it did along the way.
2. Picks a few showcase games and replays them exactly, recording every decision: the public table,
   the hidden shells (viewer only), the action and its result, and for our seat the rule that fired,
   P(live) under our model and the naive one, and the target / pick scores.
3. At every step of a showcase game it estimates our win chance with Monte-Carlo rollouts. Each rollout
   re-deals the unseen part of the gun from the real generator (so only public information is used),
   reseeds future sessions and random bots, and plays the game out with the same bots.
"""
import argparse
import copy
import json
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402
from eval import LEAGUE, REPO  # noqa: E402

import game_API as api  # noqa: E402
import wire  # noqa: E402
from engine.scoring import placement_points  # noqa: E402
from schemas import NextRound, Stage  # noqa: E402

MY_BOT = os.path.join(REPO, "my_bot")
UPS = ("juice", "coffee", "mag_glass", "pool_floatie", "hand_saw")


def naive_p(obs):
    ej = obs.ejections
    lives = obs.lives_init - obs.lives_shot - ej.lives_ejected
    total = obs.rounds_remaining + ej.unknowns_ejected
    return lives / total if total else 0.5


def _act(state, bots):
    play = state.stage == Stage.PLAY
    obs = api.play_observation(state) if play else api.upgrade_observation(state)
    action = bots[state.turn].play(obs) if play else bots[state.turn].pick_upgrade(obs)
    (api.step_PLAY if play else api.step_PICK_UPGRADE)(state, action)


def _redeal(state, rng):
    """Re-deal the unseen shells of the current session with the engine's coin-flip generator."""
    if state.stage != Stage.PLAY or state.rounds_remaining <= 0:
        return
    idx = len(state.round_contents) - state.rounds_remaining
    keep = 1 if state.player_next_round_view != NextRound.UNKNOWN else 0  # a revealed shell stays put
    rest = state.round_contents[idx + keep:]
    lives, blanks = sum(rest), len(rest) - sum(rest)
    new = []
    for _ in rest:
        if lives and blanks:
            live = rng.randrange(2) == 1
        else:
            live = lives > 0
        new.append(live)
        lives -= live
        blanks -= not live
    state.round_contents = state.round_contents[:idx + keep] + new


def win_chance(state, bots, ours, steps_done, n, seed):
    """Monte-Carlo estimate of (P(win), expected placement points / (k-1)) from this exact moment."""
    rng = random.Random(seed)
    k = len(state.players)
    wins = pts = 0.0
    for _ in range(n):
        s = copy.deepcopy(state)
        bs = copy.deepcopy(bots)
        for b in bs:
            if hasattr(b, "explain"):
                b.explain = None
            if hasattr(b, "rng"):
                b.rng = random.Random(rng.random())
        s.rng = random.Random(rng.random())
        _redeal(s, rng)
        steps = steps_done
        try:
            while not s.game_over and steps < 300:
                steps += 1
                _act(s, bs)
        except Exception:
            pass
        place = sim._placements(s)[ours]
        wins += place == 1
        pts += placement_points(place, k) / (k - 1)
    return round(wins / n, 3), round(pts / n, 3)


def record_game(seed, specs, names, ours, rollouts=0):
    k = len(specs)
    bots = [sim.make_bot(s) for s in specs]
    bots[ours].explain = []
    state = api.initGame(seed, k)
    steps = []
    n = 0
    while not state.game_over and n < 300:
        seat, stage = state.turn, state.stage
        play = stage == Stage.PLAY
        rec = {"seat": seat, "stage": stage.value, "table": {
            "session": state.session_no,
            "health": [p.health for p in state.players],
            "upgrades": [[getattr(p.upgrades, u) for u in UPS] for p in state.players],
            "floatie": [p.has_pool_floatie_on for p in state.players],
        }}
        if rollouts:
            rec["wp"] = win_chance(state, bots, ours, n, rollouts, seed * 1000 + n)
        n += 1
        obs = api.play_observation(state) if play else api.upgrade_observation(state)
        if play:
            idx = len(state.round_contents) - state.rounds_remaining
            rec["gun"] = {"lives_init": state.lives_init, "blanks_init": state.blanks_init,
                          "remaining": state.rounds_remaining, "lives_shot": state.lives_shot,
                          "blanks_shot": state.blanks_shot, "sequence": [int(x) for x in state.round_contents],
                          "next_index": idx, "next_live": int(state.round_contents[idx]),
                          "damage": state.damage}
            if seat == ours:
                rec["p_naive"] = round(naive_p(obs), 3)
        else:
            rec["available"] = [getattr(state.available_upgrades, u) for u in UPS]
        action = bots[seat].play(obs) if play else bots[seat].pick_upgrade(obs)
        rec["action"] = wire.action_play_to_wire(action) if play else wire.action_pick_to_wire(action)
        if seat == ours:
            why = bots[ours].explain[-1]
            rec["why"] = {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in why.items()}
        before = [p.health for p in state.players]
        lives_before = state.lives_shot
        (api.step_PLAY if play else api.step_PICK_UPGRADE)(state, action)
        if play and rec["action"]["action"] == "SHOOT":
            rec["result"] = {"live": int(state.lives_shot > lives_before),
                             "damage": [b - p.health for b, p in zip(before, state.players)]}
        steps.append(rec)
    end = {"health": [p.health for p in state.players]}
    if rollouts:
        place = sim._placements(state)[ours]
        end["wp"] = (float(place == 1), placement_points(place, k) / (k - 1))
    return {"seed": seed, "names": names, "ours": ours, "placements": sim._placements(state), "steps": steps,
            "capped": not state.game_over, "end": end}


def summarise(game):
    """Per-game facts about our seat: how it ended, kills, deaths and what we did."""
    ours, names, steps = game["ours"], game["names"], game["steps"]
    k = len(names)
    rules = Counter()
    picks = Counter()
    f = defaultdict(float)
    kills, opp_deaths, our_death = [], [], None
    for st in steps:
        if st["seat"] == ours and "why" in st:
            rules[st["why"]["rule"]] += 1
            if st["stage"] == "PICK_UPGRADE":
                picks[st["action"]["chosen_upgrade"]] += 1
        r = st.get("result")
        if not r:
            continue
        shooter, target = st["seat"], st["action"]["target"]
        dmg = r["damage"][target]
        if shooter == ours and target != ours:
            f["dmg_dealt"] += dmg
            if 0 < st["why"]["p_live"] < 1:
                f["coinflip_shots"] += 1
                f["coinflip_hits"] += r["live"]
                if st["p_naive"] < 0.5:  # a lives/shells bot would have shot itself here
                    f["naive_self_shots"] += 1
                    f["naive_self_hits"] += r["live"]
            else:
                f["certain_shots"] += 1
        if target == ours:
            f["dmg_taken"] += dmg
        if dmg and st["table"]["health"][target] - dmg <= 0:  # this shot eliminated `target`
            event = {"victim": target, "shooter": shooter, "damage": dmg, "saws": st["gun"]["damage"] - 1}
            if shooter == ours:
                p = st["why"]["p_live"]
                event["method"] = "certain" if p >= 1 else "coinflip"
                kills.append(event)
            if target == ours:
                our_death = {"by": "self" if shooter == ours else names[shooter], "damage": dmg,
                             "saws": event["saws"], "opponents_left": sum(h > 0 for i, h in enumerate(st["table"]["health"]) if i != ours)}
            else:
                event["by_us"] = shooter == ours
                opp_deaths.append(event)
    place = game["placements"][ours]
    won = place == 1
    if won:
        if game["capped"]:
            how = "cap"
        elif opp_deaths and opp_deaths[-1]["by_us"]:
            how = "final_kill"
        else:
            how = "outlasted"
    else:
        how = None
    return {
        "k": k, "won": won, "place": place, "how": how, "capped": game["capped"],
        "opponents": [n for i, n in enumerate(names) if i != ours],
        "rules": dict(rules), "picks": dict(picks), "f": dict(f),
        "kills": kills, "opp_deaths": opp_deaths, "our_death": our_death,
        "final_health": game["end"]["health"][ours],
        "length": len(steps),
    }


def _scan(job):
    seed, specs, names, ours = job
    game = record_game(seed, specs, names, ours)
    s = summarise(game)
    s["job"] = job
    return s


def _record(job_and_label):
    job, label, rollouts = job_and_label
    game = record_game(*job, rollouts=rollouts)
    game["label"] = label
    return game


def aggregate(summaries):
    four = [s for s in summaries if s["k"] == 4]
    won = [s for s in four if s["won"]]
    lost = [s for s in four if not s["won"]]
    out = {"games_4p": len(four), "places": [sum(s["place"] == p for s in four) for p in (1, 2, 3, 4)]}
    out["win_how"] = dict(Counter(s["how"] for s in won))
    kills = [e for s in four for e in s["kills"]]
    out["kills"] = {
        "total": len(kills),
        "certain": sum(e["method"] == "certain" for e in kills),
        "coinflip": sum(e["method"] == "coinflip" for e in kills),
        "with_saws": sum(e["saws"] > 0 for e in kills),
        "saw_hist": dict(Counter(e["saws"] for e in kills)),
        "per_game_won": round(sum(len(s["kills"]) for s in won) / max(1, len(won)), 2),
        "per_game_lost": round(sum(len(s["kills"]) for s in lost) / max(1, len(lost)), 2),
    }
    opp = [e for s in four for e in s["opp_deaths"]]
    out["opp_eliminations"] = {"total": len(opp), "by_us": sum(e["by_us"] for e in opp)}
    deaths = [s["our_death"] for s in four if s["our_death"]]
    out["our_deaths"] = {
        "total": len(deaths),
        "by": dict(Counter(d["by"] for d in deaths)),
        "sawed": sum(d["saws"] > 0 for d in deaths),
        "opponents_left": dict(Counter(d["opponents_left"] for d in deaths)),
    }

    def mean(rows, fn):
        return round(sum(fn(s) for s in rows) / max(1, len(rows)), 2)

    metrics = [
        ("Damage dealt", lambda s: s["f"].get("dmg_dealt", 0)),
        ("Damage taken", lambda s: s["f"].get("dmg_taken", 0)),
        ("Opponents we eliminated", lambda s: len(s["kills"])),
        ("Juice drunk", lambda s: s["rules"].get("juice", 0)),
        ("Free turns from certain blanks", lambda s: s["rules"].get("blank_self", 0)),
        ("Magnifying glasses used", lambda s: s["rules"].get("mag", 0)),
        ("Shots on a certain live", lambda s: s["f"].get("certain_shots", 0)),
        ("Coin-flip shots at opponents", lambda s: s["f"].get("coinflip_shots", 0)),
        ("Hand saws used", lambda s: s["rules"].get("saw", 0) + s["rules"].get("saw_gamble", 0)),
        ("Pool floaties thrown", lambda s: s["rules"].get("floatie", 0)),
        ("Juice picked", lambda s: s["picks"].get("JUICE", 0)),
        ("Hand saws picked", lambda s: s["picks"].get("HAND_SAW", 0)),
    ]
    out["won_vs_lost"] = [{"metric": name, "won": mean(won, fn), "lost": mean(lost, fn)} for name, fn in metrics]
    per_opp = defaultdict(lambda: [0, 0])
    for s in four:
        for n in set(s["opponents"]):
            per_opp[n][0] += s["won"]
            per_opp[n][1] += 1
    out["win_rate_with"] = {n: round(w / t, 3) for n, (w, t) in per_opp.items()}
    rules = Counter()
    for s in summaries:
        rules.update(s["rules"])
    out["rule_counts"] = dict(rules)
    shots = [s["f"] for s in summaries]
    out["coinflip"] = {"shots": int(sum(f.get("coinflip_shots", 0) for f in shots)),
                       "hits": int(sum(f.get("coinflip_hits", 0) for f in shots))}
    out["naive_would_self"] = {"n": int(sum(f.get("naive_self_shots", 0) for f in shots)),
                               "live": int(sum(f.get("naive_self_hits", 0) for f in shots))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(MY_BOT, "params.json"), help="frozen weights for our bot")
    ap.add_argument("--out", default=os.path.join(REPO, "viz", "replays.json"))
    ap.add_argument("--pool", type=int, default=1500, help="games to scan for statistics and showcase picks")
    ap.add_argument("--rollouts", type=int, default=64, help="rollouts per step for the win-chance line")
    args = ap.parse_args()
    with open(args.params) as f:
        params = json.load(f)
    rng = random.Random(7)
    jobs = []
    for g in range(args.pool):
        k = (4, 4, 4, 3, 2)[g % 5]
        ours = rng.randrange(k)
        specs = [(p, {"seed": rng.randrange(10**9)}) for p in (rng.choice(LEAGUE) for _ in range(k - 1))]
        specs.insert(ours, (MY_BOT, params))
        names = [("our bot" if s[0] == MY_BOT else os.path.basename(s[0])) for s in specs]
        jobs.append((rng.randrange(2**31), specs, names, ours))
    from cem import no_throttle
    no_throttle()
    with Pool(initializer=no_throttle) as pool:
        summaries = pool.map(_scan, jobs, chunksize=8)
        stats = aggregate(summaries)
        wants = [
            ("4-player win with a saw kill", lambda s: s["k"] == 4 and s["won"] and any(e["saws"] for e in s["kills"])
             and s["length"] < 170),
            ("4-player win by outlasting", lambda s: s["k"] == 4 and s["how"] == "outlasted" and s["length"] < 170),
            ("4-player loss", lambda s: s["k"] == 4 and s["place"] >= 3 and s["length"] > 60),
            ("1v1 duel", lambda s: s["k"] == 2 and s["won"] and s["rules"].get("floatie", 0) >= 1),
            ("3-player comeback", lambda s: s["k"] == 3 and s["won"] and s["f"].get("dmg_taken", 0) >= 4),
        ]
        chosen = []
        for label, cond in wants:
            hit = next((s for s in summaries if cond(s) and s["job"] not in [c[0] for c in chosen]), None)
            if hit:
                chosen.append((hit["job"], label, args.rollouts))
        games = pool.map(_record, chosen, chunksize=1)
    out = {"games": games, "stats": stats, "pool_games": len(summaries), "params": params,
           "rule_counts": stats["rule_counts"]}
    out["shots"] = {"naive_would_self": stats["naive_would_self"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(games)} showcase games {[g['label'] for g in games]}")
    print(json.dumps({k: v for k, v in stats.items() if k != "rule_counts"}, indent=1))



if __name__ == "__main__":
    main()
