"""Fast in-process match runner for training and evaluation.

Mirrors engine/run_match.py (wire round-trip of observations and actions, fallback moves after a
fault, two-strike elimination, 300-action cap, placements) without subprocesses or containers.
"""
import importlib.util
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import game_API as api  # noqa: E402
import wire  # noqa: E402
from engine.run_match import _fallback_pick, _fallback_play, _placements  # noqa: E402
from engine.scoring import placement_points  # noqa: E402
from schemas import Stage  # noqa: E402

_classes = {}


def load_bot_class(path):
    """Bot class from <path>/bot.py, each directory loaded once under its own module name."""
    path = os.path.abspath(path)
    if path not in _classes:
        name = "bot_" + str(len(_classes))
        spec = importlib.util.spec_from_file_location(name, os.path.join(path, "bot.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _classes[path] = module.Bot
    return _classes[path]


def make_bot(spec):
    """spec = (path, params-or-None). params=None means the directory's params.json (if any)."""
    path, params = spec
    if params is None:
        pj = os.path.join(path, "params.json")
        params = {}
        if os.path.exists(pj):
            import json
            with open(pj) as f:
                params = json.load(f)
    bot = load_bot_class(path)()
    if hasattr(bot, "init"):
        bot.init(params)
    return bot


def play_match(specs, seed, max_actions=300, max_strikes=2, roundtrip=True, timing=False):
    """specs[i] plays seat i. Returns dict with placements, points, faults, (optional) per-seat time."""
    k = len(specs)
    bots = [make_bot(s) for s in specs]
    state = api.initGame(seed, k)
    rng = random.Random(seed ^ 0x5EED)
    strikes = [0] * k
    faults = [[] for _ in range(k)]
    spent = [0.0] * k
    decisions = [0] * k
    steps = 0
    while not state.game_over and steps < max_actions:
        steps += 1
        seat, stage = state.turn, state.stage
        play = stage == Stage.PLAY
        obs = api.play_observation(state) if play else api.upgrade_observation(state)
        try:
            if roundtrip:
                data = wire.to_wire(obs)
                obs = wire.obs_play_from_wire(data) if play else wire.obs_pick_from_wire(data)
            t0 = time.perf_counter()
            action = bots[seat].play(obs) if play else bots[seat].pick_upgrade(obs)
            spent[seat] += time.perf_counter() - t0
            decisions[seat] += 1
            if roundtrip:
                if play:
                    action = wire.action_play_from_wire(wire.action_play_to_wire(action))
                else:
                    action = wire.action_pick_from_wire(wire.action_pick_to_wire(action))
            (api.step_PLAY if play else api.step_PICK_UPGRADE)(state, action)
        except Exception as exc:  # any bot failure is a strike, as in the real engine
            faults[seat].append(repr(exc)[:200])
            strikes[seat] += 1
            if strikes[seat] >= max_strikes:
                api.eliminate_player(state, seat)
            elif not state.game_over:
                if play:
                    api.step_PLAY(state, _fallback_play(state, rng))
                else:
                    api.step_PICK_UPGRADE(state, _fallback_pick(state))
    placements = _placements(state)
    out = {
        "placements": placements,
        "points": [placement_points(p, k) for p in placements],
        "faults": faults,
        "actions": steps,
        "capped": not state.game_over,
    }
    if timing:
        out["spent"] = spent
        out["decisions"] = decisions
    return out
