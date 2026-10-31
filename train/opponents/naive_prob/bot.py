"""Baseline that believes P(live) = lives_left / shells_left (the common, wrong model)."""
from schemas import *


def _use(up, target=None):
    a = Action_PLAY(); a.action = ActionType.USE_UPGRADE; a.useUpgrade = up; a.target = target
    return a


def _shoot(t):
    a = Action_PLAY(); a.action = ActionType.SHOOT; a.useUpgrade = None; a.target = t
    return a


class Bot:
    def init(self, params=None):
        pass

    def play(self, obs):
        me = obs.turn
        mine = obs.players[me].upgrades
        others = [i for i, p in enumerate(obs.players) if i != me and p.health > 0]
        ej = obs.ejections
        lives = obs.lives_init - obs.lives_shot - ej.lives_ejected
        total = obs.rounds_remaining + ej.unknowns_ejected
        p = lives / total if total else 0.5
        if obs.next_round == NextRound.LIVE:
            p = 1.0
        elif obs.next_round == NextRound.BLANK:
            p = 0.0
        if obs.players[me].health < obs.max_health and mine.juice > 0:
            return _use(UpgradeType.JUICE)
        if 0 < p < 1 and mine.mag_glass > 0:
            return _use(UpgradeType.MAG_GLASS)
        target = min(others, key=lambda i: obs.players[i].health)
        if p >= 0.5:
            if mine.hand_saw > 0 and p >= 0.75 and obs.players[target].health > 1:
                return _use(UpgradeType.HAND_SAW)
            return _shoot(target)
        return _shoot(me)

    def pick_upgrade(self, obs):
        a = Action_PICK_UPGRADES()
        for up, name in ((UpgradeType.HAND_SAW, "hand_saw"), (UpgradeType.MAG_GLASS, "mag_glass"),
                         (UpgradeType.JUICE, "juice"), (UpgradeType.POOL_FLOATIE, "pool_floatie"),
                         (UpgradeType.COFFEE, "coffee")):
            if getattr(obs.available_upgrades, name) > 0:
                a.chosen_upgrade = up
                return a
