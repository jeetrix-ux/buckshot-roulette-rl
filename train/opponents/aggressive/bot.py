"""Baseline that always shoots the weakest opponent and dumps saws whenever it has them."""
import random

from schemas import *


def _use(up, target=None):
    a = Action_PLAY(); a.action = ActionType.USE_UPGRADE; a.useUpgrade = up; a.target = target
    return a


def _shoot(t):
    a = Action_PLAY(); a.action = ActionType.SHOOT; a.useUpgrade = None; a.target = t
    return a


class Bot:
    def init(self, params=None):
        self.rng = random.Random((params or {}).get("seed"))

    def play(self, obs):
        me = obs.turn
        mine = obs.players[me].upgrades
        others = [i for i, p in enumerate(obs.players) if i != me and p.health > 0]
        if obs.next_round == NextRound.BLANK:
            return _shoot(me)
        target = min(others, key=lambda i: (obs.players[i].health, self.rng.random()))
        if mine.pool_floatie > 0:
            free = [i for i in others if not obs.players[i].has_pool_floatie_on]
            if free:
                return _use(UpgradeType.POOL_FLOATIE, self.rng.choice(free))
        if mine.hand_saw > 0 and obs.players[target].health > 1:
            return _use(UpgradeType.HAND_SAW)
        if mine.juice > 0 and obs.players[me].health <= 2:
            return _use(UpgradeType.JUICE)
        return _shoot(target)

    def pick_upgrade(self, obs):
        a = Action_PICK_UPGRADES()
        for up, name in ((UpgradeType.HAND_SAW, "hand_saw"), (UpgradeType.POOL_FLOATIE, "pool_floatie"),
                         (UpgradeType.JUICE, "juice"), (UpgradeType.COFFEE, "coffee"),
                         (UpgradeType.MAG_GLASS, "mag_glass")):
            if getattr(obs.available_upgrades, name) > 0:
                a.chosen_upgrade = up
                return a
