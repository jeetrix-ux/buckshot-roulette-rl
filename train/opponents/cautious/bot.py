"""Baseline that hoards information and health: glass/coffee first, juice early, shoots self on doubt."""
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
        if obs.players[me].health < obs.max_health and mine.juice > 0:
            return _use(UpgradeType.JUICE)
        if obs.next_round == NextRound.UNKNOWN and mine.mag_glass > 0:
            return _use(UpgradeType.MAG_GLASS)
        if obs.next_round == NextRound.UNKNOWN and mine.coffee > 0:
            return _use(UpgradeType.COFFEE)
        target = max(others, key=lambda i: sum(vars(obs.players[i].upgrades).values()))
        if obs.next_round == NextRound.LIVE:
            if mine.hand_saw > 0 and obs.players[target].health > 1:
                return _use(UpgradeType.HAND_SAW)
            return _shoot(target)
        if obs.next_round == NextRound.BLANK:
            return _shoot(me)
        ej = obs.ejections
        lives = obs.lives_init - obs.lives_shot - ej.lives_ejected
        blanks = obs.blanks_init - obs.blanks_shot - ej.blanks_ejected
        return _shoot(me if blanks > lives else target)

    def pick_upgrade(self, obs):
        a = Action_PICK_UPGRADES()
        for up, name in ((UpgradeType.MAG_GLASS, "mag_glass"), (UpgradeType.JUICE, "juice"),
                         (UpgradeType.COFFEE, "coffee"), (UpgradeType.HAND_SAW, "hand_saw"),
                         (UpgradeType.POOL_FLOATIE, "pool_floatie")):
            if getattr(obs.available_upgrades, name) > 0:
                a.chosen_upgrade = up
                return a
