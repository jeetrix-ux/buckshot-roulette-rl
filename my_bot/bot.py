"""Buckshot Roulette bot: exact shell inference + saw combos, with weights tuned by CEM self-play.

Key fact exploited (see game_API._setup_round): the gun is filled by a fair coin flip while both
shell types remain, then the leftovers fill the tail. So the next shell is 50/50 whenever both
types are left, and certain once either type is exhausted - not lives/(lives+blanks).
"""
import time
from math import comb

from schemas import *

UPGRADES = ("juice", "coffee", "mag_glass", "pool_floatie", "hand_saw")
TYPE_OF = {
    "juice": UpgradeType.JUICE,
    "coffee": UpgradeType.COFFEE,
    "mag_glass": UpgradeType.MAG_GLASS,
    "pool_floatie": UpgradeType.POOL_FLOATIE,
    "hand_saw": UpgradeType.HAND_SAW,
}

DEFAULTS = {
    # shooting
    "p_self": 0.30,        # shoot self when P(live) is below this
    "p_saw": 0.99,         # use saws on an unknown shell only when P(live) >= this
    "saw_nonlethal": 4,    # max saws spent on a known live that cannot kill
    "coffee_rounds": 2,    # use coffee (no mag glass) when this few shells remain
    "juice_below": 5,      # drink juice when health is below this
    # target scoring (higher = shoot first)
    "t_health": -0.6,
    "t_kill": 4.0,
    "t_saw": 0.5,
    "t_mag": 0.3,
    "t_next": 0.8,
    "t_floated": -0.5,
    "t_upg": 0.05,
    # pool floatie
    "f_use": 0.0,          # use a floatie when the best target scores above this
    "f_next": 1.5,
    "f_threat": 0.4,
    "f_health": 0.1,
    # upgrade picks: base value, value per already-held copy, juice per missing health
    "u_juice": 1.0, "u_coffee": 0.6, "u_mag_glass": 1.6, "u_pool_floatie": 1.2, "u_hand_saw": 2.0,
    "h_juice": -0.15, "h_coffee": -0.2, "h_mag_glass": -0.25, "h_pool_floatie": -0.3, "h_hand_saw": -0.1,
    "u_hurt": 0.5,
    "u_deny": 0.3,         # bonus for taking what opponents are short of
    # health-aware defence: "danger" = we have the lowest health at the table, so focus-fire bots aim at us
    "t_danger_health": 0.0,  # extra target score per point of target health while in danger
    "t_divert": 0.0,         # bonus for a hit that drops the target below us (they become the focus)
    "f_danger": 0.0,         # extra floatie score while in danger
    "u_danger_juice": 0.0,   # extra juice pick value while in danger
    "u_danger_floatie": 0.0, # extra floatie pick value while in danger
    # saw combos: hoard glass + saws for one-shot kills
    "mag_min_saws": 0,       # use the mag glass on an unknown shell only when holding this many saws...
    "mag_keep": 0,           # ...or more than this many glasses
    "p_saw_lethal": 1.01,    # stack saws on an uncertain shell when P(live) >= this and the saws make it a kill
    "u_combo_saw": 0.0,      # extra saw pick value while holding a mag glass
    "u_combo_mag": 0.0,      # extra mag-glass pick value while holding a saw
    "u_saw_reach": 0.0,      # extra saw pick value while our saws can't yet one-shot the weakest rival
    # threat model: who is likely to shoot us next
    "t_aims_us": 0.0,        # target bonus when we are this rival's lowest-health target (focus fire)
    "t_soon": 0.0,           # target bonus scaled by how soon this rival acts (1 = next, 1/k = last)
    "t_lethal_us": 0.0,      # target bonus when this rival's saws + a live shell would kill us
    "f_aims_us": 0.0,        # floatie score for the same three threats
    "f_lethal_us": 0.0,
}

EXHAUST_W = 2.5  # extra posterior weight on unknown splits that exhaust one shell type
LEAN_AFTER = 100  # decisions per match before optional upgrade actions are cut back
TIME_GUARD = 0.15  # seconds of own compute per match before switching to the trivial policy


def next_live_prob(obs):
    """P(next shell is live) from public counts, my private ejections and the mag-glass view."""
    if obs.next_round == NextRound.LIVE:
        return 1.0
    if obs.next_round == NextRound.BLANK:
        return 0.0
    ej = obs.ejections
    lives = obs.lives_init - obs.lives_shot - ej.lives_ejected
    blanks = obs.blanks_init - obs.blanks_shot - ej.blanks_ejected
    unknown = ej.unknowns_ejected
    if unknown == 0:
        return _p_from(lives, blanks)
    # Others' coffee removed `unknown` shells of unknown type: weight each split by its count.
    num = den = 0.0
    for j in range(unknown + 1):
        rl, rb = lives - j, blanks - (unknown - j)
        if rl < 0 or rb < 0 or rl + rb == 0:
            continue
        w = comb(unknown, j)
        if rl == 0 or rb == 0:  # histories past exhaustion had forced (probability-1) steps
            w *= EXHAUST_W
        num += w * _p_from(rl, rb)
        den += w
    return num / den if den else 0.5


def _p_from(lives, blanks):
    if lives <= 0:
        return 0.0
    if blanks <= 0:
        return 1.0
    return 0.5


class Bot:
    def init(self, params=None):
        self.w = dict(DEFAULTS)
        if params:
            self.w.update({k: v for k, v in params.items() if k in DEFAULTS})
        self.spent = 0.0
        self.decisions = 0
        self.explain = None  # set to a list to record why each decision was made (replays only)
        self.rule = ""
        self.scores = {}

    # ------------------------------------------------------------------ play
    def play(self, obs):
        started = time.perf_counter()
        self.decisions += 1
        self.rule, self.scores = "", {}
        try:
            if self.spent > TIME_GUARD:
                action = self._trivial(obs)
            else:
                action = self._play(obs)
            if not self._legal(obs, action):
                action = self._trivial(obs)
        except Exception:
            action = self._trivial(obs)
        self.spent += time.perf_counter() - started
        if self.explain is not None:
            self.explain.append({"rule": self.rule or "fallback", "p_live": next_live_prob(obs),
                                 "target_scores": dict(self.scores), "saws_used": self._saws_used(obs)})
        return action

    def _play(self, obs):
        w = self.w
        me = obs.turn
        players = obs.players
        mine = players[me].upgrades
        hp = players[me].health
        others = [i for i, p in enumerate(players) if i != me and p.health > 0]
        # Every request costs ~2 ms of container IPC against a 0.5 s match budget, so in very long
        # matches drop the optional upgrade actions that are worth least.
        lean = self.decisions > LEAN_AFTER

        if mine.juice > 0 and hp < obs.max_health and hp < (3 if lean else w["juice_below"]):
            self.rule = "juice"
            return _use(UpgradeType.JUICE)

        if mine.pool_floatie > 0 and not lean:
            target, score = self._floatie_target(obs, others)
            if target is not None and score > w["f_use"]:
                self.rule = "floatie"
                return _use(UpgradeType.POOL_FLOATIE, target)

        p = next_live_prob(obs)
        if (0.0 < p < 1.0 and mine.mag_glass > 0 and (not lean or mine.hand_saw > 0)
                and (mine.hand_saw >= w["mag_min_saws"] or mine.mag_glass > w["mag_keep"])):
            self.rule = "mag"
            return _use(UpgradeType.MAG_GLASS)
        if 0.0 < p < 1.0 and mine.coffee > 0 and obs.rounds_remaining <= w["coffee_rounds"] and not lean:
            self.rule = "coffee"
            return _use(UpgradeType.COFFEE)

        if p <= 0.0:
            self.rule = "blank_self"
            return _shoot(me)
        used = self._saws_used(obs)
        damage = 1 + used
        sawing = p >= 1.0 or p >= w["p_saw"]
        # All-in on an uncertain shell: only worth it when the stacked saws turn the hit into a kill.
        killable = [i for i in others if players[i].health <= damage + mine.hand_saw]
        if not sawing and killable and mine.hand_saw + used > 0 and p >= w["p_saw_lethal"]:
            target = self._best_target(obs, killable, damage + mine.hand_saw)
            if players[target].health > damage:
                self.rule = "saw_gamble"
                return _use(UpgradeType.HAND_SAW)
            self.rule = "gamble_shoot"
            return _shoot(target)
        target = self._best_target(obs, others, damage + mine.hand_saw if sawing else damage)
        if sawing:
            need = players[target].health - damage  # extra saws that still add damage
            if mine.hand_saw > 0 and need > 0 and (mine.hand_saw >= need or used < w["saw_nonlethal"]):
                self.rule = "saw"
                return _use(UpgradeType.HAND_SAW)
            self.rule = "live_shoot"
            return _shoot(target)
        if p < w["p_self"]:
            self.rule = "risky_self"
            return _shoot(me)
        self.rule = "shoot"
        return _shoot(target)

    def _saws_used(self, obs):
        # Damage resets after every shot; saws used this turn are remembered between calls.
        key = (obs.session_no, obs.rounds_remaining, obs.turn)
        if getattr(self, "_saw_key", None) != key:
            self._saw_key, self._saw_count = key, 0
        return self._saw_count

    def _best_target(self, obs, others, damage):
        w = self.w
        me = obs.turn
        n = len(obs.players)
        nxt = self._next_alive(obs, me)
        hp = obs.players[me].health
        danger = _in_danger(obs, me)
        threat = _threats(obs, me)
        best, best_score = others[0], None
        for i in others:
            pl = obs.players[i]
            u = pl.upgrades
            aims, soon, lethal = threat[i]
            score = (w["t_health"] * pl.health
                     + w["t_aims_us"] * aims + w["t_soon"] * soon + w["t_lethal_us"] * lethal
                     + danger * (w["t_danger_health"] * pl.health + w["t_divert"] * (0 < pl.health - damage < hp))
                     + w["t_kill"] * (pl.health <= damage)
                     + w["t_saw"] * u.hand_saw
                     + w["t_mag"] * u.mag_glass
                     + w["t_next"] * (i == nxt)
                     + w["t_floated"] * pl.has_pool_floatie_on
                     + w["t_upg"] * sum(getattr(u, k) for k in UPGRADES)
                     - 1e-3 * ((i - me) % n))
            if self.explain is not None:
                self.scores[i] = round(score, 3)
            if best_score is None or score > best_score:
                best, best_score = i, score
        return best

    def _floatie_target(self, obs, others):
        w = self.w
        nxt = self._next_alive(obs, obs.turn)
        danger = _in_danger(obs, obs.turn)
        threat = _threats(obs, obs.turn)
        best, best_score = None, None
        for i in others:
            pl = obs.players[i]
            if pl.has_pool_floatie_on:
                continue
            u = pl.upgrades
            aims, _, lethal = threat[i]
            score = (w["f_danger"] * danger + w["f_aims_us"] * aims + w["f_lethal_us"] * lethal
                     + w["f_next"] * (i == nxt)
                     + w["f_threat"] * (u.hand_saw + u.mag_glass)
                     + w["f_health"] * pl.health)
            if best_score is None or score > best_score:
                best, best_score = i, score
        return best, best_score

    @staticmethod
    def _next_alive(obs, me):
        n = len(obs.players)
        for off in range(1, n):
            i = (me + off) % n
            if obs.players[i].health > 0:
                return i
        return me

    # --------------------------------------------------------------- pick
    def pick_upgrade(self, obs):
        started = time.perf_counter()
        self.decisions += 1
        try:
            action = self._pick(obs)
            if getattr(obs.available_upgrades, _name_of(action.chosen_upgrade)) <= 0:
                raise ValueError
        except Exception:
            action = _pick_first(obs)
        self.spent += time.perf_counter() - started
        if self.explain is not None:
            self.explain.append({"rule": "pick", "pick_scores": dict(self.scores)})
            self.scores = {}
        return action

    def _pick(self, obs):
        w = self.w
        me = obs.turn
        mine = obs.players[me].upgrades
        avail = obs.available_upgrades
        deficit = obs.max_health - obs.players[me].health
        rivals = [p for i, p in enumerate(obs.players) if i != me and p.health > 0]
        danger = _in_danger(obs, me)
        best, best_score = None, None
        for name in UPGRADES:
            if getattr(avail, name) <= 0:
                continue
            score = w["u_" + name] + w["h_" + name] * getattr(mine, name)
            if name == "juice":
                score += w["u_hurt"] * deficit + w["u_danger_juice"] * danger
            elif name == "pool_floatie":
                score += w["u_danger_floatie"] * danger
            elif name == "hand_saw":
                score += w["u_combo_saw"] * (mine.mag_glass > 0)
                if rivals:
                    score += w["u_saw_reach"] * (mine.hand_saw + 1 < min(p.health for p in rivals))
            elif name == "mag_glass":
                score += w["u_combo_mag"] * (mine.hand_saw > 0)
            if rivals:
                rival_min = min(getattr(p.upgrades, name) for p in rivals)
                score += w["u_deny"] * (rival_min == 0) * (name in ("hand_saw", "mag_glass"))
            if self.explain is not None:
                self.scores[name] = round(score, 3)
            if best_score is None or score > best_score:
                best, best_score = name, score
        action = Action_PICK_UPGRADES()
        action.chosen_upgrade = TYPE_OF[best]
        return action

    # ------------------------------------------------------------- safety
    def _trivial(self, obs):
        me = obs.turn
        p = next_live_prob(obs)
        if p <= 0.0:
            return _shoot(me)
        others = [i for i, pl in enumerate(obs.players) if i != me and pl.health > 0]
        return _shoot(min(others, key=lambda i: obs.players[i].health) if others else me)

    def _legal(self, obs, action):
        me = obs.turn
        if action.action == ActionType.SHOOT:
            t = action.target
            return isinstance(t, int) and 0 <= t < len(obs.players) and obs.players[t].health > 0
        up = action.useUpgrade
        name = _name_of(up)
        if getattr(obs.players[me].upgrades, name) <= 0:
            return False
        if up == UpgradeType.POOL_FLOATIE:
            t = action.target
            ok = (isinstance(t, int) and 0 <= t < len(obs.players) and t != me
                  and obs.players[t].health > 0 and not obs.players[t].has_pool_floatie_on)
            return ok
        if up in (UpgradeType.COFFEE, UpgradeType.MAG_GLASS):
            return obs.rounds_remaining > 0 and not (up == UpgradeType.MAG_GLASS and obs.next_round != NextRound.UNKNOWN)
        if up == UpgradeType.HAND_SAW:
            self._saws_used(obs)
            self._saw_count += 1
        return True


def _in_danger(obs, me):
    """1 when we have the (joint) lowest health of everyone alive, so focus-fire bots aim at us."""
    hp = obs.players[me].health
    rivals = [p.health for i, p in enumerate(obs.players) if i != me and p.health > 0]
    return int(bool(rivals) and hp <= min(rivals))


def _threats(obs, me):
    """For each living rival: (we are its lowest-health target, how soon it acts, its saws + a live kill us)."""
    players = obs.players
    n = len(players)
    hp = players[me].health
    order = [(me + off) % n for off in range(1, n) if players[(me + off) % n].health > 0]
    out = {}
    for rank, i in enumerate(order):
        rest = [p.health for j, p in enumerate(players) if j not in (i, me) and p.health > 0]
        out[i] = (int(not rest or hp <= min(rest)),
                  (len(order) - rank) / len(order),
                  int(players[i].upgrades.hand_saw + 1 >= hp))
    return out


def _name_of(up):
    return {v: k for k, v in TYPE_OF.items()}[up]


def _use(up, target=None):
    action = Action_PLAY()
    action.action = ActionType.USE_UPGRADE
    action.useUpgrade = up
    action.target = target
    return action


def _shoot(target):
    action = Action_PLAY()
    action.action = ActionType.SHOOT
    action.useUpgrade = None
    action.target = target
    return action


def _pick_first(obs):
    action = Action_PICK_UPGRADES()
    for name in UPGRADES:
        if getattr(obs.available_upgrades, name) > 0:
            action.chosen_upgrade = TYPE_OF[name]
            return action
    action.chosen_upgrade = UpgradeType.JUICE
    return action
