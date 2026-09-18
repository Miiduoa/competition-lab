"""Kaggriculture v3 agent for kuchinwei.

vs v2:
1. Earlier melon switch: SWITCH_DAY 6 → 4 (faster cash into high-$ crops).
2. Diversify ~1/4 tiles into TOMATO (ongoing, hinge scarcity curve) to reduce
   melon-glut risk and beat competing melon agents head-to-head.
3. Same peak-harvest / hire / land / claim-set pathing as v2.

Local 720 (mean over seeds): clearly above v2 vs random/starter, and wins
v2 head-to-head. Public metric is skill-rating style — not raw cash.

NOTE: kaggle-environments loads the *last* top-level callable as the agent.
"""

CROPS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "peak": 4},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "peak": 3},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "peak": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "peak": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "peak": 6},
}

SWITCH_DAY = 4
TOMATO_CAP = 12
TOMATO_RATIO = 3  # plant tomato while tomato_tiles * RATIO <= melon_tiles
CASH_RESERVE = 600
MAX_HANDS = 8
LAND_PRICES = (1000, 2000, 4000)
LAST_DAY = 29
SELL_FLOOR = 2


def _step_toward(fx, fy, tx, ty):
    if fx < tx:
        return "EAST"
    if fx > tx:
        return "WEST"
    if fy < ty:
        return "SOUTH"
    if fy > ty:
        return "NORTH"
    return "PASS"


def _nearest(pos, targets, claimed):
    best = None
    best_d = None
    fx, fy = pos
    for t in targets:
        if t in claimed:
            continue
        d = abs(t[0] - fx) + abs(t[1] - fy)
        if best_d is None or d < best_d:
            best_d = d
            best = t
    return best


def _is_ready(tile, day):
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return False
    cd = CROPS.get(tile.get("crop"))
    if not cd:
        return False
    age = day - tile["planted_day"]
    units = tile.get("yield_units", 0)
    if units <= 0:
        return False
    if cd["ongoing"]:
        return age >= cd["first_yield_day"]
    if age < cd["first_yield_day"]:
        return False
    return age >= cd["max_yield_day"] or units >= cd["peak"]


def _scan(me, day):
    dying, ready, need_water, weeds, empties = [], [], [], [], []
    n_plants = tomato_tiles = melon_tiles = 0
    for y, row in enumerate(me["tiles"]):
        for x, cell in enumerate(row):
            if cell is None:
                empties.append((x, y))
            elif isinstance(cell, dict) and cell.get("kind") == "WEED":
                weeds.append((x, y))
            elif isinstance(cell, dict) and cell.get("kind") == "PLANT":
                n_plants += 1
                crop = cell.get("crop")
                if crop == "TOMATO":
                    tomato_tiles += 1
                elif crop == "MELON":
                    melon_tiles += 1
                if _is_ready(cell, day):
                    ready.append((x, y))
                elif not cell.get("watered_today", False):
                    if cell.get("consecutive_unwatered", 0) >= 1:
                        dying.append((x, y))
                    else:
                        need_water.append((x, y))
    return dying, ready, need_water, weeds, empties, n_plants, tomato_tiles, melon_tiles


def _plant_crop(day, tomato_tiles, melon_tiles):
    if day < SWITCH_DAY:
        return "WHEAT"
    if day + CROPS["MELON"]["first_yield_day"] > LAST_DAY:
        if day + CROPS["TOMATO"]["first_yield_day"] <= LAST_DAY:
            return "TOMATO"
        return "WHEAT"
    if (
        day <= 22
        and tomato_tiles < TOMATO_CAP
        and tomato_tiles * TOMATO_RATIO <= max(1, melon_tiles)
    ):
        return "TOMATO"
    return "MELON"


def _can_plant(day, crop):
    return day + CROPS[crop]["first_yield_day"] <= LAST_DAY


def _local_action(tile, day, seeds, crop, can_plant):
    if _is_ready(tile, day):
        return ["HARVEST"]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if not tile.get("watered_today", False):
            return ["WATER"]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None and can_plant and seeds.get(crop, 0) > 0:
        return ["PLANT", crop]
    return None


def _unit_action(pos, tile, day, seeds, crop, buckets, claimed, can_plant):
    local = _local_action(tile, day, seeds, crop, can_plant)
    if local is not None:
        claimed.add(tuple(pos))
        if local[0] == "PLANT":
            seeds[crop] = seeds.get(crop, 0) - 1
        return local

    for bucket in buckets:
        target = _nearest(pos, bucket, claimed)
        if target is None:
            continue
        claimed.add(target)
        if target == tuple(pos):
            return ["PASS"]
        return [_step_toward(pos[0], pos[1], target[0], target[1])]
    return ["PASS"]


def agent(obs, config=None):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs.get("day", 0)
    seeds = dict(private.get("seeds", {}) or {})
    shed = private.get("shed", {}) or {}
    money = float(me.get("money", 0))
    prices = (obs.get("market") or {}).get("prices") or {}
    fx, fy = me["farmer"]
    tiles = me["tiles"]
    hands = me.get("hands", []) or []

    dying, ready, need_water, weeds, empties, n_plants, tomato_tiles, melon_tiles = _scan(me, day)
    crop = _plant_crop(day, tomato_tiles, melon_tiles)

    market = []

    shed_used = sum(shed.values())
    for item, n in list(shed.items()):
        if n <= 0 or item in ("GOOSE", "COW", "SHEEP"):
            continue
        price = prices.get(item, 999)
        if item == "MELON" and price < SELL_FLOOR and shed_used < 80:
            continue
        market.append(["SELL", item, int(n)])

    can_plant = _can_plant(day, crop)
    want_seeds = min(10, len(empties) + 2) if can_plant else 0
    to_buy = max(0, want_seeds - seeds.get(crop, 0))
    seed_cost = CROPS[crop]["seed"]
    while to_buy > 0 and len(market) < 8 and money - seed_cost >= CASH_RESERVE:
        n = min(to_buy, 6)
        while n > 0 and money - seed_cost * n < CASH_RESERVE:
            n -= 1
        if n <= 0:
            break
        market.append(["BUY_SEED", crop, int(n)])
        money -= seed_cost * n
        seeds[crop] = seeds.get(crop, 0) + int(n)
        to_buy -= n

    n_unlocked = len(me.get("unlocked_quadrants", ["NW"]))
    if (
        n_unlocked < 4
        and len(empties) <= 2
        and len(market) < 9
        and money >= LAND_PRICES[n_unlocked - 1] + CASH_RESERVE
    ):
        market.append(["BUY_LAND"])

    want_hands = min(MAX_HANDS, max(0, (n_plants + len(empties)) // 3))
    for _ in range(max(0, want_hands - len(hands))):
        if len(market) >= 10:
            break
        market.append(["HIRE"])

    plant_empties = empties if can_plant else []
    buckets = (dying, ready, need_water, weeds, plant_empties)
    claimed = set()

    farmer = _unit_action((fx, fy), tiles[fy][fx], day, seeds, crop, buckets, claimed, can_plant)
    hands_actions = [
        _unit_action((hx, hy), tiles[hy][hx], day, seeds, crop, buckets, claimed, can_plant)
        for hx, hy in hands
    ]

    return {"farmer": farmer, "hands": hands_actions, "market": market[:10]}
