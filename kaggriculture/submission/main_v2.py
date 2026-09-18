"""Kaggriculture v2 agent for kuchinwei.

Strategy (vs baseline early-harvest wheat loop):
1. Crop choice: WHEAT for days 0..SWITCH_DAY-1 (fast cash), then MELON (high $/tile).
2. Harvest at peak unfertilized yield (not first_yield_day).
3. Watering priority: dying plants -> ready harvests -> routine water -> weeds -> plant.
4. Pathing: nearest target with claim-set so farmer + hands do not collide.
5. Hire up to MAX_HANDS cheap farm hands each day; BUY_LAND when crowded + funded.
6. Avoid waste: cash reserve before seed/land spend; no plantings that cannot mature;
   skip $1-floor melon dumps unless shed is nearly full.

NOTE: kaggle-environments loads the *last* top-level callable as the agent.
"""

CROPS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "peak": 4},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "peak": 3},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "peak": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "peak": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "peak": 6},
}

SWITCH_DAY = 6          # plant wheat before this day, melons after
CASH_RESERVE = 600      # keep liquid for hires / emergencies
MAX_HANDS = 8
LAND_PRICES = (1000, 2000, 4000)
LAST_DAY = 29           # season days 0..29
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
    # Peak unfertilized yield, or past max_yield_day (decay follows).
    return age >= cd["max_yield_day"] or units >= cd["peak"]


def _scan(me, day):
    dying, ready, need_water, weeds, empties = [], [], [], [], []
    n_plants = 0
    for y, row in enumerate(me["tiles"]):
        for x, cell in enumerate(row):
            if cell is None:
                empties.append((x, y))
            elif isinstance(cell, dict) and cell.get("kind") == "WEED":
                weeds.append((x, y))
            elif isinstance(cell, dict) and cell.get("kind") == "PLANT":
                n_plants += 1
                if _is_ready(cell, day):
                    ready.append((x, y))
                elif not cell.get("watered_today", False):
                    if cell.get("consecutive_unwatered", 0) >= 1:
                        dying.append((x, y))
                    else:
                        need_water.append((x, y))
    return dying, ready, need_water, weeds, empties, n_plants


def _plant_crop(day):
    return "WHEAT" if day < SWITCH_DAY else "MELON"


def _can_plant(day, crop):
    cd = CROPS[crop]
    # Must be able to reach first_yield before season ends.
    return day + cd["first_yield_day"] <= LAST_DAY


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
    crop = _plant_crop(day)
    seeds = dict(private.get("seeds", {}) or {})
    shed = private.get("shed", {}) or {}
    money = float(me.get("money", 0))
    prices = (obs.get("market") or {}).get("prices") or {}
    fx, fy = me["farmer"]
    tiles = me["tiles"]
    hands = me.get("hands", []) or []

    market = []

    # Sell shed goods. Hold melons at $1 floor unless shed is tight.
    shed_used = sum(shed.values())
    for item, n in list(shed.items()):
        if n <= 0 or item in ("GOOSE", "COW", "SHEEP"):
            continue
        price = prices.get(item, 999)
        if item == "MELON" and price < SELL_FLOOR and shed_used < 80:
            continue
        market.append(["SELL", item, int(n)])

    dying, ready, need_water, weeds, empties, n_plants = _scan(me, day)

    # Hire hands for watering / planting throughput (fib costs are tiny early).
    want_hands = min(MAX_HANDS, max(0, (n_plants + len(empties)) // 3))
    for _ in range(max(0, want_hands - len(hands))):
        if len(market) >= 10:
            break
        market.append(["HIRE"])

    # Unlock next quadrant when current land is full and we keep a reserve.
    n_unlocked = len(me.get("unlocked_quadrants", ["NW"]))
    if (
        n_unlocked < 4
        and len(empties) <= 2
        and len(market) < 10
        and money >= LAND_PRICES[n_unlocked - 1] + CASH_RESERVE
    ):
        market.append(["BUY_LAND"])

    can_plant = _can_plant(day, crop)
    want_seeds = min(10, len(empties) + 2) if can_plant else 0
    to_buy = max(0, want_seeds - seeds.get(crop, 0))
    seed_cost = CROPS[crop]["seed"]
    while to_buy > 0 and len(market) < 10 and money - seed_cost >= CASH_RESERVE:
        n = min(to_buy, 6)
        while n > 0 and money - seed_cost * n < CASH_RESERVE:
            n -= 1
        if n <= 0:
            break
        market.append(["BUY_SEED", crop, int(n)])
        money -= seed_cost * n
        seeds[crop] = seeds.get(crop, 0) + int(n)
        to_buy -= n

    plant_empties = empties if can_plant else []
    buckets = (dying, ready, need_water, weeds, plant_empties)
    claimed = set()

    farmer = _unit_action((fx, fy), tiles[fy][fx], day, seeds, crop, buckets, claimed, can_plant)
    hands_actions = [
        _unit_action((hx, hy), tiles[hy][hx], day, seeds, crop, buckets, claimed, can_plant)
        for hx, hy in hands
    ]

    return {"farmer": farmer, "hands": hands_actions, "market": market[:10]}
