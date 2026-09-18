# Kaggriculture improve v3 (kuchinwei)

Date: 2026-09-17 (Asia/Taipei)  
前版：v2 melon-strategy，公開分 **600.0**  
Agent files: `submission/main_v3.py` → copied to `submission/main.py`  
v2 backup: `submission/main_v2.py`

## Strategy changes vs v2

| Area | v2 | v3 |
|------|----|----|
| Switch day | WHEAT days 0–5, then MELON | WHEAT days 0–3 (`SWITCH_DAY=4`), then MELON/TOMATO |
| Crops | Melon only after switch | Melon primary + **TOMATO** strip (`TOMATO_CAP=12`, ~1 tomato per 3 melons) |
| Market order | Sell → hire → land → seeds | Sell → **seeds** → land → hire (hire 不再塞滿 10 slots) |
| Fertilizer / animals | none | Tried; fert logistics / geese **hurt** local money → dropped |
| Rest | Peak harvest, hire≤8, BUY_LAND, claim-set | Same as v2 |

Rejected ideas (local worse or fragile): BUY_PRODUCT fertilizer + PICKUP/FERTILIZE pathing; goose CARE/FEED stack; SWITCH_DAY≥7 (beats random but loses hard H2H vs v2); strawberry mix.

## Local smoke metrics (`make('kaggriculture')`, debug=True)

All runs status **DONE** / **DONE** (no ERROR). Seeds: 1,2,3,7,11. Rewards = end-of-season **money** (not public skill rating).

### 96 / 240 smoke (v3 vs random)

| episodeSteps | seed | v3 | random |
|--------------|------|----|--------|
| 96 | 1 | 1284 | 2120 |
| 240 | 1 | 616 | 1330 |

(Short episodes still cash-poor while capital is in crops — same pattern as v2.)

### 720 v3 vs `random`

| seed | v3 | random |
|------|----|--------|
| 1 | 31380 | 0 |
| 2 | 29995 | 0 |
| 3 | 29378 | 0 |
| 7 | 30249 | 0 |
| 11 | 29720 | 0 |
| **avg** | **30144** | 0 |

### 720 v2 vs `random` (same seeds, same session)

| seed | v2 |
|------|----|
| 1 | 25800 |
| 2 | 25800 |
| 3 | 25825 |
| 7 | 28641 |
| 11 | 28653 |
| **avg** | **26944** |

### 720 v3 vs `starter`

| seed | v3 | starter |
|------|----|---------|
| 1 | 30113 | 3496 |
| 2 | 31105 | 3528 |
| 3 | 29763 | 3639 |
| 7 | 30336 | 3712 |
| 11 | 52060 | 3648 |
| **avg** | **34675** | ~3605 |

### 720 v3 vs v2 head-to-head

| seed | v3 | v2 |
|------|----|----|
| 1 | 20932 | 2227 |
| 2 | 20006 | 7595 |
| 3 | 20989 | 1602 |
| 7 | 31792 | 1429 |
| 11 | 27051 | 4686 |
| **avg** | **24154** | **3508** |
| wins | **5 / 5** | |

## Interpretation

- Full-season local money vs random: v3 **~30144** vs v2 **~26944** (**+~12%**).
- Clear wins vs starter and vs v2 itself (tomato diversification + earlier switch helps under melon glut).
- Public LB is skill-rating style (v2 was **600.0**); local money is for strategy comparison only — do not invent public scores.

## Submission

**Ready to submit** when Kaggle auth is available:

```bash
cd /workspace/im-admissions-116/prep/competitions/kaggriculture
kaggle competitions submit kaggriculture -f submission/main.py -m "v3 wheat→melon+tomato strip SWITCH_DAY=4 local>v2"
```

Or Kaggle web **Submit Agent** with description mentioning **v3**.

As of 2026-09-17 ~22:50 Asia/Taipei: this executor box has **no** `~/.kaggle/access_token` / `KAGGLE_API_TOKEN`, and no browser Task tool — **not yet submitted**. Public score: **pending**.
