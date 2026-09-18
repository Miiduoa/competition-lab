# Kaggriculture improve v2 (kuchinwei)

Date: 2026-09-15  
前版 baseline 已由 v2 取代；v2 最新公開分：**600.0** — melon strategy  
Agent files: `submission/main_v2.py` (canonical) → copied to `submission/main.py`

## Strategy changes vs baseline

| Area | Baseline v1 | v2 |
|------|-------------|-----|
| Crop | WHEAT only, harvest at age≥2 | WHEAT days 0–5, then **MELON**; harvest at **peak** unfertilized yield |
| Watering | Only current tile / nearest empty | Priority: **dying → ready → water → dig → plant** |
| Pathing | Nearest empty only | Nearest priority target + **claim-set** (farmer+hands) |
| Labor | Solo farmer | Hire up to **8** hands/day (fib costs) |
| Land | Never | `BUY_LAND` when full + cash reserve |
| Waste | Early harvest, late plant OK | Cash reserve $600; no plant if cannot mature; hold melon at $1 floor |

## Local smoke metrics (`make('kaggriculture')`, debug=True)

All runs status **DONE** / **DONE** (no ERROR).

### v2 vs `random`

| episodeSteps | seeds | P0 rewards (v2) | P0 avg | P1 (random) notes |
|--------------|-------|-----------------|--------|-------------------|
| 96 | 1,2,3,7,11 | 1284 ×5 | **1284** | random ~2280–2820 (v2 capital still in crops) |
| 240 | 1,2,3,7,11 | 648,621,666,603,603 | **628** | random ~1510–2240 (pre-/early melon cash-out) |
| **720** | 1,2,3,7,11 | 25800,25800,25642,28659,25794 | **26339** | random **0** |

### v2 vs built-in `starter` (720)

| seed | v2 | starter |
|------|-----|---------|
| 1 | 25800 | 3590 |
| 2 | 28653 | 3526 |
| 3 | 25642 | 3639 |
| 7 | 28659 | 3657 |
| 11 | 28653 | 3460 |

### v2 vs recreated baseline wheat-loop (720)

| seed | v2 | baseline |
|------|-----|----------|
| 1 | 28627 | 3074 |
| 2 | 28627 | 3057 |
| 3 | 25625 | 2990 |
| 7 | 28633 | 3057 |
| 11 | 25777 | 3015 |

Baseline reference (from `BASELINE_NOTES.md`): 96-step ~2747; 240-step ~2845; 720 vs random ~3150.

## Interpretation

- Full-season (**720**) local money is **~8×** baseline wheat-loop and dominates `starter` / `random`.
- Short smokes (96/240) look weaker in **cash** because melons lock capital until ~day 10; Kaggle episodes use full season.
- v2 Public LB score **600.0** is skill-rating style, not raw money；本機測試用於說明策略差異，不另宣稱排名或其他分數。

## Submission result

**v2 已提交並完成評分** — `submission/main.py`（與 `main_v2.py` 相同）；最新公開分 **600.0**。

Suggested message: `v2 wheat-warmup then melon peak-harvest + hire/land`

本次紀錄以 Kaggle v2 已提交結果為準；不另宣稱排名或其他分數。

```bash
kaggle competitions submit kaggriculture -f submission/main.py -m "v2 wheat-warmup then melon peak-harvest + hire/land"
```
