# Kaggriculture baseline notes

- Competition: https://www.kaggle.com/competitions/kaggriculture
- Team: kuchinwei (solo; already joined)
- Agent path: `submission/main.py`
- Env package: `kaggle-environments` (includes `make("kaggriculture", ...)`)

## What the agent does

Simple **wheat loop** baseline (valid action shape every turn):

1. **Market:** if no WHEAT seeds and money ≥ $10 → `BUY_SEED WHEAT 1`; sell any WHEAT in the shed.
2. **On current tile:** `PLANT WHEAT` on empty; `WATER` if plant not watered today; `HARVEST` when age ≥ 2 (wheat `first_yield_day`); `DIG` weeds.
3. **Otherwise:** walk toward the nearest empty unlocked tile (`NORTH`/`SOUTH`/`EAST`/`WEST`), else `PASS`.
4. Returns `{"farmer": [...], "hands": [], "market": [...]}`.

Helpers `_step_toward` / `_nearest_empty` are defined **above** `agent` because kaggle-environments loads the **last** top-level callable in the file as the agent.

## Local test result

```text
pip install -U kaggle-environments
cd .../kaggriculture
python -c "
from kaggle_environments import make
env = make('kaggriculture', configuration={'episodeSteps': 96}, debug=True)
env.run(['submission/main.py', 'random'])
print([(s['reward'], s['status']) for s in env.steps[-1]])
"
```

- **96-step smoke:** PASS — Player0 reward≈2747 DONE, Player1 (random)≈2260 DONE
- **240-step extended:** PASS — Player0≈2845 DONE, Player1≈1850 DONE
- No ERROR / exception

Full season default is `episodeSteps=720` (24 turns × 30 days); shorter configs are fine for local smoke.

## How to submit (after identity verification)

Blocked until Kaggle shows identity verification complete (page still requires verify before Submit Agent).

```bash
# single file
kaggle competitions submit kaggriculture -f submission/main.py -m "baseline v1 wheat loop"

# or multi-file archive with main.py at archive root:
# tar -czf submission.tar.gz -C submission main.py
# kaggle competitions submit kaggriculture -f submission.tar.gz -m "baseline v1"

kaggle competitions submissions kaggriculture
```

Then inspect episodes/logs on the competition site for runtime errors.
