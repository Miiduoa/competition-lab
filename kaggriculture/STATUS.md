# Kaggriculture 狀態

- joined=yes（已接受競賽規則並成功加入）
- baseline_ready=yes
- 驗證證據：Competition 頁面顯示 `Submit Agent`；Team 頁面列出 `kuchinwei (You)` 為 `Team Leader`（solo team）。
- 截圖：`kaggriculture/joined.png`
- Agent：`submission/main.py`（wheat plant/water/harvest/sell + walk-to-empty）
- 詳見：`BASELINE_NOTES.md`

## 重要時程（UTC）

- 開始：2026-07-29
- Entry deadline：2026-09-23 23:59（須在此日前接受規則）
- Team merge deadline：2026-09-23 23:59
- Final submission deadline：2026-09-30 23:59
- 預計持續跑分／收斂：2026-10-01 至約 2026-10-15

## Baseline / 本機測試

1. 已安裝：`pip install -U kaggle-environments`（含 `kaggriculture` env）。
2. 已建立 `submission/main.py`，agent function 為檔案最後一個 top-level callable。
3. 本機測試結果：
   - `make('kaggriculture', configuration={'episodeSteps': 96})` → `main.py` vs `random`：**PASS**（P0≈2747 DONE / P1≈2260 DONE）
   - 延長 240 steps：**PASS**（P0≈2845 DONE / P1≈1850 DONE）
4. **目前最新提交**：Kaggle Submit Agent 接受 `submission/main.py`，描述 `v2 melon-strategy`；目前狀態為 **Complete**，v2 公開分數 **600.0**（不記錄舊版名次）。
5. 監控：`kaggle competitions submissions kaggriculture`；再用 episodes/logs 檢查行為。

## 目前注意事項

- joined=yes；baseline_ready=yes；**v2 agent 已提交，Complete**；公開分數 **600.0**。
- identity verification 已不再阻擋 Submit Agent；v2 截圖：`kaggriculture/submit_v2.png`。

## Verification check (historical, 2026-09-15 07:30 UTC)

- Phone verification: **yes** — Settings → Account shows `Verified`.
- Identity verification: **no** — exact UI text: `You have not verified your identity using Persona, a trusted 3rd-party service. Verifying your identity allows you to join competitions that require identity verification.`
- Can Submit Agent now: **no** — Kaggriculture shows `This competition requires identity verification` and `To submit to this competition, you'll need to verify your identity.` The action opens `To continue, please verify your identity` with `Continue to Persona`.
- Evidence screenshot: `kaggriculture/id_verify_status.png`

## Initial submission check (2026-09-15 07:37 UTC)

- Submit Agent flow opened without identity-verification blocker; page showed 5 submissions remaining.
- Uploaded `submission/main.py` and submitted the initial v1 agent with description `baseline v1 wheat-loop`.
- At the initial check, Kaggle Submissions showed `main.py — Pending`; this was later completed as recorded below.
- Evidence screenshot: `kaggriculture/submit_v1.png`


## Historical v1 submission (已由 v2 取代)

- 初版 `baseline v1 wheat-loop` 曾完成提交；其公開分與名次為舊版紀錄，現在以 v2 公開分 **600.0** 為準。
- Evidence screenshot: `kaggriculture/score_check.png`


## Improve v2 (2026-09-15)

- Agent: `submission/main_v2.py` copied to `submission/main.py` (wheat→melon peak harvest, hire, land, priority pathing).
- Local 720 vs random avg money **~26339** (baseline was ~3150); vs starter/baseline wheat: clear wins, all DONE.
- Details: `IMPROVE_v2.md`. v2 已提交並完成評分，最新公開分 **600.0**。


## v2 submission (2026-09-15 08:29 UTC)

- Submitted `submission/main.py` (v2 melon-strategy; description `v2 melon-strategy local>>baseline`) for `kuchinwei`.
- Kaggle Submissions shows `main.py Complete`; v2 public score **600.0**。
- Evidence screenshot: `kaggriculture/submit_v2.png`


## v2 result (2026-09-15 08:31 UTC)

- Kaggle status: **Complete**.
- Public score: **600.0** (submission details).
- Screenshot updated: `kaggriculture/submit_v2.png`


## Improve v3 (2026-09-17 Asia/Taipei ~22:50)

- Agent: `submission/main_v3.py` copied to `submission/main.py` (v2 kept as `main_v2.py`).
- Changes: `SWITCH_DAY=4`; tomato strip (cap 12, ratio 3); market order sell→seeds→land→hire.
- Local 720 mean money vs random: **v3 ≈ 30144** vs **v2 ≈ 26944**; vs starter ≈ 34675; H2H vs v2 **5/5** (avg ~24154 vs ~3508). Details: `IMPROVE_v3.md`.
- **Submit status**: files ready; **not submitted yet** — Kaggle CLI unauthenticated on this box (no access_token / KAGGLE_API_TOKEN). Need browser Submit Agent or CLI token. Public score: **pending** (do not invent).
- Still current public LB reference until v3 scores: v2 **600.0**.

## v3 submission (2026-09-17 22:57 Asia/Taipei)

- 已以 `kuchinwei` 提交 canonical `submission/main.py`（v3 wheat→melon+tomato, `SWITCH_DAY=4`）。
- 描述：`v3 wheat→melon+tomato SWITCH_DAY=4 local>v2`
- Kaggle 狀態：**Complete**。
- 公開分數：**600.0**。
- Submit time：2026-09-17 22:57:17 Asia/Taipei（UTC+8）。
- 截圖：`kaggriculture/submit_v3.png`
