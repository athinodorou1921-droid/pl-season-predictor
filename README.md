# Premier League 2026-27 Season Predictor

Predicting final Premier League points and positions from prior-season performance, using **SQL for feature engineering** and **Python for modelling and walk-forward backtesting**.

## Summary

- Built a SQLite database of 11 Premier League seasons (2015-16 to 2025-26, 4,180 matches) plus Championship data.
- Engineered team-season features in SQL using window functions (`LAG`, `RANK`, rolling `AVG ... OVER`), including detection of promoted teams.
- Evaluated with a **walk-forward backtest** over 7 seasons (2019-20 to 2025-26), training only on past seasons each time.
- Final model: **3.08 mean absolute position error vs 3.35 for a naive baseline** (~8% lower), beating it in 5 of 7 seasons.
- Adding shots-on-target share reduced error vs a goals/points-only model; Championship form for promoted teams was tested and rejected.

## Data

- Match results and shot statistics from [football-data.co.uk](https://www.football-data.co.uk) (Premier League `E0`, Championship `E1`).
- Loaded into SQLite (`pl.db`) and queried with pandas `read_sql`.

## Method

### 1. League tables rebuilt in SQL
Each match is split into home and away team-rows (`UNION ALL`), aggregated per team-season, and ranked with `RANK() OVER (PARTITION BY Season ORDER BY points, goal difference)`. Rebuilt tables were checked against official final standings.

### 2. Features (all from the *previous* season, so no look-ahead)

| Feature | Description |
|---|---|
| `prev_pts` | Points last season |
| `prev_gd` | Goal difference last season |
| `prev_sot_share` | Team's share of all shots on target in its matches last season |
| `avg_pts_3` | Average points over the previous 3 seasons |
| `promoted` | 1 if the team was not in the Premier League the previous season |

### 3. Handling stale data
Two subtle issues came up and were fixed:

- **Returning teams:** `LAG()` returns a team's previous *row*, which for a team returning after relegation can be several years old. Promoted teams are detected by checking the gap in season index, and their lagged stats are neutralised (set to 0) so the `promoted` flag captures a "typical promoted side" effect.
- **Rolling averages:** a row-based 3-season window (`ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING`) pulled in seasons from a decade earlier for teams with gaps in their history. It was replaced with a range-based window on season index (`RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING`), so only genuinely recent seasons are used. The fix improved backtest error from 3.14 to 3.08.

### 4. Model and evaluation
- Linear regression predicting season points; predicted points are ranked to give positions.
- **Walk-forward validation:** for each test season, train on all earlier seasons only, then predict. Seven out-of-sample test seasons in total.
- **Baseline:** teams finish in the same order as last season (promoted teams at the bottom).
- Metric: mean absolute error in league position.

## Results

| Season | Model MAE | Baseline MAE |
|---|---|---|
| 2019-20 | 3.50 | 3.30 |
| 2020-21 | 2.75 | 3.25 |
| 2021-22 | 2.40 | 2.70 |
| 2022-23 | 4.30 | 4.40 |
| 2023-24 | 2.30 | 2.80 |
| 2024-25 | 2.50 | 3.30 |
| 2025-26 | 3.80 | 3.70 |
| **Average** | **3.08** | **3.35** |

**Interpretation:** the model is on average ~0.27 places more accurate per team than the baseline and wins in 5 of 7 seasons. With only 7 test seasons, this is suggestive rather than statistically conclusive: a model with no real edge would win 5+ of 7 by chance about 23% of the time.

### Feature experiments

| Feature set | Walk-forward MAE |
|---|---|
| Points / goal difference / 3-season average | 3.42 |
| + shots-on-target share | 3.14 |
| + Championship points per game and shot share (promoted teams) | 3.16 (rejected) |

- **Shots on target help.** Underlying performance appears to predict future results better than results alone, since goals and points are noisier. The shots model beat the goals-only model in 6 of 7 seasons.
- **Championship form did not help.** It only affects 3 teams per season, the training sample is small (~30 promoted teams), and second-tier dominance translates unevenly to the top flight. It was dropped to avoid overfitting.

*(Feature-experiment figures were computed before the rolling-window fix; the final model's numbers are in the table above.)*

## 2026-27 prediction (pre-season)

| Pos | Team | Pred. pts |
|---|---|---|
| 1 | Arsenal | 77.9 |
| 2 | Man City | 74.2 |
| 3 | Liverpool | 64.4 |
| 4 | Man United | 61.1 |
| 5 | Chelsea | 58.7 |
| ... | ... | ... |
| 17 | Sunderland | 45.7 |
| 18 | Coventry | 34.4 |
| 19 | Ipswich | 34.4 |
| 20 | Hull | 34.4 |

Notable call: **Sunderland** finished 7th in 2025-26 but are predicted 17th, because the model sees their results as having outrun their underlying shot numbers and expects regression. Predictions will be scored against the actual table in May 2027.

## Limitations and next steps

- **Promoted teams are indistinguishable:** all three receive the same prediction. Better promoted-team data (e.g. squad value, transfer spend) could help.
- **No squad information:** transfers, injuries and managerial changes aren't captured.
- **Small evaluation sample:** more seasons, or match-level modelling, would give stronger statistical evidence.
- Possible extensions: expected goals (xG), regularised or non-linear models, and probabilistic outputs (e.g. title and relegation probabilities via simulation).

## How to run

```bash
pip install pandas scikit-learn notebook
jupyter notebook
```

Open `pl_predictor.ipynb` and run all cells. Data is downloaded automatically from football-data.co.uk.
