```python
# =========================
# NFL SCORE PREDICTOR (FULL / REVISED / CLEAN)
# - Uses the Excel dataset (team offense stats + opponent offense stats per game: opp_*)
# - Builds offense efficiency + TRUE defense (derived from opponent offense vs you)
# - Builds rolling last-5 features for BOTH offense and defense (shifted -> no leakage)
# - Includes home-vs-away effects (is_home + interactions)
# - Trains XGBoost if available (fallback if not)
# - Validates on last 2 completed weeks before WEEK_TO_PREDICT
# - Predicts BOTH teams' points for the games you list
# - Prints FEATURE IMPORTANCE (top 30)
# - Saves predictions CSV to Desktop
# =========================

import os
import numpy as np
import pandas as pd

# =========================
# SETTINGS YOU CHANGE
# =========================
DATA_FILE = r"C:\PUT FILE NAME HERE"
SHEET_NAME = 0

WEEK_TO_PREDICT = 17

# Put games here as: ("AWAY", "HOME")  <-- edit this list each week
WEEK_GAMES = [
    ("DAL", "WAS"),
    ("DET", "MIN"),
    ("DEN", "KC"),
    ("HOU", "LAC"),
    ("BAL", "GB"),
    ("TB", "MIA"),
    ("NE", "NYJ"),
    ("JAX", "IND"),
    ("ARI", "CIN"),
    ("PIT", "CLE"),
    ("NO", "TEN"),
    ("SEA", "CAR"),
    ("NYG", "LV"),
    ("PHI", "BUF"),
    ("CHI", "SF"),
    ("LA", "ATL"),
]

ROLL_WINDOW = 5
MIN_PERIODS = 3
TARGET = "points_scored"

OUT_BASENAME = f"nfl_week_{WEEK_TO_PREDICT}_predictions.csv"

# =========================
# HELPERS
# =========================
def get_desktop_path():
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else os.getcwd()

def normalize_team_code(x):
    if pd.isna(x):
        return x
    x = str(x).strip().upper()
    alias = {
        "JAC": "JAX",
        "WSH": "WAS",
        "WFT": "WAS",
        "LA": "LAR",   # Rams often appear as LA
    }
    return alias.get(x, x)

def safe_div(numer, denom):
    return np.where((denom == 0) | pd.isna(denom), np.nan, numer / denom)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# =========================
# MODEL SELECTION
# =========================
def choose_model():
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=0,
        )
        return model, "XGBoost", True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=600,
            random_state=42,
        )
        return model, "HistGradientBoostingRegressor", False

# =========================
# FEATURE ENGINEERING
# =========================
def add_offense_efficiency(df):
    df = df.copy()

    # basic counts
    df["plays"]  = df["pass_attempts"] + df["rush_attempts"]
    df["drives"] = safe_div(df["first_downs"], 2.5)  # rough proxy

    # offense efficiency
    df["yards_per_play"]       = safe_div(df["total_yards"], df["plays"])
    df["points_per_drive"]     = safe_div(df["points_scored"], df["drives"])
    df["turnovers_per_drive"]  = safe_div(df["turnovers"], df["drives"])
    df["sack_rate"]            = safe_div(df["sacked"], (df["pass_attempts"] + df["sacked"]))

    # home interactions (lets model learn different weights at home)
    df["home_yards_per_play"]   = df["yards_per_play"] * df["is_home"]
    df["home_points_per_drive"] = df["points_per_drive"] * df["is_home"]
    df["home_turnovers_per_drive"] = df["turnovers_per_drive"] * df["is_home"]
    df["home_sack_rate"]        = df["sack_rate"] * df["is_home"]

    return df

def add_true_defense_from_opponent_offense(df):
    """
    TRUE defense based on what opponent offense did AGAINST this team.
    Your dataset already contains opp_* per row, so this is real defensive info.
    """
    df = df.copy()

    # "allowed" totals
    df["points_allowed"] = df["opp_points_scored"]
    df["yards_allowed"]  = df["opp_total_yards"]

    # what the defense faced
    df["def_plays_faced"]  = df["opp_pass_attempts"] + df["opp_rush_attempts"]
    df["def_drives_faced"] = safe_div(df["opp_first_downs"], 2.5)

    # defensive efficiency (allowed rates)
    df["def_yards_per_play_allowed"]     = safe_div(df["yards_allowed"], df["def_plays_faced"])
    df["def_points_per_drive_allowed"]   = safe_div(df["points_allowed"], df["def_drives_faced"])

    # turnovers forced: opponent turnovers vs this team
    df["turnovers_forced"] = df["opp_turnovers"]
    df["def_turnovers_forced_per_drive"] = safe_div(df["turnovers_forced"], df["def_drives_faced"])

    # sack rate forced: opponent sacks taken / (opp pass attempts + opp sacks taken)
    df["def_sack_rate_forced"] = safe_div(df["opp_sacked"], (df["opp_pass_attempts"] + df["opp_sacked"]))

    # home interactions for defense too (home crowd effect, travel, etc.)
    df["home_def_points_per_drive_allowed"] = df["def_points_per_drive_allowed"] * df["is_home"]
    df["home_def_yards_per_play_allowed"]   = df["def_yards_per_play_allowed"] * df["is_home"]

    return df

def add_rolling_means(df, cols, window=5, min_periods=3):
    """
    Rolling mean over last N games PER TEAM, shifted by 1 so it uses only prior games.
    """
    df = df.copy()
    for c in cols:
        out = f"{c}_roll{window}"
        df[out] = (
            df.groupby("team")[c]
              .shift(1)
              .rolling(window=window, min_periods=min_periods)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

required_cols = [
    "team", "nfl_week", "is_home", TARGET,
    # offense
    "first_downs","rush_attempts","rush_yards","rush_tds",
    "pass_completions","pass_attempts","pass_yards","pass_tds",
    "interceptions","sacked","sacked_yards_lost","net_pass_yards","total_yards",
    "fumbles","fumbles_lost","turnovers","penalties","penalty_yards",
    # opponent offense (to build defense)
    "opp_team","opp_points_scored","opp_first_downs","opp_rush_attempts","opp_rush_yards","opp_rush_tds",
    "opp_pass_completions","opp_pass_attempts","opp_pass_yards","opp_pass_tds",
    "opp_interceptions","opp_sacked","opp_sacked_yards_lost","opp_net_pass_yards","opp_total_yards",
    "opp_fumbles","opp_fumbles_lost","opp_turnovers","opp_penalties","opp_penalty_yards",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in your Excel: {missing}")

# normalize teams
df["team"] = df["team"].apply(normalize_team_code)
df["opp_team"] = df["opp_team"].apply(normalize_team_code)

# week numeric
df["nfl_week"] = pd.to_numeric(df["nfl_week"], errors="coerce").astype("Int64")

# optional weather columns
if "wind_mph" not in df.columns:
    df["wind_mph"] = np.nan
if "opp_wind_mph" not in df.columns:
    df["opp_wind_mph"] = np.nan

# sort for rolling
df = df.sort_values(["team", "nfl_week"]).reset_index(drop=True)

# =========================
# BUILD FEATURES
# =========================
df = add_offense_efficiency(df)
df = add_true_defense_from_opponent_offense(df)

# clean inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# rolling offense form (includes scoring form)
OFF_ROLL_COLS = [
    "yards_per_play","points_per_drive","turnovers_per_drive","sack_rate",TARGET,
    "home_yards_per_play","home_points_per_drive","home_turnovers_per_drive","home_sack_rate"
]
df = add_rolling_means(df, OFF_ROLL_COLS, window=ROLL_WINDOW, min_periods=MIN_PERIODS)

# rolling TRUE defense form
DEF_ROLL_COLS = [
    "points_allowed","yards_allowed","def_yards_per_play_allowed","def_points_per_drive_allowed",
    "def_turnovers_forced_per_drive","def_sack_rate_forced",
    "home_def_points_per_drive_allowed","home_def_yards_per_play_allowed"
]
df = add_rolling_means(df, DEF_ROLL_COLS, window=ROLL_WINDOW, min_periods=MIN_PERIODS)

# =========================
# FEATURES LIST (FULL)
# =========================
FEATURES = [
    # context
    "is_home",
    "wind_mph",

    # raw offense stats
    "first_downs","rush_attempts","rush_yards","rush_tds",
    "pass_completions","pass_attempts","pass_yards","pass_tds",
    "interceptions","sacked","sacked_yards_lost","net_pass_yards","total_yards",
    "fumbles","fumbles_lost","turnovers","penalties","penalty_yards",

    # offense efficiency + home interactions
    "plays","drives",
    "yards_per_play","points_per_drive","turnovers_per_drive","sack_rate",
    "home_yards_per_play","home_points_per_drive","home_turnovers_per_drive","home_sack_rate",

    # opponent offense raw (still useful as matchup signal)
    "opp_points_scored","opp_first_downs","opp_rush_attempts","opp_rush_yards","opp_rush_tds",
    "opp_pass_completions","opp_pass_attempts","opp_pass_yards","opp_pass_tds",
    "opp_interceptions","opp_sacked","opp_sacked_yards_lost","opp_net_pass_yards","opp_total_yards",
    "opp_fumbles","opp_fumbles_lost","opp_turnovers","opp_penalties","opp_penalty_yards",
    "opp_wind_mph",

    # TRUE defense derived from opp_*
    "points_allowed","yards_allowed","def_plays_faced","def_drives_faced",
    "def_yards_per_play_allowed","def_points_per_drive_allowed",
    "turnovers_forced","def_turnovers_forced_per_drive","def_sack_rate_forced",
    "home_def_points_per_drive_allowed","home_def_yards_per_play_allowed",
]

# add rolling columns (offense + defense)
for c in OFF_ROLL_COLS:
    FEATURES.append(f"{c}_roll{ROLL_WINDOW}")
for c in DEF_ROLL_COLS:
    FEATURES.append(f"{c}_roll{ROLL_WINDOW}")

# keep only existing columns (avoids KeyErrors if something is absent)
FEATURES = [c for c in FEATURES if c in df.columns]

# =========================
# TRAIN / VALIDATION SPLIT
# validate on last 2 completed weeks before WEEK_TO_PREDICT
# =========================
val_start = max(1, WEEK_TO_PREDICT - 2)
val_end   = WEEK_TO_PREDICT - 1

train_df = df[df["nfl_week"] < val_start].copy()
val_df   = df[df["nfl_week"].between(val_start, val_end)].copy()

# drop rows missing needed predictors or target
train_df = train_df.dropna(subset=FEATURES + [TARGET])
val_df   = val_df.dropna(subset=FEATURES + [TARGET])

print("Rows after dropna:")
print("  Train:", len(train_df))
print("  Val  :", len(val_df))
print(f"Validation weeks: {val_start} to {val_end}")

# =========================
# TRAIN MODEL
# =========================
model, model_name, is_xgb = choose_model()
print("\nModel:", model_name)

X_train = train_df[FEATURES].astype(float)
y_train = train_df[TARGET].astype(float)

X_val = val_df[FEATURES].astype(float)
y_val = val_df[TARGET].astype(float)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print("\nValidation performance:")
print("  RMSE:", round(rmse(y_val, val_preds), 2))
print("  MAE :", round(mae(y_val, val_preds), 2))

# =========================
# FEATURE IMPORTANCE (TOP 30)
# =========================
def print_feature_importance(model, feature_names, X_ref, y_ref, top_n=30, use_xgb=False):
    print("\nFEATURE IMPORTANCE (top {})".format(top_n))

    if use_xgb and hasattr(model, "feature_importances_"):
        imps = np.array(model.feature_importances_, dtype=float)
        order = np.argsort(imps)[::-1][:top_n]
        for i, idx in enumerate(order, 1):
            print(f"{i:>2}. {feature_names[idx]:<40} {imps[idx]:.6f}")
        return

    # fallback: permutation importance (slower but works on any sklearn model)
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            model, X_ref, y_ref,
            n_repeats=10, random_state=42, scoring="neg_mean_absolute_error"
        )
        imps = r.importances_mean
        order = np.argsort(imps)[::-1][:top_n]
        for i, idx in enumerate(order, 1):
            print(f"{i:>2}. {feature_names[idx]:<40} {imps[idx]:.6f}")
    except Exception as e:
        print("Could not compute permutation importance:", e)

print_feature_importance(model, FEATURES, X_val, y_val, top_n=30, use_xgb=is_xgb)

# =========================
# BUILD MATCHUP FEATURES (OFFENSE vs OPPONENT DEFENSE)
# =========================
df_sorted = df.sort_values(["team", "nfl_week"]).reset_index(drop=True)
global_means = df_sorted[FEATURES].mean(numeric_only=True)

# Inject THESE opponent DEFENSE rolling signals into the team’s feature row
OPP_DEF_ROLL = [
    f"def_points_per_drive_allowed_roll{ROLL_WINDOW}",
    f"def_yards_per_play_allowed_roll{ROLL_WINDOW}",
    f"def_turnovers_forced_per_drive_roll{ROLL_WINDOW}",
    f"def_sack_rate_forced_roll{ROLL_WINDOW}",
    f"points_allowed_roll{ROLL_WINDOW}",
    f"yards_allowed_roll{ROLL_WINDOW}",
]

def get_latest_row_before_week(team, week):
    team = normalize_team_code(team)
    sub = df_sorted[(df_sorted["team"] == team) & (df_sorted["nfl_week"] < week)].sort_values("nfl_week")
    if len(sub) == 0:
        return None
    return sub.iloc[-1]

def build_feature_row(team, opponent, is_home, week):
    team = normalize_team_code(team)
    opponent = normalize_team_code(opponent)

    base = get_latest_row_before_week(team, week)
    if base is None:
        row = global_means.copy()
    else:
        row = base[FEATURES].copy()

    # set is_home
    if "is_home" in row.index:
        row["is_home"] = int(is_home)

    # wind: unknown for future -> use team mean so far
    if "wind_mph" in row.index:
        row["wind_mph"] = df_sorted.loc[df_sorted["team"] == team, "wind_mph"].mean()

    # re-compute home interaction columns for the FUTURE row (important!)
    # because we changed is_home.
    # (Only recompute if the columns exist.)
    def recompute_home_interactions(r):
        # needs base efficiency columns available; if NaN, stays NaN
        if "home_yards_per_play" in r.index and "yards_per_play" in r.index and "is_home" in r.index:
            r["home_yards_per_play"] = r["yards_per_play"] * r["is_home"]
        if "home_points_per_drive" in r.index and "points_per_drive" in r.index and "is_home" in r.index:
            r["home_points_per_drive"] = r["points_per_drive"] * r["is_home"]
        if "home_turnovers_per_drive" in r.index and "turnovers_per_drive" in r.index and "is_home" in r.index:
            r["home_turnovers_per_drive"] = r["turnovers_per_drive"] * r["is_home"]
        if "home_sack_rate" in r.index and "sack_rate" in r.index and "is_home" in r.index:
            r["home_sack_rate"] = r["sack_rate"] * r["is_home"]

        if "home_def_points_per_drive_allowed" in r.index and "def_points_per_drive_allowed" in r.index and "is_home" in r.index:
            r["home_def_points_per_drive_allowed"] = r["def_points_per_drive_allowed"] * r["is_home"]
        if "home_def_yards_per_play_allowed" in r.index and "def_yards_per_play_allowed" in r.index and "is_home" in r.index:
            r["home_def_yards_per_play_allowed"] = r["def_yards_per_play_allowed"] * r["is_home"]
        return r

    row = recompute_home_interactions(row)

    # Inject opponent TRUE DEFENSE rolling features (THIS is the key improvement)
    opp_latest = get_latest_row_before_week(opponent, week)
    if opp_latest is not None:
        for f in OPP_DEF_ROLL:
            if (f in row.index) and (f in opp_latest.index):
                row[f] = opp_latest[f]

    # fill remaining missing values with global means
    row = row.astype(float).fillna(global_means)

    return row.to_dict()

# =========================
# PREDICT WEEK GAMES
# =========================
pred_rows = []
for away, home in WEEK_GAMES:
    away = normalize_team_code(away)
    home = normalize_team_code(home)

    away_feat = build_feature_row(away, home, is_home=0, week=WEEK_TO_PREDICT)
    home_feat = build_feature_row(home, away, is_home=1, week=WEEK_TO_PREDICT)

    away_pts = float(model.predict(pd.DataFrame([away_feat])[FEATURES])[0])
    home_pts = float(model.predict(pd.DataFrame([home_feat])[FEATURES])[0])

    away_disp = int(round(away_pts))
    home_disp = int(round(home_pts))

    pred_rows.append({
        "away_team": away,
        "away_predicted_points": round(away_pts, 2),
        "home_team": home,
        "home_predicted_points": round(home_pts, 2),
        "predicted_final_score": f"{home} {home_disp} - {away} {away_disp}"
    })

pred_df = pd.DataFrame(pred_rows)

print("\nPREDICTED SCORES (WEEK {})".format(WEEK_TO_PREDICT))
display(pred_df)

# =========================
# SAVE TO DESKTOP
# =========================
out_path = os.path.join(get_desktop_path(), OUT_BASENAME)
pred_df.to_csv(out_path, index=False)
print("\nSaved predictions to:", out_path)

```

    Rows after dropna:
      Train: 269
      Val  : 42
    Validation weeks: 15 to 16
    
    Model: XGBoost
    
    Validation performance:
      RMSE: 1.62
      MAE : 0.9
    
    FEATURE IMPORTANCE (top 30)
     1. points_per_drive                         0.446457
     2. drives                                   0.172036
     3. first_downs                              0.100224
     4. total_yards                              0.057721
     5. home_points_per_drive                    0.030930
     6. pass_tds                                 0.022181
     7. turnovers_per_drive                      0.019891
     8. home_points_per_drive_roll5              0.014355
     9. rush_tds                                 0.012938
    10. plays                                    0.009309
    11. yards_per_play                           0.008588
    12. pass_attempts                            0.008312
    13. pass_completions                         0.007726
    14. fumbles_lost                             0.006916
    15. sacked_yards_lost                        0.006472
    16. net_pass_yards                           0.005268
    17. pass_yards                               0.004498
    18. yards_per_play_roll5                     0.004190
    19. opp_interceptions                        0.004142
    20. sack_rate                                0.003871
    21. home_yards_per_play                      0.003630
    22. home_sack_rate                           0.003351
    23. opp_wind_mph                             0.003155
    24. points_allowed_roll5                     0.003127
    25. def_yards_per_play_allowed               0.003118
    26. def_points_per_drive_allowed_roll5       0.002868
    27. penalty_yards                            0.002024
    28. opp_sacked_yards_lost                    0.001970
    29. rush_attempts                            0.001802
    30. def_plays_faced                          0.001656
    
    PREDICTED SCORES (WEEK 17)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>away_team</th>
      <th>away_predicted_points</th>
      <th>home_team</th>
      <th>home_predicted_points</th>
      <th>predicted_final_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAL</td>
      <td>16.51</td>
      <td>WAS</td>
      <td>17.64</td>
      <td>WAS 18 - DAL 17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DET</td>
      <td>23.57</td>
      <td>MIN</td>
      <td>15.80</td>
      <td>MIN 16 - DET 24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DEN</td>
      <td>20.02</td>
      <td>KC</td>
      <td>10.47</td>
      <td>KC 10 - DEN 20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOU</td>
      <td>23.80</td>
      <td>LAC</td>
      <td>34.22</td>
      <td>LAC 34 - HOU 24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAL</td>
      <td>23.58</td>
      <td>GB</td>
      <td>14.71</td>
      <td>GB 15 - BAL 24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TB</td>
      <td>20.03</td>
      <td>MIA</td>
      <td>21.45</td>
      <td>MIA 21 - TB 20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NE</td>
      <td>28.13</td>
      <td>NYJ</td>
      <td>7.28</td>
      <td>NYJ 7 - NE 28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JAX</td>
      <td>32.60</td>
      <td>IND</td>
      <td>29.28</td>
      <td>IND 29 - JAX 33</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ARI</td>
      <td>19.87</td>
      <td>CIN</td>
      <td>41.38</td>
      <td>CIN 41 - ARI 20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PIT</td>
      <td>29.68</td>
      <td>CLE</td>
      <td>19.81</td>
      <td>CLE 20 - PIT 30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NO</td>
      <td>29.57</td>
      <td>TEN</td>
      <td>26.31</td>
      <td>TEN 26 - NO 30</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SEA</td>
      <td>38.37</td>
      <td>CAR</td>
      <td>23.46</td>
      <td>CAR 23 - SEA 38</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NYG</td>
      <td>15.85</td>
      <td>LV</td>
      <td>20.52</td>
      <td>LV 21 - NYG 16</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PHI</td>
      <td>30.09</td>
      <td>BUF</td>
      <td>23.42</td>
      <td>BUF 23 - PHI 30</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHI</td>
      <td>21.63</td>
      <td>SF</td>
      <td>42.52</td>
      <td>SF 43 - CHI 22</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LAR</td>
      <td>38.06</td>
      <td>ATL</td>
      <td>25.23</td>
      <td>ATL 25 - LAR 38</td>
    </tr>
  </tbody>
</table>
</div>


    
    Saved predictions to: C:\Users\eweis\Desktop\nfl_week_17_predictions.csv
    


```python

```


```python

```


```python
# With Interaction Terms 
# =========================
# NFL SCORE PREDICTOR (FULL / REVISED / CLEAN)
# - Uses the Excel dataset (team offense stats + opponent offense stats per game: opp_*)
# - Builds offense efficiency + TRUE defense (derived from opponent offense vs you)
# - Builds rolling last-5 features for BOTH offense and defense (shifted -> no leakage)
# - Includes home-vs-away effects (is_home + interactions)
# - Trains XGBoost if available (fallback if not)
# - Validates on last 2 completed weeks before WEEK_TO_PREDICT
# - Predicts BOTH teams' points for the games you list
# - Prints FEATURE IMPORTANCE (top 30)
# - Saves predictions CSV to Desktop
# =========================

import os
import numpy as np
import pandas as pd

# =========================
# SETTINGS YOU CHANGE
# =========================
DATA_FILE = r"C:\Users\eweis\Desktop\NFL_ML_thru_12.22.25.xlsx"
SHEET_NAME = 0

WEEK_TO_PREDICT = 17

# Put games here as: ("AWAY", "HOME")  <-- edit this list each week
WEEK_GAMES = [
    ("DAL", "WAS"),
    ("DET", "MIN"),
    ("DEN", "KC"),
    ("HOU", "LAC"),
    ("BAL", "GB"),
    ("TB", "MIA"),
    ("NE", "NYJ"),
    ("JAX", "IND"),
    ("ARI", "CIN"),
    ("PIT", "CLE"),
    ("NO", "TEN"),
    ("SEA", "CAR"),
    ("NYG", "LV"),
    ("PHI", "BUF"),
    ("CHI", "SF"),
    ("LA", "ATL"),
]

ROLL_WINDOW = 5
MIN_PERIODS = 3
TARGET = "points_scored"

OUT_BASENAME = f"nfl_week_{WEEK_TO_PREDICT}_predictions.csv"

# =========================
# HELPERS
# =========================
def get_desktop_path():
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else os.getcwd()

def normalize_team_code(x):
    if pd.isna(x):
        return x
    x = str(x).strip().upper()
    alias = {
        "JAC": "JAX",
        "WSH": "WAS",
        "WFT": "WAS",
        "LA": "LAR",   # Rams often appear as LA
    }
    return alias.get(x, x)

def safe_div(numer, denom):
    return np.where((denom == 0) | pd.isna(denom), np.nan, numer / denom)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# =========================
# MODEL SELECTION
# =========================
def choose_model():
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=0,
        )
        return model, "XGBoost", True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=600,
            random_state=42,
        )
        return model, "HistGradientBoostingRegressor", False

# =========================
# FEATURE ENGINEERING
# =========================
def add_offense_efficiency(df):
    df = df.copy()

    # basic counts
    df["plays"]  = df["pass_attempts"] + df["rush_attempts"]
    df["drives"] = safe_div(df["first_downs"], 2.5)  # rough proxy

    # offense efficiency
    df["yards_per_play"]       = safe_div(df["total_yards"], df["plays"])
    df["points_per_drive"]     = safe_div(df["points_scored"], df["drives"])
    df["turnovers_per_drive"]  = safe_div(df["turnovers"], df["drives"])
    df["sack_rate"]            = safe_div(df["sacked"], (df["pass_attempts"] + df["sacked"]))

    # home interactions (lets model learn different weights at home)
    df["home_yards_per_play"]   = df["yards_per_play"] * df["is_home"]
    df["home_points_per_drive"] = df["points_per_drive"] * df["is_home"]
    df["home_turnovers_per_drive"] = df["turnovers_per_drive"] * df["is_home"]
    df["home_sack_rate"]        = df["sack_rate"] * df["is_home"]

    return df

def add_true_defense_from_opponent_offense(df):
    """
    TRUE defense based on what opponent offense did AGAINST this team.
    Your dataset already contains opp_* per row, so this is real defensive info.
    """
    df = df.copy()

    # "allowed" totals
    df["points_allowed"] = df["opp_points_scored"]
    df["yards_allowed"]  = df["opp_total_yards"]

    # what the defense faced
    df["def_plays_faced"]  = df["opp_pass_attempts"] + df["opp_rush_attempts"]
    df["def_drives_faced"] = safe_div(df["opp_first_downs"], 2.5)

    # defensive efficiency (allowed rates)
    df["def_yards_per_play_allowed"]     = safe_div(df["yards_allowed"], df["def_plays_faced"])
    df["def_points_per_drive_allowed"]   = safe_div(df["points_allowed"], df["def_drives_faced"])

    # turnovers forced: opponent turnovers vs this team
    df["turnovers_forced"] = df["opp_turnovers"]
    df["def_turnovers_forced_per_drive"] = safe_div(df["turnovers_forced"], df["def_drives_faced"])

    # sack rate forced: opponent sacks taken / (opp pass attempts + opp sacks taken)
    df["def_sack_rate_forced"] = safe_div(df["opp_sacked"], (df["opp_pass_attempts"] + df["opp_sacked"]))

    # home interactions for defense too (home crowd effect, travel, etc.)
    df["home_def_points_per_drive_allowed"] = df["def_points_per_drive_allowed"] * df["is_home"]
    df["home_def_yards_per_play_allowed"]   = df["def_yards_per_play_allowed"] * df["is_home"]

    return df

def add_rolling_means(df, cols, window=5, min_periods=3):
    """
    Rolling mean over last N games PER TEAM, shifted by 1 so it uses only prior games.
    """
    df = df.copy()
    for c in cols:
        out = f"{c}_roll{window}"
        df[out] = (
            df.groupby("team")[c]
              .shift(1)
              .rolling(window=window, min_periods=min_periods)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

required_cols = [
    "team", "nfl_week", "is_home", TARGET,
    # offense
    "first_downs","rush_attempts","rush_yards","rush_tds",
    "pass_completions","pass_attempts","pass_yards","pass_tds",
    "interceptions","sacked","sacked_yards_lost","net_pass_yards","total_yards",
    "fumbles","fumbles_lost","turnovers","penalties","penalty_yards",
    # opponent offense (to build defense)
    "opp_team","opp_points_scored","opp_first_downs","opp_rush_attempts","opp_rush_yards","opp_rush_tds",
    "opp_pass_completions","opp_pass_attempts","opp_pass_yards","opp_pass_tds",
    "opp_interceptions","opp_sacked","opp_sacked_yards_lost","opp_net_pass_yards","opp_total_yards",
    "opp_fumbles","opp_fumbles_lost","opp_turnovers","opp_penalties","opp_penalty_yards",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in your Excel: {missing}")

# normalize teams
df["team"] = df["team"].apply(normalize_team_code)
df["opp_team"] = df["opp_team"].apply(normalize_team_code)

# week numeric
df["nfl_week"] = pd.to_numeric(df["nfl_week"], errors="coerce").astype("Int64")

# optional weather columns
if "wind_mph" not in df.columns:
    df["wind_mph"] = np.nan
if "opp_wind_mph" not in df.columns:
    df["opp_wind_mph"] = np.nan

# sort for rolling
df = df.sort_values(["team", "nfl_week"]).reset_index(drop=True)

# =========================
# BUILD FEATURES
# =========================
df = add_offense_efficiency(df)
df = add_true_defense_from_opponent_offense(df)

# clean inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# rolling offense form (includes scoring form)
OFF_ROLL_COLS = [
    "yards_per_play","points_per_drive","turnovers_per_drive","sack_rate",TARGET,
    "home_yards_per_play","home_points_per_drive","home_turnovers_per_drive","home_sack_rate"
]
df = add_rolling_means(df, OFF_ROLL_COLS, window=ROLL_WINDOW, min_periods=MIN_PERIODS)

# rolling TRUE defense form
DEF_ROLL_COLS = [
    "points_allowed","yards_allowed","def_yards_per_play_allowed","def_points_per_drive_allowed",
    "def_turnovers_forced_per_drive","def_sack_rate_forced",
    "home_def_points_per_drive_allowed","home_def_yards_per_play_allowed"
]
df = add_rolling_means(df, DEF_ROLL_COLS, window=ROLL_WINDOW, min_periods=MIN_PERIODS)

# =========================
# OFFENSE × DEFENSE INTERACTIONS (ADD THESE)
# =========================

# (1) Scoring efficiency vs opponent defensive resistance (recent form)
df["off_ppd_vs_def_ppd"] = (
    df[f"points_per_drive_roll{ROLL_WINDOW}"] /
    (df[f"def_points_per_drive_allowed_roll{ROLL_WINDOW}"] + 0.01)
)

# (2) Yardage mismatch (recent offense vs recent defense)
df["off_ypp_vs_def_ypp"] = (
    df[f"yards_per_play_roll{ROLL_WINDOW}"] -
    df[f"def_yards_per_play_allowed_roll{ROLL_WINDOW}"]
)

# (3) Pace × efficiency (helps distinguish “empty yards” vs sustained offense)
df["pace_efficiency"] = df["plays"] * df["yards_per_play"]

# (4) Turnover clash: your turnover tendency × their turnover-forcing tendency
df["turnover_pressure"] = (
    df[f"turnovers_per_drive_roll{ROLL_WINDOW}"] *
    df[f"def_turnovers_forced_per_drive_roll{ROLL_WINDOW}"]
)

# (5) Sack clash: your sack rate × their sack rate forced
df["sack_pressure"] = (
    df[f"sack_rate_roll{ROLL_WINDOW}"] *
    df[f"def_sack_rate_forced_roll{ROLL_WINDOW}"]
)

# (6) Home boost: home × scoring efficiency (recent)
df["home_ppd_boost"] = df["is_home"] * df[f"points_per_drive_roll{ROLL_WINDOW}"]

# (7) Home defense effect: home × defensive stinginess (recent)
df["home_def_suppression"] = df["is_home"] * df[f"def_points_per_drive_allowed_roll{ROLL_WINDOW}"]

# clean any inf created by ratios
df.replace([np.inf, -np.inf], np.nan, inplace=True)




# =========================
# FEATURES LIST (FULL)
# =========================
FEATURES = [
    # context
    "is_home",
    "wind_mph",

    # raw offense stats
    "first_downs","rush_attempts","rush_yards","rush_tds",
    "pass_completions","pass_attempts","pass_yards","pass_tds",
    "interceptions","sacked","sacked_yards_lost","net_pass_yards","total_yards",
    "fumbles","fumbles_lost","turnovers","penalties","penalty_yards",

    # offense efficiency + home interactions
    "plays","drives",
    "yards_per_play","points_per_drive","turnovers_per_drive","sack_rate",
    "home_yards_per_play","home_points_per_drive","home_turnovers_per_drive","home_sack_rate",

    # opponent offense raw (still useful as matchup signal)
    "opp_points_scored","opp_first_downs","opp_rush_attempts","opp_rush_yards","opp_rush_tds",
    "opp_pass_completions","opp_pass_attempts","opp_pass_yards","opp_pass_tds",
    "opp_interceptions","opp_sacked","opp_sacked_yards_lost","opp_net_pass_yards","opp_total_yards",
    "opp_fumbles","opp_fumbles_lost","opp_turnovers","opp_penalties","opp_penalty_yards",
    "opp_wind_mph",

    # TRUE defense derived from opp_*
    "points_allowed","yards_allowed","def_plays_faced","def_drives_faced",
    "def_yards_per_play_allowed","def_points_per_drive_allowed",
    "turnovers_forced","def_turnovers_forced_per_drive","def_sack_rate_forced",
    "home_def_points_per_drive_allowed","home_def_yards_per_play_allowed",
]

# add rolling columns (offense + defense)
for c in OFF_ROLL_COLS:
    FEATURES.append(f"{c}_roll{ROLL_WINDOW}")
for c in DEF_ROLL_COLS:
    FEATURES.append(f"{c}_roll{ROLL_WINDOW}")

# keep only existing columns (avoids KeyErrors if something is absent)
FEATURES = [c for c in FEATURES if c in df.columns]


INTERACTION_FEATURES = [
    "off_ppd_vs_def_ppd",
    "off_ypp_vs_def_ypp",
    "pace_efficiency",
    "turnover_pressure",
    "sack_pressure",
    "home_ppd_boost",
    "home_def_suppression",
]

FEATURES += [c for c in INTERACTION_FEATURES if c in df.columns]


# =========================
# TRAIN / VALIDATION SPLIT
# validate on last 2 completed weeks before WEEK_TO_PREDICT
# =========================
val_start = max(1, WEEK_TO_PREDICT - 2)
val_end   = WEEK_TO_PREDICT - 1

train_df = df[df["nfl_week"] < val_start].copy()
val_df   = df[df["nfl_week"].between(val_start, val_end)].copy()

# drop rows missing needed predictors or target
train_df = train_df.dropna(subset=FEATURES + [TARGET])
val_df   = val_df.dropna(subset=FEATURES + [TARGET])

print("Rows after dropna:")
print("  Train:", len(train_df))
print("  Val  :", len(val_df))
print(f"Validation weeks: {val_start} to {val_end}")

# =========================
# TRAIN MODEL
# =========================
model, model_name, is_xgb = choose_model()
print("\nModel:", model_name)

X_train = train_df[FEATURES].astype(float)
y_train = train_df[TARGET].astype(float)

X_val = val_df[FEATURES].astype(float)
y_val = val_df[TARGET].astype(float)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
print("\nValidation performance:")
print("  RMSE:", round(rmse(y_val, val_preds), 2))
print("  MAE :", round(mae(y_val, val_preds), 2))

# =========================
# FEATURE IMPORTANCE (TOP 30)
# =========================
def print_feature_importance(model, feature_names, X_ref, y_ref, top_n=30, use_xgb=False):
    print("\nFEATURE IMPORTANCE (top {})".format(top_n))

    if use_xgb and hasattr(model, "feature_importances_"):
        imps = np.array(model.feature_importances_, dtype=float)
        order = np.argsort(imps)[::-1][:top_n]
        for i, idx in enumerate(order, 1):
            print(f"{i:>2}. {feature_names[idx]:<40} {imps[idx]:.6f}")
        return

    # fallback: permutation importance (slower but works on any sklearn model)
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            model, X_ref, y_ref,
            n_repeats=10, random_state=42, scoring="neg_mean_absolute_error"
        )
        imps = r.importances_mean
        order = np.argsort(imps)[::-1][:top_n]
        for i, idx in enumerate(order, 1):
            print(f"{i:>2}. {feature_names[idx]:<40} {imps[idx]:.6f}")
    except Exception as e:
        print("Could not compute permutation importance:", e)

print_feature_importance(model, FEATURES, X_val, y_val, top_n=30, use_xgb=is_xgb)

# =========================
# BUILD MATCHUP FEATURES (OFFENSE vs OPPONENT DEFENSE)
# =========================
df_sorted = df.sort_values(["team", "nfl_week"]).reset_index(drop=True)
global_means = df_sorted[FEATURES].mean(numeric_only=True)

# Now we inject THESE opponent DEFENSE rolling signals into the team’s feature row
OPP_DEF_ROLL = [
    f"def_points_per_drive_allowed_roll{ROLL_WINDOW}",
    f"def_yards_per_play_allowed_roll{ROLL_WINDOW}",
    f"def_turnovers_forced_per_drive_roll{ROLL_WINDOW}",
    f"def_sack_rate_forced_roll{ROLL_WINDOW}",
    f"points_allowed_roll{ROLL_WINDOW}",
    f"yards_allowed_roll{ROLL_WINDOW}",
]

def get_latest_row_before_week(team, week):
    team = normalize_team_code(team)
    sub = df_sorted[(df_sorted["team"] == team) & (df_sorted["nfl_week"] < week)].sort_values("nfl_week")
    if len(sub) == 0:
        return None
    return sub.iloc[-1]

def build_feature_row(team, opponent, is_home, week):
    team = normalize_team_code(team)
    opponent = normalize_team_code(opponent)

    base = get_latest_row_before_week(team, week)
    if base is None:
        row = global_means.copy()
    else:
        row = base[FEATURES].copy()

    # set is_home
    if "is_home" in row.index:
        row["is_home"] = int(is_home)

    # wind: unknown for future -> use team mean so far
    if "wind_mph" in row.index:
        row["wind_mph"] = df_sorted.loc[df_sorted["team"] == team, "wind_mph"].mean()

    # re-compute home interaction columns for the FUTURE row (important!)
    # because we changed is_home.
    # (Only recompute if the columns exist.)
    def recompute_home_interactions(r):
        # needs base efficiency columns available; if NaN, stays NaN
        if "home_yards_per_play" in r.index and "yards_per_play" in r.index and "is_home" in r.index:
            r["home_yards_per_play"] = r["yards_per_play"] * r["is_home"]
        if "home_points_per_drive" in r.index and "points_per_drive" in r.index and "is_home" in r.index:
            r["home_points_per_drive"] = r["points_per_drive"] * r["is_home"]
        if "home_turnovers_per_drive" in r.index and "turnovers_per_drive" in r.index and "is_home" in r.index:
            r["home_turnovers_per_drive"] = r["turnovers_per_drive"] * r["is_home"]
        if "home_sack_rate" in r.index and "sack_rate" in r.index and "is_home" in r.index:
            r["home_sack_rate"] = r["sack_rate"] * r["is_home"]

        if "home_def_points_per_drive_allowed" in r.index and "def_points_per_drive_allowed" in r.index and "is_home" in r.index:
            r["home_def_points_per_drive_allowed"] = r["def_points_per_drive_allowed"] * r["is_home"]
        if "home_def_yards_per_play_allowed" in r.index and "def_yards_per_play_allowed" in r.index and "is_home" in r.index:
            r["home_def_yards_per_play_allowed"] = r["def_yards_per_play_allowed"] * r["is_home"]
        return r

    row = recompute_home_interactions(row)

    # Inject opponent TRUE DEFENSE rolling features (THIS is the key improvement)
    opp_latest = get_latest_row_before_week(opponent, week)
    if opp_latest is not None:
        for f in OPP_DEF_ROLL:
            if (f in row.index) and (f in opp_latest.index):
                row[f] = opp_latest[f]

    # fill remaining missing values with global means
    row = row.astype(float).fillna(global_means)

    return row.to_dict()

# =========================
# PREDICT WEEK GAMES
# =========================
pred_rows = []
for away, home in WEEK_GAMES:
    away = normalize_team_code(away)
    home = normalize_team_code(home)

    away_feat = build_feature_row(away, home, is_home=0, week=WEEK_TO_PREDICT)
    home_feat = build_feature_row(home, away, is_home=1, week=WEEK_TO_PREDICT)

    away_pts = float(model.predict(pd.DataFrame([away_feat])[FEATURES])[0])
    home_pts = float(model.predict(pd.DataFrame([home_feat])[FEATURES])[0])

    away_disp = int(round(away_pts))
    home_disp = int(round(home_pts))

    pred_rows.append({
        "away_team": away,
        "away_predicted_points": round(away_pts, 2),
        "home_team": home,
        "home_predicted_points": round(home_pts, 2),
        "predicted_final_score": f"{home} {home_disp} - {away} {away_disp}"
    })

pred_df = pd.DataFrame(pred_rows)

print("\nPREDICTED SCORES (WEEK {})".format(WEEK_TO_PREDICT))
display(pred_df)

# =========================
# SAVE TO DESKTOP
# =========================
out_path = os.path.join(get_desktop_path(), OUT_BASENAME)
pred_df.to_csv(out_path, index=False)
print("\nSaved predictions to:", out_path)

```

    Rows after dropna:
      Train: 269
      Val  : 42
    Validation weeks: 15 to 16
    
    Model: XGBoost
    
    Validation performance:
      RMSE: 1.71
      MAE : 1.08
    
    FEATURE IMPORTANCE (top 30)
     1. points_per_drive                         0.302199
     2. pass_tds                                 0.107550
     3. yards_per_play                           0.084741
     4. total_yards                              0.077609
     5. first_downs                              0.071892
     6. rush_tds                                 0.069612
     7. home_points_per_drive                    0.052632
     8. drives                                   0.043242
     9. rush_attempts                            0.023478
    10. turnovers_per_drive                      0.018069
    11. turnovers                                0.017453
    12. home_ppd_boost                           0.012072
    13. home_points_per_drive_roll5              0.009457
    14. net_pass_yards                           0.007993
    15. plays                                    0.007650
    16. opp_total_yards                          0.007506
    17. def_yards_per_play_allowed               0.006235
    18. penalties                                0.005174
    19. pass_completions                         0.005012
    20. opp_sacked_yards_lost                    0.003750
    21. home_def_yards_per_play_allowed          0.003475
    22. def_plays_faced                          0.003471
    23. sack_rate_roll5                          0.003305
    24. pass_attempts                            0.003288
    25. def_yards_per_play_allowed_roll5         0.002953
    26. yards_per_play_roll5                     0.002897
    27. pace_efficiency                          0.002464
    28. home_yards_per_play                      0.002422
    29. pass_yards                               0.002392
    30. opp_penalties                            0.002176
    
    PREDICTED SCORES (WEEK 17)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>away_team</th>
      <th>away_predicted_points</th>
      <th>home_team</th>
      <th>home_predicted_points</th>
      <th>predicted_final_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAL</td>
      <td>17.29</td>
      <td>WAS</td>
      <td>17.62</td>
      <td>WAS 18 - DAL 17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DET</td>
      <td>23.49</td>
      <td>MIN</td>
      <td>16.06</td>
      <td>MIN 16 - DET 23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DEN</td>
      <td>20.50</td>
      <td>KC</td>
      <td>10.27</td>
      <td>KC 10 - DEN 20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOU</td>
      <td>22.74</td>
      <td>LAC</td>
      <td>35.41</td>
      <td>LAC 35 - HOU 23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAL</td>
      <td>24.07</td>
      <td>GB</td>
      <td>14.65</td>
      <td>GB 15 - BAL 24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TB</td>
      <td>20.23</td>
      <td>MIA</td>
      <td>22.31</td>
      <td>MIA 22 - TB 20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NE</td>
      <td>28.01</td>
      <td>NYJ</td>
      <td>7.01</td>
      <td>NYJ 7 - NE 28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JAX</td>
      <td>32.77</td>
      <td>IND</td>
      <td>28.70</td>
      <td>IND 29 - JAX 33</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ARI</td>
      <td>19.14</td>
      <td>CIN</td>
      <td>40.13</td>
      <td>CIN 40 - ARI 19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PIT</td>
      <td>30.34</td>
      <td>CLE</td>
      <td>19.82</td>
      <td>CLE 20 - PIT 30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NO</td>
      <td>28.05</td>
      <td>TEN</td>
      <td>26.99</td>
      <td>TEN 27 - NO 28</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SEA</td>
      <td>38.19</td>
      <td>CAR</td>
      <td>24.09</td>
      <td>CAR 24 - SEA 38</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NYG</td>
      <td>16.25</td>
      <td>LV</td>
      <td>21.27</td>
      <td>LV 21 - NYG 16</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PHI</td>
      <td>29.77</td>
      <td>BUF</td>
      <td>24.31</td>
      <td>BUF 24 - PHI 30</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHI</td>
      <td>21.88</td>
      <td>SF</td>
      <td>42.23</td>
      <td>SF 42 - CHI 22</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LAR</td>
      <td>38.64</td>
      <td>ATL</td>
      <td>25.61</td>
      <td>ATL 26 - LAR 39</td>
    </tr>
  </tbody>
</table>
</div>


    
    Saved predictions to: C:\Users\eweis\Desktop\nfl_week_17_predictions.csv
    


```python

```


```python

```


```python
# Revised to reflect points per drive 
# =========================
# NFL SCORE PREDICTOR (DRIVES EXPLICIT + MATCHUP DEFENSE + INTERACTIONS)
# FULL / CLEAN / PASTE-AND-RUN
#
# What this does:
# 1) Loads the Excel: team offense + opponent offense (opp_*) per game
# 2) Builds offense efficiency + TRUE defense (from opponent offense vs you)
# 3) Builds rolling last-5 features (shifted so no leakage)
# 4) Merges OPPONENT defensive rolling form for the SAME week (matchup learning)
# 5) Trains TWO models:
#     A) drives_model   -> predicts drives
#     B) ppd_model      -> predicts points_per_drive
#    Final predicted points = predicted_drives * predicted_ppd
# 6) Validates on last 2 completed weeks before WEEK_TO_PREDICT
# 7) Predicts scores for the games you list
# 8) Prints feature importance (top 30) for BOTH models
# 9) Saves predictions CSV to Desktop
# =========================

import os
import numpy as np
import pandas as pd

# =========================
# SETTINGS YOU CHANGE
# =========================
DATA_FILE = r"C:\Users\eweis\Desktop\NFL_ML_thru_12.22.25.xlsx"
SHEET_NAME = 0

WEEK_TO_PREDICT = 17

# Put games here as: ("AWAY", "HOME")
WEEK_GAMES = [
    ("DAL", "WAS"),
    ("DET", "MIN"),
    ("DEN", "KC"),
    ("HOU", "LAC"),
    ("BAL", "GB"),
    ("TB", "MIA"),
    ("NE", "NYJ"),
    ("JAX", "IND"),
    ("ARI", "CIN"),
    ("PIT", "CLE"),
    ("NO", "TEN"),
    ("SEA", "CAR"),
    ("NYG", "LV"),
    ("PHI", "BUF"),
    ("CHI", "SF"),
    ("LA", "ATL"),
]

ROLL_WINDOW = 5
MIN_PERIODS = 3

OUT_BASENAME = f"nfl_week_{WEEK_TO_PREDICT}_predictions.csv"


# =========================
# HELPERS
# =========================
def get_desktop_path():
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else os.getcwd()

def normalize_team_code(x):
    if pd.isna(x):
        return x
    x = str(x).strip().upper()
    alias = {
        "JAC": "JAX",
        "WSH": "WAS",
        "WFT": "WAS",
        "LA": "LAR",  # Rams often appear as LA
    }
    return alias.get(x, x)

def safe_div(numer, denom):
    return np.where((denom == 0) | pd.isna(denom), np.nan, numer / denom)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# =========================
# MODEL SELECTION
# =========================
def choose_model():
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=0,
        )
        return model, "XGBoost", True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=600,
            random_state=42,
        )
        return model, "HistGradientBoostingRegressor", False

# =========================
# FEATURE ENGINEERING
# =========================
def add_offense_efficiency(df):
    df = df.copy()

    # basic counts
    df["plays"]  = df["pass_attempts"] + df["rush_attempts"]
    df["drives"] = safe_div(df["first_downs"], 2.5)  # proxy drives

    # offense efficiency (TARGET-LIKE, but OK as TARGETS; we will NOT use non-rolling as features)
    df["yards_per_play"]       = safe_div(df["total_yards"], df["plays"])
    df["points_per_drive"]     = safe_div(df["points_scored"], df["drives"])
    df["turnovers_per_drive"]  = safe_div(df["turnovers"], df["drives"])
    df["sack_rate"]            = safe_div(df["sacked"], (df["pass_attempts"] + df["sacked"]))

    # home interactions (we WILL use rolling versions of these as features)
    df["home_yards_per_play"]        = df["yards_per_play"] * df["is_home"]
    df["home_points_per_drive"]      = df["points_per_drive"] * df["is_home"]
    df["home_turnovers_per_drive"]   = df["turnovers_per_drive"] * df["is_home"]
    df["home_sack_rate"]             = df["sack_rate"] * df["is_home"]

    return df

def add_true_defense_from_opponent_offense(df):
    """
    TRUE defense based on what opponent offense did AGAINST this team.
    Uses opp_* columns (already in your file).
    """
    df = df.copy()

    # allowed totals
    df["points_allowed"] = df["opp_points_scored"]
    df["yards_allowed"]  = df["opp_total_yards"]

    # what the defense faced
    df["def_plays_faced"]  = df["opp_pass_attempts"] + df["opp_rush_attempts"]
    df["def_drives_faced"] = safe_div(df["opp_first_downs"], 2.5)

    # defensive efficiency (allowed rates)
    df["def_yards_per_play_allowed"]   = safe_div(df["yards_allowed"], df["def_plays_faced"])
    df["def_points_per_drive_allowed"] = safe_div(df["points_allowed"], df["def_drives_faced"])

    # turnovers forced
    df["turnovers_forced"] = df["opp_turnovers"]
    df["def_turnovers_forced_per_drive"] = safe_div(df["turnovers_forced"], df["def_drives_faced"])

    # sack rate forced
    df["def_sack_rate_forced"] = safe_div(df["opp_sacked"], (df["opp_pass_attempts"] + df["opp_sacked"]))

    # home defense interactions
    df["home_def_points_per_drive_allowed"] = df["def_points_per_drive_allowed"] * df["is_home"]
    df["home_def_yards_per_play_allowed"]   = df["def_yards_per_play_allowed"] * df["is_home"]

    return df

def add_rolling_means(df, cols, window=5, min_periods=3):
    """
    Rolling mean over last N games PER TEAM, shifted by 1 so it uses only prior games.
    """
    df = df.copy()
    for c in cols:
        out = f"{c}_roll{window}"
        df[out] = (
            df.groupby("team")[c]
              .shift(1)
              .rolling(window=window, min_periods=min_periods)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df


# =========================
# LOAD DATA
# =========================
df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
df.columns = [c.strip() for c in df.columns]

required_cols = [
    "team", "nfl_week", "is_home", "points_scored",
    # offense
    "first_downs","rush_attempts","rush_yards","rush_tds",
    "pass_completions","pass_attempts","pass_yards","pass_tds",
    "interceptions","sacked","sacked_yards_lost","net_pass_yards","total_yards",
    "fumbles","fumbles_lost","turnovers","penalties","penalty_yards",
    # opponent offense (for defense)
    "opp_team","opp_points_scored","opp_first_downs","opp_rush_attempts","opp_rush_yards","opp_rush_tds",
    "opp_pass_completions","opp_pass_attempts","opp_pass_yards","opp_pass_tds",
    "opp_interceptions","opp_sacked","opp_sacked_yards_lost","opp_net_pass_yards","opp_total_yards",
    "opp_fumbles","opp_fumbles_lost","opp_turnovers","opp_penalties","opp_penalty_yards",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in your Excel: {missing}")

# normalize team codes
df["team"] = df["team"].apply(normalize_team_code)
df["opp_team"] = df["opp_team"].apply(normalize_team_code)

# week numeric
df["nfl_week"] = pd.to_numeric(df["nfl_week"], errors="coerce").astype("Int64")

# optional weather
if "wind_mph" not in df.columns:
    df["wind_mph"] = np.nan
if "opp_wind_mph" not in df.columns:
    df["opp_wind_mph"] = np.nan

# sort
df = df.sort_values(["team", "nfl_week"]).reset_index(drop=True)

# =========================
# BUILD BASE FEATURES + ROLLING FORM
# =========================
df = add_offense_efficiency(df)
df = add_true_defense_from_opponent_offense(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# These will be TARGETS for the two-model approach:
# - drives
# - points_per_drive

# Rolling OFFENSE form
OFF_ROLL_COLS = [
    "yards_per_play","turnovers_per_drive","sack_rate",
    "plays","first_downs","rush_attempts","pass_attempts",
    "home_yards_per_play","home_turnovers_per_drive","home_sack_rate",
    "points_scored","points_per_drive","drives"
]
df = add_rolling_means(df, OFF_ROLL_COLS, window=ROLL_WINDOW, min_periods=MIN_PERIODS)

# Rolling TRUE DEFENSE form
DEF_ROLL_COLS = [
    "points_allowed","yards_allowed",
    "def_plays_faced","def_drives_faced",
    "def_yards_per_play_allowed","def_points_per_drive_allowed",
    "def_turnovers_forced_per_drive","def_sack_rate_forced",
    "home_def_points_per_drive_allowed","home_def_yards_per_play_allowed"
]
df = add_rolling_means(df, DEF_ROLL_COLS, window=ROLL_WINDOW, min_periods=MIN_PERIODS)

# =========================
# MERGE OPPONENT DEFENSIVE ROLLING FORM (MATCHUP LEARNING)
# For each row (team vs opp_team in week w), attach opp_team's defensive roll stats entering week w.
# =========================
OPP_DEF_ROLL_COLS = [
    f"def_points_per_drive_allowed_roll{ROLL_WINDOW}",
    f"def_yards_per_play_allowed_roll{ROLL_WINDOW}",
    f"def_turnovers_forced_per_drive_roll{ROLL_WINDOW}",
    f"def_sack_rate_forced_roll{ROLL_WINDOW}",
    f"points_allowed_roll{ROLL_WINDOW}",
    f"yards_allowed_roll{ROLL_WINDOW}",
]

opp_def = df[["team", "nfl_week"] + OPP_DEF_ROLL_COLS].copy()
opp_def = opp_def.rename(columns={"team": "opp_team"})
opp_def = opp_def.rename(columns={c: f"opp_{c}" for c in OPP_DEF_ROLL_COLS})

df = df.merge(opp_def, on=["opp_team", "nfl_week"], how="left")

# =========================
# INTERACTIONS (USE ROLLING + OPPONENT DEFENSE ROLLING)
# =========================
# Safe tiny constant to avoid division-by-zero
EPS = 0.01

# Offense scoring efficiency (recent) vs opponent defensive resistance (recent)
df["off_ppd_vs_opp_def_ppd"] = (
    df[f"points_per_drive_roll{ROLL_WINDOW}"] /
    (df[f"opp_def_points_per_drive_allowed_roll{ROLL_WINDOW}"] + EPS)
)

# Offense yardage efficiency (recent) vs opponent defensive yardage allowed (recent)
df["off_ypp_vs_opp_def_ypp"] = (
    df[f"yards_per_play_roll{ROLL_WINDOW}"] -
    df[f"opp_def_yards_per_play_allowed_roll{ROLL_WINDOW}"]
)

# Turnover clash: your turnover tendency × their turnover-forcing tendency
df["turnover_pressure"] = (
    df[f"turnovers_per_drive_roll{ROLL_WINDOW}"] *
    df[f"opp_def_turnovers_forced_per_drive_roll{ROLL_WINDOW}"]
)

# Sack clash: your sack tendency × their sack forcing tendency
df["sack_pressure"] = (
    df[f"sack_rate_roll{ROLL_WINDOW}"] *
    df[f"opp_def_sack_rate_forced_roll{ROLL_WINDOW}"]
)

# Pace proxy: plays (recent) × ypp (recent)
df["pace_efficiency"] = df[f"plays_roll{ROLL_WINDOW}"] * df[f"yards_per_play_roll{ROLL_WINDOW}"]

# Home boost on offense efficiency
df["home_ppd_boost"] = df["is_home"] * df[f"points_per_drive_roll{ROLL_WINDOW}"]

# Home suppression on opponent defense metric (lets model learn home advantage vs opponent defense)
df["home_opp_def_suppression"] = df["is_home"] * df[f"opp_def_points_per_drive_allowed_roll{ROLL_WINDOW}"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =========================
# FEATURES USED FOR TRAINING (PREGAME-AVAILABLE ONLY)
# IMPORTANT: We DO NOT use same-game raw totals like total_yards to predict same-game score.
# We use rolling (last-5) form + opponent defense rolling form + context.
# =========================
FEATURES = [
    "is_home",
    "wind_mph",

    # Team offense form (rolling)
    f"yards_per_play_roll{ROLL_WINDOW}",
    f"turnovers_per_drive_roll{ROLL_WINDOW}",
    f"sack_rate_roll{ROLL_WINDOW}",
    f"plays_roll{ROLL_WINDOW}",
    f"first_downs_roll{ROLL_WINDOW}",
    f"rush_attempts_roll{ROLL_WINDOW}",
    f"pass_attempts_roll{ROLL_WINDOW}",
    f"points_scored_roll{ROLL_WINDOW}",
    f"points_per_drive_roll{ROLL_WINDOW}",
    f"drives_roll{ROLL_WINDOW}",

    # Team defense form (rolling)
    f"def_points_per_drive_allowed_roll{ROLL_WINDOW}",
    f"def_yards_per_play_allowed_roll{ROLL_WINDOW}",
    f"def_turnovers_forced_per_drive_roll{ROLL_WINDOW}",
    f"def_sack_rate_forced_roll{ROLL_WINDOW}",

    # Opponent defensive form (rolling, merged)
    f"opp_def_def_points_per_drive_allowed_roll{ROLL_WINDOW}",
    f"opp_def_def_yards_per_play_allowed_roll{ROLL_WINDOW}",
    f"opp_def_def_turnovers_forced_per_drive_roll{ROLL_WINDOW}",
    f"opp_def_def_sack_rate_forced_roll{ROLL_WINDOW}",

    # Interactions
    "off_ppd_vs_opp_def_ppd",
    "off_ypp_vs_opp_def_ypp",
    "turnover_pressure",
    "sack_pressure",
    "pace_efficiency",
    "home_ppd_boost",
    "home_opp_def_suppression",
]

# Keep only columns that exist
FEATURES = [c for c in FEATURES if c in df.columns]

# =========================
# TRAIN / VALIDATION SPLIT
# Validate on last 2 completed weeks before WEEK_TO_PREDICT
# =========================
val_start = max(1, WEEK_TO_PREDICT - 2)
val_end   = WEEK_TO_PREDICT - 1

train_df = df[df["nfl_week"] < val_start].copy()
val_df   = df[df["nfl_week"].between(val_start, val_end)].copy()

# Targets for two models
DRIVES_TARGET = "drives"
PPD_TARGET    = "points_per_drive"

# Drop rows missing needed predictors or targets
train_df = train_df.dropna(subset=FEATURES + [DRIVES_TARGET, PPD_TARGET])
val_df   = val_df.dropna(subset=FEATURES + [DRIVES_TARGET, PPD_TARGET])

print("Rows after dropna:")
print("  Train:", len(train_df))
print("  Val  :", len(val_df))
print(f"Validation weeks: {val_start} to {val_end}")

# =========================
# TRAIN TWO MODELS
# =========================
drives_model, drives_name, drives_is_xgb = choose_model()
ppd_model, ppd_name, ppd_is_xgb = choose_model()

print("\nDrives model:", drives_name)
print("PPD model   :", ppd_name)

X_train = train_df[FEATURES].astype(float)
X_val   = val_df[FEATURES].astype(float)

y_train_drives = train_df[DRIVES_TARGET].astype(float)
y_train_ppd    = train_df[PPD_TARGET].astype(float)

y_val_drives = val_df[DRIVES_TARGET].astype(float)
y_val_ppd    = val_df[PPD_TARGET].astype(float)

drives_model.fit(X_train, y_train_drives)
ppd_model.fit(X_train, y_train_ppd)

# Validation: drives + ppd + implied points
val_pred_drives = drives_model.predict(X_val)
val_pred_ppd    = ppd_model.predict(X_val)
val_pred_points = val_pred_drives * val_pred_ppd
val_true_points = val_df["points_scored"].astype(float).values

print("\nValidation performance:")
print("  Drives RMSE:", round(rmse(y_val_drives, val_pred_drives), 2), " MAE:", round(mae(y_val_drives, val_pred_drives), 2))
print("  PPD RMSE   :", round(rmse(y_val_ppd, val_pred_ppd), 2),      " MAE:", round(mae(y_val_ppd, val_pred_ppd), 2))
print("  POINTS (drives*ppd) RMSE:", round(rmse(val_true_points, val_pred_points), 2),
      " MAE:", round(mae(val_true_points, val_pred_points), 2))

# =========================
# FEATURE IMPORTANCE (TOP 30) FOR BOTH MODELS
# =========================
def print_feature_importance(model, feature_names, top_n=30, use_xgb=False, title="FEATURE IMPORTANCE"):
    print(f"\n{title} (top {top_n})")
    if use_xgb and hasattr(model, "feature_importances_"):
        imps = np.array(model.feature_importances_, dtype=float)
        order = np.argsort(imps)[::-1][:top_n]
        for i, idx in enumerate(order, 1):
            print(f"{i:>2}. {feature_names[idx]:<45} {imps[idx]:.6f}")
        return

    # Permutation importance fallback
    try:
        from sklearn.inspection import permutation_importance
        # Use a small subset for speed if needed
        # (You can raise n_repeats later)
        r = permutation_importance(
            model, X_val, model.predict(X_val),
            n_repeats=10, random_state=42, scoring=None
        )
        imps = r.importances_mean
        order = np.argsort(imps)[::-1][:top_n]
        for i, idx in enumerate(order, 1):
            print(f"{i:>2}. {feature_names[idx]:<45} {imps[idx]:.6f}")
    except Exception as e:
        print("Could not compute permutation importance:", e)

print_feature_importance(drives_model, FEATURES, top_n=30, use_xgb=drives_is_xgb, title="DRIVES MODEL IMPORTANCE")
print_feature_importance(ppd_model, FEATURES, top_n=30, use_xgb=ppd_is_xgb, title="PPD MODEL IMPORTANCE")

# =========================
# BUILD FEATURE ROW FOR FUTURE MATCHUP (Week to predict)
# Uses latest available rolling form BEFORE week W.
# Also injects opponent defensive rolling form.
# =========================
df_sorted = df.sort_values(["team", "nfl_week"]).reset_index(drop=True)
global_means = df_sorted[FEATURES].mean(numeric_only=True)

def get_latest_row_before_week(team, week):
    team = normalize_team_code(team)
    sub = df_sorted[(df_sorted["team"] == team) & (df_sorted["nfl_week"] < week)].sort_values("nfl_week")
    if len(sub) == 0:
        return None
    return sub.iloc[-1]

def build_feature_row(team, opponent, is_home, week):
    team = normalize_team_code(team)
    opponent = normalize_team_code(opponent)

    base = get_latest_row_before_week(team, week)
    opp  = get_latest_row_before_week(opponent, week)

    if base is None:
        row = global_means.copy()
    else:
        row = base[FEATURES].copy()

    # set context
    if "is_home" in row.index:
        row["is_home"] = int(is_home)

    # future wind unknown -> use team mean
    if "wind_mph" in row.index:
        row["wind_mph"] = df_sorted.loc[df_sorted["team"] == team, "wind_mph"].mean()

    # Inject opponent defensive rolling form into opponent columns if present
    if opp is not None:
        for c in FEATURES:
            if c.startswith("opp_def_"):
                if c in opp.index:
                    row[c] = opp[c]

    # Recompute interaction features for THIS matchup (important!)
    def getv(name):
        return float(row[name]) if name in row.index and pd.notna(row[name]) else np.nan

    # Only compute if required pieces exist
    if f"points_per_drive_roll{ROLL_WINDOW}" in row.index and f"opp_def_def_points_per_drive_allowed_roll{ROLL_WINDOW}" in row.index:
        row["off_ppd_vs_opp_def_ppd"] = getv(f"points_per_drive_roll{ROLL_WINDOW}") / (getv(f"opp_def_def_points_per_drive_allowed_roll{ROLL_WINDOW}") + EPS)

    if f"yards_per_play_roll{ROLL_WINDOW}" in row.index and f"opp_def_def_yards_per_play_allowed_roll{ROLL_WINDOW}" in row.index:
        row["off_ypp_vs_opp_def_ypp"] = getv(f"yards_per_play_roll{ROLL_WINDOW}") - getv(f"opp_def_def_yards_per_play_allowed_roll{ROLL_WINDOW}")

    if f"turnovers_per_drive_roll{ROLL_WINDOW}" in row.index and f"opp_def_def_turnovers_forced_per_drive_roll{ROLL_WINDOW}" in row.index:
        row["turnover_pressure"] = getv(f"turnovers_per_drive_roll{ROLL_WINDOW}") * getv(f"opp_def_def_turnovers_forced_per_drive_roll{ROLL_WINDOW}")

    if f"sack_rate_roll{ROLL_WINDOW}" in row.index and f"opp_def_def_sack_rate_forced_roll{ROLL_WINDOW}" in row.index:
        row["sack_pressure"] = getv(f"sack_rate_roll{ROLL_WINDOW}") * getv(f"opp_def_def_sack_rate_forced_roll{ROLL_WINDOW}")

    if f"plays_roll{ROLL_WINDOW}" in row.index and f"yards_per_play_roll{ROLL_WINDOW}" in row.index:
        row["pace_efficiency"] = getv(f"plays_roll{ROLL_WINDOW}") * getv(f"yards_per_play_roll{ROLL_WINDOW}")

    if "is_home" in row.index and f"points_per_drive_roll{ROLL_WINDOW}" in row.index:
        row["home_ppd_boost"] = getv("is_home") * getv(f"points_per_drive_roll{ROLL_WINDOW}")

    if "is_home" in row.index and f"opp_def_def_points_per_drive_allowed_roll{ROLL_WINDOW}" in row.index:
        row["home_opp_def_suppression"] = getv("is_home") * getv(f"opp_def_def_points_per_drive_allowed_roll{ROLL_WINDOW}")

    # Fill missing with global means
    row = row.astype(float).replace([np.inf, -np.inf], np.nan).fillna(global_means)

    return row.to_dict()

# =========================
# PREDICT WEEK GAMES
# =========================
pred_rows = []
for away, home in WEEK_GAMES:
    away = normalize_team_code(away)
    home = normalize_team_code(home)

    away_feat = build_feature_row(away, home, is_home=0, week=WEEK_TO_PREDICT)
    home_feat = build_feature_row(home, away, is_home=1, week=WEEK_TO_PREDICT)

    X_away = pd.DataFrame([away_feat])[FEATURES].astype(float)
    X_home = pd.DataFrame([home_feat])[FEATURES].astype(float)

    away_drives = float(drives_model.predict(X_away)[0])
    away_ppd    = float(ppd_model.predict(X_away)[0])
    away_pts    = away_drives * away_ppd

    home_drives = float(drives_model.predict(X_home)[0])
    home_ppd    = float(ppd_model.predict(X_home)[0])
    home_pts    = home_drives * home_ppd

    pred_rows.append({
        "away_team": away,
        "away_predicted_drives": round(away_drives, 2),
        "away_predicted_ppd": round(away_ppd, 3),
        "away_predicted_points": round(away_pts, 2),

        "home_team": home,
        "home_predicted_drives": round(home_drives, 2),
        "home_predicted_ppd": round(home_ppd, 3),
        "home_predicted_points": round(home_pts, 2),

        "predicted_final_score": f"{home} {int(round(home_pts))} - {away} {int(round(away_pts))}"
    })

pred_df = pd.DataFrame(pred_rows)

print(f"\nPREDICTED SCORES (WEEK {WEEK_TO_PREDICT}) — drives × ppd")
display(pred_df)

# =========================
# SAVE TO DESKTOP
# =========================
out_path = os.path.join(get_desktop_path(), OUT_BASENAME)
pred_df.to_csv(out_path, index=False)
print("\nSaved predictions to:", out_path)

```

    Rows after dropna:
      Train: 268
      Val  : 42
    Validation weeks: 15 to 16
    
    Drives model: XGBoost
    PPD model   : XGBoost
    
    Validation performance:
      Drives RMSE: 1.9  MAE: 1.52
      PPD RMSE   : 1.31  MAE: 0.94
      POINTS (drives*ppd) RMSE: 11.35  MAE: 8.7
    
    DRIVES MODEL IMPORTANCE (top 30)
     1. home_opp_def_suppression                      0.066714
     2. yards_per_play_roll5                          0.063990
     3. def_sack_rate_forced_roll5                    0.055162
     4. rush_attempts_roll5                           0.054496
     5. points_scored_roll5                           0.053996
     6. def_points_per_drive_allowed_roll5            0.049948
     7. pace_efficiency                               0.048811
     8. turnover_pressure                             0.047738
     9. def_yards_per_play_allowed_roll5              0.046944
    10. first_downs_roll5                             0.045902
    11. pass_attempts_roll5                           0.045667
    12. off_ppd_vs_opp_def_ppd                        0.044619
    13. home_ppd_boost                                0.042384
    14. def_turnovers_forced_per_drive_roll5          0.041867
    15. drives_roll5                                  0.041430
    16. sack_rate_roll5                               0.041083
    17. turnovers_per_drive_roll5                     0.039342
    18. sack_pressure                                 0.036893
    19. plays_roll5                                   0.032933
    20. off_ypp_vs_opp_def_ypp                        0.032483
    21. points_per_drive_roll5                        0.031639
    22. wind_mph                                      0.028929
    23. is_home                                       0.007031
    
    PPD MODEL IMPORTANCE (top 30)
     1. off_ypp_vs_opp_def_ypp                        0.080327
     2. turnover_pressure                             0.061173
     3. home_ppd_boost                                0.057010
     4. pass_attempts_roll5                           0.055485
     5. sack_pressure                                 0.054351
     6. off_ppd_vs_opp_def_ppd                        0.052716
     7. pace_efficiency                               0.048845
     8. def_points_per_drive_allowed_roll5            0.047395
     9. def_turnovers_forced_per_drive_roll5          0.046008
    10. drives_roll5                                  0.044597
    11. turnovers_per_drive_roll5                     0.043549
    12. rush_attempts_roll5                           0.042900
    13. yards_per_play_roll5                          0.041972
    14. points_per_drive_roll5                        0.040177
    15. plays_roll5                                   0.039573
    16. sack_rate_roll5                               0.037877
    17. home_opp_def_suppression                      0.037059
    18. points_scored_roll5                           0.034154
    19. def_yards_per_play_allowed_roll5              0.033917
    20. wind_mph                                      0.033505
    21. first_downs_roll5                             0.033320
    22. def_sack_rate_forced_roll5                    0.026284
    23. is_home                                       0.007805
    
    PREDICTED SCORES (WEEK 17) — drives × ppd
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>away_team</th>
      <th>away_predicted_drives</th>
      <th>away_predicted_ppd</th>
      <th>away_predicted_points</th>
      <th>home_team</th>
      <th>home_predicted_drives</th>
      <th>home_predicted_ppd</th>
      <th>home_predicted_points</th>
      <th>predicted_final_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAL</td>
      <td>6.73</td>
      <td>3.967</td>
      <td>26.70</td>
      <td>WAS</td>
      <td>6.82</td>
      <td>3.389</td>
      <td>23.10</td>
      <td>WAS 23 - DAL 27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DET</td>
      <td>6.91</td>
      <td>3.101</td>
      <td>21.44</td>
      <td>MIN</td>
      <td>6.19</td>
      <td>2.380</td>
      <td>14.73</td>
      <td>MIN 15 - DET 21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DEN</td>
      <td>7.48</td>
      <td>2.805</td>
      <td>20.97</td>
      <td>KC</td>
      <td>7.06</td>
      <td>2.788</td>
      <td>19.69</td>
      <td>KC 20 - DEN 21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOU</td>
      <td>7.98</td>
      <td>3.741</td>
      <td>29.84</td>
      <td>LAC</td>
      <td>6.34</td>
      <td>3.608</td>
      <td>22.89</td>
      <td>LAC 23 - HOU 30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAL</td>
      <td>8.02</td>
      <td>2.999</td>
      <td>24.06</td>
      <td>GB</td>
      <td>8.04</td>
      <td>3.834</td>
      <td>30.83</td>
      <td>GB 31 - BAL 24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TB</td>
      <td>5.93</td>
      <td>3.073</td>
      <td>18.21</td>
      <td>MIA</td>
      <td>8.01</td>
      <td>5.150</td>
      <td>41.26</td>
      <td>MIA 41 - TB 18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NE</td>
      <td>7.64</td>
      <td>3.344</td>
      <td>25.53</td>
      <td>NYJ</td>
      <td>6.85</td>
      <td>2.854</td>
      <td>19.55</td>
      <td>NYJ 20 - NE 26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JAX</td>
      <td>8.52</td>
      <td>3.250</td>
      <td>27.69</td>
      <td>IND</td>
      <td>7.16</td>
      <td>3.533</td>
      <td>25.31</td>
      <td>IND 25 - JAX 28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ARI</td>
      <td>7.35</td>
      <td>2.963</td>
      <td>21.77</td>
      <td>CIN</td>
      <td>7.00</td>
      <td>2.416</td>
      <td>16.91</td>
      <td>CIN 17 - ARI 22</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PIT</td>
      <td>6.96</td>
      <td>3.652</td>
      <td>25.40</td>
      <td>CLE</td>
      <td>6.49</td>
      <td>2.434</td>
      <td>15.79</td>
      <td>CLE 16 - PIT 25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NO</td>
      <td>7.12</td>
      <td>2.907</td>
      <td>20.70</td>
      <td>TEN</td>
      <td>7.10</td>
      <td>3.651</td>
      <td>25.94</td>
      <td>TEN 26 - NO 21</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SEA</td>
      <td>7.73</td>
      <td>3.412</td>
      <td>26.37</td>
      <td>CAR</td>
      <td>6.97</td>
      <td>3.936</td>
      <td>27.43</td>
      <td>CAR 27 - SEA 26</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NYG</td>
      <td>7.57</td>
      <td>3.173</td>
      <td>24.04</td>
      <td>LV</td>
      <td>7.03</td>
      <td>4.350</td>
      <td>30.59</td>
      <td>LV 31 - NYG 24</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PHI</td>
      <td>6.48</td>
      <td>4.036</td>
      <td>26.16</td>
      <td>BUF</td>
      <td>6.56</td>
      <td>3.187</td>
      <td>20.91</td>
      <td>BUF 21 - PHI 26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHI</td>
      <td>6.75</td>
      <td>3.747</td>
      <td>25.28</td>
      <td>SF</td>
      <td>7.79</td>
      <td>3.368</td>
      <td>26.24</td>
      <td>SF 26 - CHI 25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LAR</td>
      <td>7.79</td>
      <td>3.420</td>
      <td>26.64</td>
      <td>ATL</td>
      <td>6.86</td>
      <td>4.376</td>
      <td>30.03</td>
      <td>ATL 30 - LAR 27</td>
    </tr>
  </tbody>
</table>
</div>


    
    Saved predictions to: C:\Users\eweis\Desktop\nfl_week_17_predictions.csv
    


```python
# Create a clean summary table
score_summary = pred_df[[
    "away_team",
    "home_team",
    "away_predicted_points",
    "home_predicted_points"
]].copy()

# Create a readable score string
score_summary["predicted_score"] = (
    score_summary["home_team"] + " " +
    score_summary["home_predicted_points"].round(0).astype(int).astype(str) +
    " – " +
    score_summary["away_team"] + " " +
    score_summary["away_predicted_points"].round(0).astype(int).astype(str)
)

# Keep only the columns you want
score_summary = score_summary[[
    "away_team",
    "home_team",
    "predicted_score"
]]

score_summary

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>away_team</th>
      <th>home_team</th>
      <th>predicted_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAL</td>
      <td>WAS</td>
      <td>WAS 23 – DAL 27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DET</td>
      <td>MIN</td>
      <td>MIN 15 – DET 21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DEN</td>
      <td>KC</td>
      <td>KC 20 – DEN 21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOU</td>
      <td>LAC</td>
      <td>LAC 23 – HOU 30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BAL</td>
      <td>GB</td>
      <td>GB 31 – BAL 24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TB</td>
      <td>MIA</td>
      <td>MIA 41 – TB 18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NE</td>
      <td>NYJ</td>
      <td>NYJ 20 – NE 26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JAX</td>
      <td>IND</td>
      <td>IND 25 – JAX 28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ARI</td>
      <td>CIN</td>
      <td>CIN 17 – ARI 22</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PIT</td>
      <td>CLE</td>
      <td>CLE 16 – PIT 25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NO</td>
      <td>TEN</td>
      <td>TEN 26 – NO 21</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SEA</td>
      <td>CAR</td>
      <td>CAR 27 – SEA 26</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NYG</td>
      <td>LV</td>
      <td>LV 31 – NYG 24</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PHI</td>
      <td>BUF</td>
      <td>BUF 21 – PHI 26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHI</td>
      <td>SF</td>
      <td>SF 26 – CHI 25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LAR</td>
      <td>ATL</td>
      <td>ATL 30 – LAR 27</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
