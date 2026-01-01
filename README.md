# NFL_2025 — Baseline Machine Learning Model for NFL Score Prediction

This repository contains an **educational, baseline machine learning dataset and modeling example** for predicting NFL game scores using team-level offensive and defensive statistics.

The project was created to support learning, comparison, and experimentation with supervised machine learning methods applied to real-world sports data. It is intentionally framed as a **baseline**, not a production system or optimized betting model.

---

## Repository Contents

This repository includes:

- **`NFL_ML_thru_12.22.25.xlsx`**  
  A structured dataset of NFL regular-season games through December 22, 2025.

- **`NFL_2025_Week17_Predictions_Offense_and_Defense(GH).ipynb`**  
  A Jupyter notebook demonstrating how the dataset can be used to train and evaluate a gradient-based regression model for score prediction.

- **`README.md`**  
  Documentation describing the dataset, modeling approach, and intended use.

---

## Dataset Overview

### Structure
- **Unit of analysis:** One team in one game  
- **Rows:** Each row represents a single team’s performance in a single NFL game  
- **Games appear twice** (once per team), allowing team-level modeling  
- **Time span:** Regular-season games through December 22, 2025  
- **Target variable:** `points_scored`

This structure allows models to learn team-level scoring patterns while incorporating opponent context.

---

## Key Variables

### Identifiers & Context
- `team` — Team abbreviation  
- `opp_team` — Opponent team  
- `nfl_week` — NFL week number  
- `game_number` — Sequential game count for the team  
- `is_home` — Home indicator (1 = home, 0 = away)  
- `wind_mph`, `opp_wind_mph` — Approximate wind speed at game location  

### Offensive Performance (Team)
Includes variables such as:
- Points scored (target)
- First downs
- Rushing attempts, yards, and touchdowns
- Passing attempts, completions, yards, and touchdowns
- Interceptions, sacks, turnovers
- Penalties and penalty yards

### Defensive / Opponent Context
Opponent statistics mirror team variables and are prefixed with `opp_`, allowing the model to learn matchup effects.

---

## Modeling Approach

The accompanying notebook demonstrates a **gradient-based regression approach** applied to tabular sports data. The focus is on:

- Feature utilization rather than heavy optimization
- Interpretability and reproducibility
- Providing a reasonable comparison baseline for student projects

Users are encouraged to:
- Engineer rolling averages or lagged features
- Explore alternative algorithms
- Experiment with different validation strategies

---

## Intended Use

This repository is intended for:
- Graduate or advanced undergraduate coursework
- Independent learning and experimentation
- Baseline comparison for sports analytics projects

It is **not** intended as:
- A betting or gambling tool
- A production-grade prediction system
- A claim of state-of-the-art performance

---

## Disclaimer

This dataset and code are provided **for educational and research comparison purposes only**. No guarantees are made regarding predictive accuracy or real-world applicability.

---

## Author

**Eric B. Weiser, Ph.D.**  
Professor of Psychology  

LinkedIn:  
https://www.linkedin.com/in/eric-b-weiser-ph-d-9b883b23/

