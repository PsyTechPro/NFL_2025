# NFL_2025 — Baseline Machine Learning Model for NFL Score Prediction

This repository contains a baseline machine learning (ML) dataset and modeling example for predicting NFL game scores using team-level offensive and defensive stats.

The project was created to support learning, comparison, and experimentation with supervised ML methods applied to real-world sports data. It is intentionally framed as a **baseline**, not a production system or optimized betting model.

---

## Repository Contents

This repository includes:

- **`NFL_ML_thru_12.22.25.xlsx`**  
  A structured dataset of NFL regular-season games through December 22, 2025 (i.e., each NFL team's first 15 games during the 2025 season, up thru Week 16).  

- **`NFL_2025_Week17_Predictions_Offense_and_Defense(GH).ipynb`**  
  A Jupyter notebook demonstrating how the dataset can be used to train and evaluate a gradient-based regression model for score predictions during Week 17 of the NFL season (Dec. 25-29, 2025).

- **`README.md`**  
  Documentation describing the dataset, modeling approach, and intended use.

---

## Dataset Overview

### Structure
- **Unit of analysis:** One team in one game  
- **Rows:** Each row represents a single team’s offensive and defensive performance in a single NFL game  
- **Time span:** Regular-season games through December 22, 2025  
- **Target variable:** `points_scored` during Week 17

This structure allows models to learn team-level scoring patterns while incorporating opponent context.

---

## Key Variables

### Identifiers & Context
- `team` — Team abbreviation  
- `opp_team` — Opponent team  
- `nfl_week` — NFL week number  
- `game_number` — Game number for the team  
- `is_home` — Home indicator (1 = home, 0 = away)  
- `wind_mph`, `opp_wind_mph` — Approximate wind speed at game location, if available (rare)   

### Offensive Performance (Team)
Includes variables such as:
- Points scored (target)
- First downs
- Rushing attempts, yards, and touchdowns
- Passing attempts, completions, yards, and touchdowns
- Interceptions, sacks, turnovers
- Penalties and penalty yards

### Defensive / Opponent Context
Opponent statistics mirror the opposing team's variables and are prefixed with `opp_`, allowing the model to learn matchup effects.

---

## Modeling Approach

The accompanying notebook demonstrates a gradient-based regression approach applied to tabular sports data. The focus is on:

- Feature utilization rather than heavy optimization
- Interpretability and reproducibility
- Providing a reasonable comparison baseline for student projects

Users are encouraged to:
- Engineer rolling averages or lagged features
- Explore alternative algorithms
- Experiment with different validation strategies

---

## Intended Use

**This repository is intended for educational purposes only!** It would be appropriate for graduate or undergraduate level coursework, as well as independent learning and experimentation. It would be ideal for sports analytic projects. 

It is **ABSOLUTELY NOT** intended as:
- A betting or gambling tool
- A production-grade prediction system
- A claim of state-of-the-art performance

---

## Disclaimer

This dataset and code are provided **for educational and research comparison purposes only**. No guarantees are made regarding predictive accuracy or real-world applicability.

---

## Author

**Eric B. Weiser, Ph.D.**  
 

If you wish to contact me, you may do so through my LinkedIn:  
https://www.linkedin.com/in/eric-b-weiser-ph-d-9b883b23/

## Attribution

If you use this dataset, code, or modeling approach in academic work, coursework, presentations, or derivative projects, please provide appropriate attribution by citing this repository or referencing the author (Eric B. Weiser).


