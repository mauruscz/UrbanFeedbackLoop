# 🏙️ The Urban Impact of AI: Modeling Feedback Loops in Next-Venue Recommendation

This repository contains the codebase and data pipeline for our simulation framework exploring the urban-scale effects of AI-driven venue recommendation systems. Our goal is to model how such systems reshape user mobility patterns and urban dynamics through behavioral feedback loops.

---

## 📥 Data Requirement

The notebook expects a file named `foursquare_complete_ny.csv` located in the `data/raw/` directory.

Due to distribution constraints, we do **not** include this file in the repository.  
Please manually download it by following these steps:

1. Visit the [Foursquare Dataset page](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).
2. Scroll to the **NYC Check-in Dataset** and click the download link:  
   [NYC Check-in Dataset (Google Drive)](https://drive.google.com/file/d/1VE67jfSN-GV6fkCfuyBEQJzk_qy-SfOk/view?usp=sharing)
3. Rename the downloaded file to:  
   ```
   foursquare_complete_ny.csv
   ```
4. Place it into the `data/raw/` folder.

Once added, you can run `data.ipynb` to begin preprocessing.

Additionally, the Foursquare taxonomy used for categorization has been taken from  
[ObservableHQ](https://observablehq.com/d/94b009d907d7c023) and stored as:

```
data/raw/tax.txt
```

This file is used during preprocessing to map POI categories.

---

## 📁 Repository Overview


### `data.ipynb`
Notebook for cleaning and transforming raw Foursquare check-in data into a format digestible by our simulation pipeline.

---

### `model/`

- **`simulation.py`**  
  Core simulation engine handling user-agent interactions, urban dynamics, and environment updates.

- **`agents.py`**  
  Agent behavior logic — defines how agents decide where to go, whether they follow recommendations, and how they evolve.

- **`recommender.py`**  
  Implementation of recommendation algorithms used to suggest venues to agents (e.g., popularity-based, collaborative, etc.).

---

### `main/`

- **`main.py`**  
  Script to run a single simulation experiment. Supports a range of configurable arguments:
  
  ```bash
  python main.py \
    -tw <training_window_days> \
    -k <epoch_length_days> \
    -sd <simulation_days> \
    -v <version_tag> \
    -s <random_seed> \
    -c <city> \
    -rb <behavioral_model> \
    -rs <recommender_system>
  ```

  **Key Arguments:**
  - `-tw` (int): Training window size (days)
  - `-k` (int): Epoch length (days)
  - `-sd` (int): Total simulation length (days)
  - `-v` (str): Version label (for experiment tracking)
  - `-s` (int): Random seed (use 500–504 to replicate paper results)
  - `-c` (str): Target city (e.g., `"nyc"`)
  - `-rb` (str): Behavioral model **(required)**
  - `-rs` (str): Recommender system **(required)**

  **Simulation Output:**  
  Results are saved to:  
  ```
  data/processed/experiments/full/seed/
  ```

  Example file:  
  `city_nyc__train_210__step_7__max_104__topK_20__alg_CPop__recProb_0.0/user_histories.csv`

  > When the `step` column is non-null, the entry comes from the **simulation phase**, not the training data.

---

### `notebook/`

- **`networks.py`**  
  Generates all co-location networks and structural data needed for post-simulation analysis.

  ```bash
  python networks.py \
    -rs <recommender_system> \
    -s <seed>
  ```

- **`figures.py`**  
  Computes and stores structural network properties (e.g., degree distribution, clustering, centrality) into:
  ```
  property_<city>/
  ```

  One `.txt` file per property.

  ```bash
  python figures.py \
    -rs <recommender_system> \
    -s <seed>
  ```

- **Automation Scripts:**
  - `run_batch_networks_calc.sh`  
  - `run_batch_figures_calc.sh`  
    > These automate batch execution for many combinations of parameters. Edit the `.sh` files to define your configuration.

---

### `run_tmux.sh`

Launches large-scale simulations across all combinations of recommenders, seeds, and acceptance probabilities using parallel `tmux` sessions.

```bash
bash run_tmux.sh
```

⚠️ **Warning**: This is computationally intensive. Use only if you have the resources to run multiple experiments concurrently.

---

## 🔁 Reproducibility

To reproduce the experiments from our paper, run simulations using seeds from `500` to `504`. These are the exact seeds used in our published results.

---

## 📊 Output Format

Simulation outputs include detailed `.csv` files describing user trajectories and decisions. You’ll find them in:

```
data/processed/experiments/full/seed/
```

Use the analysis scripts in `notebook/` to extract network-level insights and generate publication-ready figures.

