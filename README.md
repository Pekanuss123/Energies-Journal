# Multi-Step Prosumer Energy Forecasting

Code repository accompanying the journal paper:

> **"Enhanced Estimation of PV Power Production and Consumption with Multi-Step
> Prediction in Smart Energy Grids"**  
> *Submitted to Energies (MDPI), 2026*

---

## Repository structure

```
Multi-Step/
├── Analysis.ipynb               # Unified notebook — run this end-to-end
├── requirements.txt             # Pinned Python dependencies
│
├── Helper_Functions/
│   ├── preprocessing.py         # Swedish prosumer data pipeline
│   ├── preprocessing_citylearn.py  # CityLearn 2022 data pipeline
│   ├── models.py                # Direct / Recursive / Hybrid LightGBM
│   ├── models_conference.py     # Single-step baselines (GP, CatBoost, LinearQR, …)
│   ├── evaluation.py            # Metrics (MAE, R², sMAPE, MQL_Σ, PICP, CFE, MPIR)
│   └── analysis.py              # §5.1–5.4 experiments (training-size sweep,
│                                #   aggregation scenarios, autocorrelation,
│                                #   feature-target correlation, persistence)
│
├── Dataset/
│   ├── clients/                 # ← Swedish prosumer CSVs go here (see below)
│   ├── weather/                 # ← Swedish weather CSVs go here
│   ├── ELAD_Data/               # ← CityLearn Building_*.csv + weather.csv go here
│   └── combined_datasets/       # Written by the pipeline automatically
│
├── Results/                     # Written by the notebook (forecast CSVs, metrics)
└── Images/                      # Pre-generated paper figures (Figs 7–8)
```

---

## Data availability

### Swedish prosumer dataset (private)

The 7 Uppsala and 5 Halmstad prosumer time series used in the paper are
**not included in this repository**.  They originate from a research
collaboration and contain potentially identifiable household energy data; their
redistribution is restricted.

Per the paper's Data Availability Statement, the data may be requested from the
corresponding author subject to a data-sharing agreement.

If you have obtained the files, place them as follows before running the
notebook:

| File pattern | Destination |
|---|---|
| `Uppsala_*.csv`, `Halmstad_*.csv` | `Dataset/clients/` |
| Weather CSV(s) | `Dataset/weather/` |

Expected columns in each prosumer CSV: `Date`, `Produced`, `Total Consumed`,
plus weather features (temperature, irradiance, etc.).

### CityLearn 2022 dataset (public)

The California building simulation dataset used in §5.6 is **publicly
available** from the CityLearn 2022 Energy Challenge:

- **Repository:** <https://github.com/intelligent-environments-lab/CityLearn>
- **Challenge page:** <https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge>
- **Direct download:** `citylearn/data/citylearn_challenge_2022_phase_1/`  
  (files: `Building_1.csv` … `Building_17.csv` and `weather.csv`)

Place the downloaded files in `Dataset/ELAD_Data/` before running Section 3c
of the notebook.

---

## How to run the notebook end-to-end

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or, using conda:

```bash
conda create -n multistep python=3.11
conda activate multistep
pip install -r requirements.txt
```

### 2. Prepare data

- Place Swedish prosumer files in `Dataset/clients/` and `Dataset/weather/`
  (skip if unavailable — CityLearn-only results are still reproducible).
- Place CityLearn files in `Dataset/ELAD_Data/`.

### 3. Open and run the notebook

```bash
jupyter notebook Analysis.ipynb
```

Run cells in order.  Section-by-section guide:

| Section | Action | Est. runtime |
|---|---|---|
| 1 — Environment setup | Imports only | < 1 s |
| 2 — Configuration | Set paths/params | < 1 s |
| 3 — Data loading | Merges raw CSVs; writes `combined_datasets/` | < 1 min |
| 4 — Preprocessing | StandardScaler + sliding windows | < 1 min |
| 5 — Multi-step forecast | 3 strategies × all prosumers, grid-search | **2–6 h** on CPU |
| 5b — Conference baselines | 5 single-step models × all prosumers | 30–60 min |
| 5c — Training-size sweep | §5.1 Figure 3 | 30–60 min |
| 5d — Aggregation scenarios | §5.2 Figure 4 | 15–30 min |
| 5e — Auto-correlation | §5.3 Figure 5 | < 1 min |
| 5f — Feature correlation | §5.4 Figure 6 | < 1 min |
| 5g — Persistence baseline | Table 4 benchmark | < 1 min |
| 6 — Evaluation metrics | Per-step MAE/sMAPE/MQL_Σ/PICP/CFE/MPIR | < 1 min |
| 7 — Visualisation | Reproduces paper error plots | < 1 min |

> **Tip:** Sections 5e, 5f, 6, and 7 can be run independently of the long
> training sections if you already have result CSVs in `Results/`.

### GPU acceleration

LightGBM and CatBoost both support GPU training.  To enable it, change
`device='cpu'` → `device='gpu'` in `Helper_Functions/models.py` and
`Helper_Functions/models_conference.py`.

---

## Reproducing specific paper tables and figures

| Paper element | Notebook section |
|---|---|
| Figure 3 — MQL_Σ vs training size | 5c |
| Figure 4 — Aggregation scenarios | 5d |
| Figure 5 — Auto-correlation | 5e |
| Figure 6 — Feature-target correlation | 5f |
| Table 3 — Single-step model comparison | 5b (output: `Results/Conference/table3_reproduction.csv`) |
| Table 4 — Persistence baseline | 5g (output: `Results/persistence_baseline_summary.csv`) |
| Table 5 — Multi-step model comparison | 6 (output: `Results/Metrics/summary_table.csv`) |
| Figures 7–8 — MAE/MPIR error plots | Pre-generated PNGs in `Images/`; regenerate via Section 7 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{...,
  title   = {Enhanced Estimation of PV Power Production and Consumption
             with Multi-Step Prediction in Smart Energy Grids},
  journal = {Energies},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT — see `LICENSE`.
