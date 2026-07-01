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

### GPU acceleration

LightGBM and CatBoost both support GPU training.  To enable it, change
`device='cpu'` → `device='gpu'` in `Helper_Functions/models.py` and
`Helper_Functions/models_conference.py`.

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
