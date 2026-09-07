# REIT Valuation Engine
### Clustering & Explainable AI for Mumbai Real Estate

An institutional-grade valuation and investment-intelligence pipeline for REIT
underwriting teams and portfolio managers assessing residential property
portfolios across the Mumbai Metropolitan Region (MMR). The engine segments
properties into micro-market tiers, predicts fair-market valuations, simulates
rental yields and REIT cap rates, and explains every prediction with SHAP/LIME.

---

## 1. Architecture

```
                          ┌───────────────────────┐
                          │  data/raw/*.csv        │
                          │  (71,938 MMR listings) │
                          └───────────┬────────────┘
                                      ▼
                     ┌────────────────────────────────┐
                     │  src/data_pipeline.py           │
                     │  • domain-bound filtering        │
                     │  • IQR + z-score outlier removal │
                     │  • imputation (defensive)         │
                     │  • macro-zone assignment          │
                     │  • amenity score, age bins         │
                     │  • locality target + freq encoding│
                     └───────────────┬──────────────────┘
                                      ▼
                     ┌────────────────────────────────┐
                     │  src/clustering.py               │
                     │  • StandardScaler + PCA           │
                     │  • K-Means (k=4-5, silhouette/DBI)│
                     │  • tier naming (Ultra-Luxury...)  │
                     └───────────────┬──────────────────┘
                                      ▼
                     ┌────────────────────────────────┐
                     │  src/valuation_model.py          │
                     │  • LightGBM + XGBoost regressors  │
                     │  • Optuna tuning, 5-fold CV        │
                     │  • log-price target, RMSE/MAE/     │
                     │    R²/MAPE on holdout               │
                     └───────────────┬──────────────────┘
                          ┌───────────┴───────────┐
                          ▼                        ▼
             ┌─────────────────────┐   ┌─────────────────────────┐
             │ src/reit_metrics.py │   │ src/explainability.py    │
             │ • NOI, cap rate      │   │ • SHAP (summary/waterfall│
             │ • price-to-rent      │   │   /dependence)           │
             │ • Monte Carlo         │   │ • LIME local explanations│
             │   portfolio return   │   └─────────────────────────┘
             └─────────────────────┘
                          │                        │
                          └───────────┬────────────┘
                                      ▼
                     ┌────────────────────────────────┐
                     │  app/app.py (Streamlit)          │
                     │  1. Market Explorer & Cluster Map │
                     │  2. Single Property Underwriter    │
                     │  3. REIT Portfolio Screener         │
                     └────────────────────────────────┘
```

## 2. Project Structure

```
reit-valuation-engine/
├── data/
│   ├── raw/                          # source CSV
│   └── processed/                    # cleaned + engineered parquet outputs
├── models/                           # serialized joblib artifacts (gitignored)
├── src/
│   ├── config.py                     # paths, constants, REIT assumptions
│   ├── zone_mapping.py               # locality -> macro micro-market zone
│   ├── data_pipeline.py              # cleaning, outliers, feature engineering
│   ├── clustering.py                 # PCA + K-Means market-tier segmentation
│   ├── valuation_model.py            # LightGBM/XGBoost + Optuna valuation model
│   ├── reit_metrics.py               # NOI, cap rate, portfolio MC simulation
│   └── explainability.py             # SHAP + LIME explanations
├── app/
│   └── app.py                        # Streamlit underwriting dashboard
├── notebooks/
│   └── exploratory_eda.ipynb         # end-to-end analytical walkthrough
├── tests/
│   └── test_pipeline.py              # pytest suite
├── requirements.txt
├── Makefile
├── render.yaml                       # Render.com web service definition
├── runtime.txt                       # pinned Python version for Render
└── README.md
```

## 3. Key Modeling Decisions

- **No target leakage**: `price_per_sqft` is a deterministic function of the
  target (`price / area`) and is therefore *excluded* from the valuation
  model's feature set. It is still used for market **clustering**, where it
  is a legitimate unsupervised segmentation signal, not a supervised leak.
- **Log-transformed target**: the model predicts `log1p(price)`; all reported
  regression metrics (RMSE, MAE, R², MAPE) are computed after inverting back
  to INR so they are directly interpretable.
- **Locality encoding**: 408 raw locality strings are encoded two ways —
  (a) frequency encoding, and (b) a 5-fold out-of-fold smoothed target
  encoder (`LocalityTargetEncoder`) that never lets a row see its own price
  baked into its own encoded feature.
- **Macro zones without a hand-built 400-row lookup**: `zone_mapping.py`
  keyword-matches the high-volume, well-known localities to one of five
  zones (South Mumbai, Western Suburbs, Central Suburbs, Navi Mumbai, Thane &
  Extended MMR), then assigns every remaining locality to its geographically
  nearest zone centroid via haversine distance — guaranteeing full coverage.
- **Amenity score is a proxy**: the source data has no explicit amenities
  list (clubhouse, pool, gym...). `amenity_score` is a documented composite
  of the closest available signals (furnishing, balcony count,
  bathroom-to-bedroom ratio, building height, construction age, property
  type) — treat it as a luxury-density proxy, not a ground-truth amenity count.
- **Breaking the underwriter's valuation/clustering circularity**: the
  valuation model conditions on `cluster_id`, and cluster assignment uses
  `price_per_sqft` — unknown for a not-yet-priced candidate property. The
  dashboard resolves this the way a human underwriter would: it uses the
  target locality's historical median ₹/sq.ft as a *preliminary* comparable
  to place the property in the correct cluster, then the valuation model
  produces the actual fair-value estimate conditioned on that cluster.
- **REIT yield assumptions**: gross rental yield, opex ratio, vacancy, and
  appreciation drift/volatility are institutional heuristics calibrated to
  typical MMR benchmarks (`src/config.py`) — suitable for relative
  screening/ranking across a portfolio, not a substitute for a bottom-up
  appraisal or a licensed valuation.

## 4. Setup & Usage

```bash
make install        # pip install -r requirements.txt
make pipeline        # data cleaning -> clustering -> model training
make test            # pytest suite
make app             # streamlit run app/app.py
```

Or step by step:

```bash
python -m src.data_pipeline      # -> data/processed/*.parquet, models/*encoder*.joblib
python -m src.clustering         # -> models/cluster_model.joblib, kmeans/pca/scaler
python -m src.valuation_model    # -> models/valuation_model.joblib (best of LGBM/XGB)
streamlit run app/app.py
```

`python -m src.valuation_model --trials 12` (or the `OPTUNA_TRIALS` env var)
lowers the Optuna search budget for faster iteration or constrained build
environments.

## 5. Deployment on Render.com 
🚀 **Live Deployment:** [reit-valuation-engine.onrender.com](https://reit-valuation-engine.onrender.com)
