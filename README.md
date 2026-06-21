# Counterfactual Explainable AI Framework for Energy-Efficient Building Design Optimization

Interactive Streamlit dashboard for the UCI Energy Efficiency dataset (Tsanas & Xifara, 2012).
Predicts Heating Load (Y1) and Cooling Load (Y2) from 8 building shape features using
five regression models, with SHAP-based explainability.

## Project structure

```
energy_app/
├── app.py                 # Streamlit application
├── ENB2012_data.csv       # Dataset (bundled, no upload needed)
├── requirements.txt       # Pinned dependencies
├── .gitignore
└── README.md
```

## Features

- **Models:** Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **EDA tab:** correlation heatmap, target/feature distributions
- **Model performance tab:** R² / MAE / RMSE comparison table + chart, best-model highlight
- **SHAP tab:** global summary plot, feature-importance bar chart, local waterfall plot for any test instance
- **Predict tab:** manual sliders to simulate a building design and get a live prediction

All training and SHAP computation is cached (`@st.cache_data` / `@st.cache_resource`),
so the app stays responsive after the first run for a given target/test-size/seed combination.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to GitHub + Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Energy Efficiency XAI dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   Make sure `ENB2012_data.csv` is committed — it is not in `.gitignore`.

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub, click "New app"
   - Select your repo, branch `main`, main file path `app.py`
   - Click "Deploy"

   Streamlit Cloud will install everything from `requirements.txt` automatically.

3. **If the app fails to find the CSV**
   The app loads the CSV via a path relative to `app.py`
   (`os.path.join(os.path.dirname(__file__), "ENB2012_data.csv")`), so it works
   regardless of the repo name as long as the CSV sits in the same folder as `app.py`.

## Notes on scope vs. the original research pipeline

This app implements Steps 1–8 of the original pipeline (data loading → preprocessing →
EDA → multi-model training → evaluation → SHAP global/local explanation) as an interactive
dashboard. Counterfactual generation (DiCE / NSGA-II, Steps 9–12) was intentionally left
out of this deployment because:
- DiCE has heavier/less Streamlit-Cloud-friendly dependencies and longer per-request runtimes,
  which can cause timeouts on free hosting tiers.
- It's best run as a separate notebook (`04_Counterfactual.ipynb`) or added as a later
  Streamlit tab once the core dashboard is confirmed working in deployment.

Let me know if you'd like a counterfactual tab added as a follow-up — it can reuse the
trained models and cached data already set up here.

## Dataset citation

A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential
buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012.
UCI Machine Learning Repository.
