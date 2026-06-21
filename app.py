"""
Counterfactual Explainable AI Framework for Energy-Efficient Building Design Optimization
--------------------------------------------------------------------------------------
Streamlit web app.

Dataset : ENB2012_data.csv (UCI Energy Efficiency Dataset, Tsanas & Xifara, 2012)
Features: X1-X8 (building shape parameters)
Targets : Y1 (Heating Load), Y2 (Cooling Load)

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to GitHub, then deploy on https://share.streamlit.io
    pointing at app.py. ENB2012_data.csv must sit in the same folder/repo.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import shap

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Efficiency XAI Dashboard",
    page_icon="🏢",
    layout="wide",
)

DATA_FILE = os.path.join(os.path.dirname(__file__), "ENB2012_data.csv")

FEATURE_NAMES = {
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
}
TARGET_NAMES = {"Y1": "Heating Load", "Y2": "Cooling Load"}


# ----------------------------------------------------------------------
# Caching: data + model training (so the app doesn't retrain every click)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    X = df[list(FEATURE_NAMES.keys())]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=random_state),
        "XGBoost": XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=random_state, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=random_state, verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500, random_state=random_state, verbose=0
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        # Linear models benefit from scaling; tree models are scale-invariant
        # but we keep behaviour consistent with the original pipeline design.
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        results[name] = {
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        }
        fitted[name] = model

    return {
        "models": fitted,
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


@st.cache_resource(show_spinner=False)
def compute_shap(_model, _X_background, _X_explain, model_name: str):
    """
    Returns a shap.Explanation-like object. Tree models use TreeExplainer
    (fast, exact). Linear Regression uses LinearExplainer.
    """
    if model_name == "Linear Regression":
        explainer = shap.LinearExplainer(_model, _X_background)
        shap_values = explainer(_X_explain)
    else:
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer(_X_explain)
    return explainer, shap_values


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

df = load_data()

if df is None:
    st.error(
        f"Could not find **ENB2012_data.csv** next to app.py.\n\n"
        f"Expected path: `{DATA_FILE}`.\n\n"
        f"Make sure the CSV is committed to the same GitHub repo as app.py."
    )
    st.stop()

target = st.sidebar.selectbox(
    "Target variable",
    options=["Y1", "Y2"],
    format_func=lambda x: f"{x} - {TARGET_NAMES[x]}",
)

test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: UCI Energy Efficiency Dataset (Tsanas & Xifara, 2012). "
    "768 samples, 8 building shape features, 2 targets (Heating/Cooling Load)."
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏢 Counterfactual Explainable AI Framework")
st.subheader("Energy-Efficient Building Design Optimization")
st.write(
    "An interactive dashboard for multi-model energy load prediction and "
    "SHAP-based explainability on the UCI Energy Efficiency dataset."
)

tab_overview, tab_eda, tab_models, tab_shap, tab_predict = st.tabs(
    ["📋 Overview", "🔍 EDA", "📈 Model Performance", "🧠 SHAP Explainability", "🎯 Predict"]
)

# ----------------------------------------------------------------------
# Tab: Overview
# ----------------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)
        st.markdown(
            "**Features (X1-X8):** " +
            ", ".join(f"`{k}` {v}" for k, v in FEATURE_NAMES.items())
        )
        st.markdown(
            "**Targets:** `Y1` Heating Load, `Y2` Cooling Load"
        )
    with col2:
        st.markdown("### Dataset Info")
        st.metric("Rows", df.shape[0])
        st.metric("Features", 8)
        st.metric("Missing values", int(df.isnull().sum().sum()))
        st.metric("Duplicate rows", int(df.duplicated().sum()))

# ----------------------------------------------------------------------
# Tab: EDA
# ----------------------------------------------------------------------
with tab_eda:
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Target Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.hist(df["Y1"], bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title("Heating Load (Y1)")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.hist(df["Y2"], bins=30, color="#DD8452", edgecolor="white")
        ax.set_title("Cooling Load (Y2)")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Feature Distributions")
    feat = st.selectbox("Choose a feature", list(FEATURE_NAMES.keys()),
                         format_func=lambda x: f"{x} - {FEATURE_NAMES[x]}")
    fig, ax = plt.subplots()
    ax.hist(df[feat], bins=30, color="#55A868", edgecolor="white")
    ax.set_title(f"{feat} - {FEATURE_NAMES[feat]}")
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------
# Train models (cached on target/test_size/seed)
# ----------------------------------------------------------------------
with st.spinner("Training models (cached after first run)..."):
    bundle = train_models(df, target, test_size, int(random_state))

# ----------------------------------------------------------------------
# Tab: Model Performance
# ----------------------------------------------------------------------
with tab_models:
    st.markdown(f"### Model Comparison — Target: `{target}` ({TARGET_NAMES[target]})")
    res_df = pd.DataFrame(bundle["results"]).T.sort_values("R2", ascending=False)
    st.dataframe(
        res_df.style.format({"R2": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"})
        .highlight_max(subset=["R2"], color="#c6f6d5")
        .highlight_min(subset=["MAE", "RMSE"], color="#c6f6d5"),
        use_container_width=True,
    )

    best_model_name = res_df["R2"].idxmax()
    st.success(f"🏆 Best model by R²: **{best_model_name}** "
               f"(R² = {res_df.loc[best_model_name, 'R2']:.4f})")

    st.markdown("### R² Comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(res_df.index, res_df["R2"], color="#4C72B0")
    ax.set_ylabel("R²")
    ax.set_xticklabels(res_df.index, rotation=30, ha="right")
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------
# Tab: SHAP Explainability
# ----------------------------------------------------------------------
with tab_shap:
    st.markdown(f"### SHAP Explainability — Target: `{target}`")

    model_choice = st.selectbox(
        "Choose model to explain", list(bundle["models"].keys()),
        index=list(bundle["models"].keys()).index(
            res_df["R2"].idxmax()
        ) if "res_df" in dir() else 0,
    )

    model = bundle["models"][model_choice]
    X_test = bundle["X_test"]
    X_train = bundle["X_train"]
    X_test_scaled = bundle["X_test_scaled"]
    X_train_scaled = bundle["X_train_scaled"]

    # SHAP can be slow on large samples; cap for responsiveness
    sample_n = min(150, len(X_test))
    X_explain = X_test_scaled.iloc[:sample_n] if model_choice == "Linear Regression" \
        else X_test.iloc[:sample_n]
    X_bg = X_train_scaled if model_choice == "Linear Regression" else X_train

    with st.spinner(f"Computing SHAP values for {model_choice}..."):
        try:
            explainer, shap_values = compute_shap(model, X_bg, X_explain, model_choice)
        except Exception as e:
            st.error(f"SHAP computation failed for {model_choice}: {e}")
            shap_values = None

    if shap_values is not None:
        st.markdown("#### Global Explanation (Summary Plot)")
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_explain, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Feature Importance (Mean |SHAP value|)")
        fig = plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Local Explanation (Single Instance)")
        idx = st.slider("Test instance index", 0, sample_n - 1, 0)
        st.dataframe(X_test.iloc[[idx]], use_container_width=True)

        fig = plt.figure(figsize=(8, 4))
        shap.plots.waterfall(shap_values[idx], show=False)
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------
# Tab: Predict (manual what-if input)
# ----------------------------------------------------------------------
with tab_predict:
    st.markdown("### Try Your Own Building Design")
    st.write("Adjust the sliders to simulate a building design and predict its energy load.")

    pred_model_name = st.selectbox(
        "Model for prediction", list(bundle["models"].keys()), key="predict_model"
    )
    pred_model = bundle["models"][pred_model_name]

    cols = st.columns(4)
    user_input = {}
    for i, (feat, label) in enumerate(FEATURE_NAMES.items()):
        col = cols[i % 4]
        lo, hi = float(df[feat].min()), float(df[feat].max())
        default = float(df[feat].median())
        with col:
            if df[feat].nunique() <= 6:
                options = sorted(df[feat].unique().tolist())
                user_input[feat] = st.selectbox(f"{feat} - {label}", options,
                                                 index=options.index(default) if default in options else 0)
            else:
                user_input[feat] = st.slider(f"{feat} - {label}", lo, hi, default)

    input_df = pd.DataFrame([user_input])[list(FEATURE_NAMES.keys())]

    if pred_model_name == "Linear Regression":
        input_for_pred = pd.DataFrame(
            bundle["scaler"].transform(input_df), columns=input_df.columns
        )
    else:
        input_for_pred = input_df

    prediction = pred_model.predict(input_for_pred)[0]
    st.metric(f"Predicted {TARGET_NAMES[target]} ({target})", f"{prediction:.2f}")

st.markdown("---")
st.caption(
    "Built with Streamlit · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP. "
    "Dataset: A. Tsanas, A. Xifara (2012), UCI Machine Learning Repository."
)
