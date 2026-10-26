"""Dry Bean Classifier Lab — Streamlit front-end.

Upload the held-out test CSV, pick one of the five pre-trained models and see how it
performs. Models are loaded from the model/ folder; nothing is retrained here.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
    .bean-hero {padding: 1.1rem 1.4rem; border-radius: 14px;
                background: linear-gradient(120deg, #1f5f46 0%, #2e7d5b 55%, #8fbf5a 100%);
                color: #fff; margin-bottom: 1rem;}
    .bean-hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 .25rem 0;}
    .bean-hero p {margin: 0; opacity: .92;}
    div[data-testid="stMetric"] {background: rgba(46,125,91,.08); border: 1px solid rgba(46,125,91,.25);
                                 border-radius: 12px; padding: .6rem .8rem;}
    </style>
    <div class="bean-hero">
      <h1>🫘 Dry Bean Classifier Lab</h1>
      <p>Seven varieties of dry bean, sixteen shape measurements, five classic ML models.
         Upload test data, choose a model, and inspect how well it separates the beans.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading helpers
@st.cache_resource(show_spinner="Loading trained models…")
def load_bundle():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in info["model_files"].items()}
    return info, models


def weighted_ovr_auc(y_true, proba, classes):
    """One-vs-rest ROC AUC weighted by class support.

    Identical to sklearn's roc_auc_score(..., multi_class="ovr", average="weighted") when every
    class is present; if an uploaded file lacks some classes, it averages over the classes that
    are present instead of failing.
    """
    y_true = np.asarray(y_true)
    present = [c for c in classes if (y_true == c).any()]
    if len(present) < 2:
        return np.nan
    if len(present) == len(classes):
        return roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)
    scores, weights = [], []
    for c in present:
        idx = list(classes).index(c)
        scores.append(roc_auc_score((y_true == c).astype(int), proba[:, idx]))
        weights.append((y_true == c).sum())
    return float(np.average(scores, weights=weights))


def score_model(model, X, y):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": weighted_ovr_auc(y, proba, model.classes_),
        "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


try:
    info, models = load_bundle()
except Exception as err:  # pragma: no cover - shown in the UI
    st.error(f"Could not load the saved models from `model/`: {err}")
    st.stop()

FEATURES = info["feature_columns"]
TARGET = info["target"]
CLASSES = info["classes"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help=f"Must contain the 16 feature columns and the '{TARGET}' column.")
    if SAMPLE_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch")
        use_sample = st.checkbox("No file? Use the bundled test_data.csv", value=uploaded is None)
    else:
        use_sample = False

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()),
                              index=list(models.keys()).index("Support Vector Machine"))
    st.caption("Models were trained once in the notebook and are only *loaded* here — "
               "no retraining happens on the uploaded data.")

    st.divider()
    st.caption(f"Train rows: {info['train_rows']:,} · Test rows: {info['test_rows']:,} · "
               f"scikit-learn {info['sklearn_version']}")

# ---------------------------------------------------------------- read + validate data
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("👈 Upload a test CSV (or tick *Use the bundled test_data.csv*) to begin.")
    with st.expander("Expected columns"):
        st.write(FEATURES + [TARGET])
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The file is missing these required columns: {missing}")
    st.stop()

data = data[FEATURES + [TARGET]].dropna()
data[TARGET] = data[TARGET].astype(str).str.strip().str.upper()
unknown = sorted(set(data[TARGET]) - set(CLASSES))
if unknown:
    st.warning(f"Ignoring {int(data[TARGET].isin(unknown).sum())} rows with unknown labels: {unknown}")
    data = data[~data[TARGET].isin(unknown)]
if data.empty:
    st.error("No usable rows left after validation.")
    st.stop()

X_eval, y_eval = data[FEATURES], data[TARGET]
st.success(f"Loaded **{len(data):,}** rows from *{source}* · {y_eval.nunique()} bean classes present")

model = models[model_name]
metrics, y_pred = score_model(model, X_eval, y_eval)

tab_eval, tab_compare, tab_data = st.tabs(["📊 Selected model", "🏁 Compare all 5 models", "🔎 Data & predictions"])

# ---------------------------------------------------------------- tab 1: one model
with tab_eval:
    st.subheader(f"{model_name}")
    params = info["best_params"].get(model_name) or {}
    if params:
        st.caption("Tuned hyper-parameters: " + ", ".join(f"`{k.replace('clf__', '')}={v}`" for k, v in params.items()))

    cols = st.columns(6)
    for col, key in zip(cols, METRIC_ORDER):
        val = metrics[key]
        col.metric(key, "N/A" if np.isnan(val) else f"{val:.4f}")
    st.caption("Precision, Recall, F1 and AUC (one-vs-rest) use the support-weighted average over the 7 classes.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Confusion matrix**")
        labels = [c for c in CLASSES if c in set(y_eval) | set(y_pred)]
        cm = confusion_matrix(y_eval, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax, linewidths=.5)
        ax.set_xlabel("Predicted bean")
        ax.set_ylabel("Actual bean")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
        overall_acc = report.pop("accuracy")
        report_df = pd.DataFrame(report).T
        acc_row = pd.DataFrame({"precision": [np.nan], "recall": [np.nan], "f1-score": [overall_acc],
                                "support": [len(y_eval)]}, index=["accuracy"])
        report_df = pd.concat([report_df.iloc[:-2], acc_row, report_df.iloc[-2:]])
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                             "f1-score": "{:.4f}", "support": "{:,}"}, na_rep=""),
                     width="stretch", height=420)

# ---------------------------------------------------------------- tab 2: all models
with tab_compare:
    st.subheader("All five models on this test data")
    rows = {}
    for name, mdl in models.items():
        rows[name] = score_model(mdl, X_eval, y_eval)[0]
    comp = pd.DataFrame(rows).T[METRIC_ORDER]
    comp.index.name = "ML Model Name"
    winner = comp["Accuracy"].idxmax()
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d7"),
                 width="stretch")
    st.markdown(f"🏆 **Best on this data:** {winner} "
                f"(accuracy {comp.loc[winner, 'Accuracy']:.4f}, MCC {comp.loc[winner, 'MCC']:.4f})")

    long = comp.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="viridis", ax=ax)
    ax.set_ylim(max(0.0, long["Score"].min() - 0.05), 1.0)
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("⬇️ Download comparison table (CSV)", comp.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# ---------------------------------------------------------------- tab 3: data
with tab_data:
    st.subheader("Uploaded data and predictions")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Class balance in uploaded file**")
        st.bar_chart(y_eval.value_counts())
    with right:
        out = data.copy()
        out[f"Predicted ({model_name})"] = y_pred
        out["Correct?"] = np.where(out[TARGET] == y_pred, "✅", "❌")
        st.dataframe(out.head(500), width="stretch", height=380)
        st.caption("Showing the first 500 rows.")
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                           mime="text/csv")
