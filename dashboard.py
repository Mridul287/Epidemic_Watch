"""
EpidemicWatch Dashboard (Phase 8)
==================================
A lightweight demo dashboard tying together the whole EpidemicWatch pipeline:
federated-trained syndrome classifier + statistical outbreak detection.

Run locally (NOT inside a Colab cell -- Streamlit needs its own server):
    pip install streamlit plotly pandas numpy
    streamlit run dashboard.py

Optional, for the live "classify a symptom" box at the bottom:
    pip install torch transformers peft
(needs internet access on first run to download Bio_ClinicalBERT, plus your
trained Phase 3 adapter folder available locally)

Two independent data sources, either can be missing without breaking the app:
1. A "historical predictions" CSV exported from Phase 5 (day, hospital,
   predicted_category columns) -- powers the trend charts and status cards.
   If not provided, the app generates a small synthetic demo series so the
   UI is still fully explorable out of the box.
2. A Phase 3 LoRA adapter folder -- powers the live classification box.
   If not found, that section just shows a clear "not available" message
   instead of crashing the rest of the dashboard.
"""

import os
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="EpidemicWatch Dashboard", layout="wide")

# =============================================================================
# Config
# =============================================================================
CATEGORIES = ["Respiratory", "Gastrointestinal", "Febrile/Systemic",
              "Dermatological", "Musculoskeletal/Other"]
NUM_HOSPITALS = 3
TOTAL_DAYS = 90
BASELINE_WINDOW = 30
ADAPTER_DIR_DEFAULT = "phase3_baseline_outputs/adapter_augmented"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

STATUS_COLORS = {"Normal": "#2E7D32", "Elevated": "#F9A825", "Alert": "#C62828"}


# =============================================================================
# Outbreak detection (same logic validated in Phase 5, including the
# Poisson-variance-floor fix and empirically-tuned thresholds)
# =============================================================================
def _robust_baseline_sigma(baseline):
    mu = baseline.mean()
    empirical_sigma = baseline.std()
    poisson_floor = np.sqrt(max(mu, 0.5))
    return max(empirical_sigma, poisson_floor)


def cusum_detect(counts, baseline_days=BASELINE_WINDOW, k_sigma=0.5, h_sigma=8.0):
    baseline = counts[:baseline_days]
    mu = baseline.mean()
    sigma = _robust_baseline_sigma(baseline) + 1e-9
    k, h = k_sigma * sigma, h_sigma * sigma
    S, S_trace, alarm_day = 0.0, [], None
    for t, x in enumerate(counts):
        S = max(0.0, S + (x - mu - k))
        S_trace.append(S)
        if alarm_day is None and t >= baseline_days and S > h:
            alarm_day = t
    return alarm_day, np.array(S_trace)


def current_status(counts, baseline_days=BASELINE_WINDOW, recent_window=7):
    """Classify the most recent `recent_window` days as Normal / Elevated / Alert,
    using the same CUSUM logic but read at the series' current end point."""
    if len(counts) <= baseline_days:
        return "Normal"  # not enough history yet to judge
    alarm_day, S_trace = cusum_detect(counts, baseline_days=baseline_days)
    last_day = len(counts) - 1
    if alarm_day is not None and alarm_day <= last_day:
        return "Alert"
    # "Elevated": recent CUSUM statistic is building but hasn't crossed threshold yet
    baseline = counts[:baseline_days]
    sigma = _robust_baseline_sigma(baseline) + 1e-9
    h = 8.0 * sigma
    if S_trace[-1] > 0.4 * h:
        return "Elevated"
    return "Normal"


# =============================================================================
# Data loading (historical predictions, real if available, synthetic fallback)
# =============================================================================
def generate_synthetic_demo_data(seed=42):
    """Small self-contained fallback so the dashboard works with zero setup."""
    rng = np.random.default_rng(seed)
    rows = []
    outbreak_hospital, outbreak_category = 0, "Respiratory"
    outbreak_start, outbreak_end = 50, 65
    for h in range(NUM_HOSPITALS):
        for day in range(TOTAL_DAYS):
            for cat in CATEGORIES:
                base_rate = 0.15 if cat != "Musculoskeletal/Other" else 0.3
                n = rng.poisson(base_rate)
                if h == outbreak_hospital and cat == outbreak_category and outbreak_start <= day < outbreak_end:
                    n += rng.poisson(6)
                for _ in range(n):
                    rows.append({"hospital": h, "day": day, "predicted_category": cat})
    return pd.DataFrame(rows)


@st.cache_data
def load_predictions(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required = {"hospital", "day", "predicted_category"}
        if not required.issubset(df.columns):
            st.error(f"Uploaded CSV is missing required columns: {required - set(df.columns)}. "
                     f"Falling back to demo data.")
            return generate_synthetic_demo_data(), True
        return df, False
    return generate_synthetic_demo_data(), True


def build_daily_counts(df, hospital, category, total_days=TOTAL_DAYS):
    sub = df[(df["hospital"] == hospital) & (df["predicted_category"] == category)]
    counts = np.zeros(total_days)
    for day, n in sub["day"].value_counts().items():
        if 0 <= day < total_days:
            counts[int(day)] = n
    return counts


# =============================================================================
# Classifier (optional -- only used by the live text-input demo box)
# =============================================================================
@st.cache_resource
def load_classifier(adapter_dir):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
    except ImportError:
        return None  # torch/transformers/peft not installed -- classification box disables gracefully

    if not os.path.isdir(adapter_dir):
        return None
    label2id = {c: i for i, c in enumerate(CATEGORIES)}
    id2label = {i: c for c, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CATEGORIES), id2label=id2label, label2id=label2id,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "id2label": id2label}


def classify_text(clf, text):
    import torch
    enc = clf["tokenizer"](text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = clf["model"](**enc).logits
    pred_id = int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    return clf["id2label"][pred_id], dict(zip(clf["id2label"].values(), probs))


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("EpidemicWatch")
st.sidebar.caption("Federated outbreak-detection demo dashboard")

uploaded = st.sidebar.file_uploader(
    "Historical predictions CSV (from Phase 5)", type=["csv"],
    help="Columns required: hospital, day, predicted_category. "
         "If you don't have one handy, the dashboard uses synthetic demo data instead.",
)
predictions_df, using_demo_data = load_predictions(uploaded)
if using_demo_data:
    st.sidebar.info("Using synthetic demo data (no CSV uploaded). "
                     "Upload Phase 5's exported predictions for real results.")

selected_hospital = st.sidebar.selectbox("Hospital", options=list(range(NUM_HOSPITALS)),
                                          format_func=lambda h: f"Hospital {h}")
selected_category = st.sidebar.selectbox("Syndrome category", options=CATEGORIES)

adapter_dir = st.sidebar.text_input("Phase 3 adapter path (for live classification)",
                                     value=ADAPTER_DIR_DEFAULT)

st.sidebar.divider()
st.sidebar.caption(
    "This is a project demonstration, not a clinical decision-support tool. "
    "All data shown is either synthetic or a controlled research dataset."
)

# =============================================================================
# Header + status cards
# =============================================================================
st.title("EpidemicWatch — District Hospital Network")
st.caption("Federated learning outbreak surveillance across simulated district hospitals")

status_cols = st.columns(NUM_HOSPITALS)
hospital_statuses = {}
for h in range(NUM_HOSPITALS):
    # "current status" for a hospital = worst status across all its categories
    worst_status = "Normal"
    for cat in CATEGORIES:
        counts = build_daily_counts(predictions_df, h, cat)
        s = current_status(counts)
        if s == "Alert":
            worst_status = "Alert"
            break
        elif s == "Elevated" and worst_status != "Alert":
            worst_status = "Elevated"
    hospital_statuses[h] = worst_status
    with status_cols[h]:
        st.markdown(
            f"""<div style="padding:16px;border-radius:8px;background-color:{STATUS_COLORS[worst_status]}22;
                    border:2px solid {STATUS_COLORS[worst_status]};">
                <div style="font-size:14px;color:#666;">Hospital {h}</div>
                <div style="font-size:24px;font-weight:700;color:{STATUS_COLORS[worst_status]};">
                    {worst_status}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# =============================================================================
# Main trend chart for the selected hospital + category
# =============================================================================
st.subheader(f"Hospital {selected_hospital} — {selected_category} case trend")

counts = build_daily_counts(predictions_df, selected_hospital, selected_category)
alarm_day, _ = cusum_detect(counts)
days_axis = list(range(len(counts)))

fig = go.Figure()
fig.add_trace(go.Bar(x=days_axis, y=counts, name="Daily predicted cases", marker_color="#B0BEC5"))
fig.add_vline(x=BASELINE_WINDOW, line_dash="dot", line_color="gray",
              annotation_text="baseline calibration ends", annotation_position="top")
if alarm_day is not None:
    fig.add_vline(x=alarm_day, line_dash="dash", line_color="#C62828",
                  annotation_text=f"CUSUM alarm (day {alarm_day})", annotation_position="top")
fig.update_layout(xaxis_title="Day", yaxis_title="Predicted cases",
                   height=400, margin=dict(t=30, b=30))
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Network-wide status table
# =============================================================================
st.subheader("Network status — all hospitals")

table_rows = []
for h in range(NUM_HOSPITALS):
    row = {"Hospital": h}
    for cat in CATEGORIES:
        c = build_daily_counts(predictions_df, h, cat)
        row[cat] = current_status(c)
    row["Overall"] = hospital_statuses[h]
    table_rows.append(row)

status_df = pd.DataFrame(table_rows).set_index("Hospital")


def _style_status(val):
    color = STATUS_COLORS.get(val, "#000")
    return f"color: {color}; font-weight: 600;"


st.dataframe(status_df.style.map(_style_status), use_container_width=True)

st.divider()

# =============================================================================
# Live classification demo
# =============================================================================
st.subheader("Try the classifier")
st.caption("Type a symptom description and see the live model's predicted syndrome category.")

clf = load_classifier(adapter_dir)
if clf is None:
    st.warning(
        f"Live classification is unavailable -- either `torch`/`transformers`/`peft` "
        f"aren't installed, or no trained adapter was found at '{adapter_dir}'. "
        f"The rest of the dashboard still works normally. Install the ML packages and/or "
        f"point the sidebar path at your Phase 3 `adapter_augmented` folder to enable this."
    )
else:
    sample_text = st.text_input("Symptom description",
                                 placeholder="e.g. persistent cough, shortness of breath, mild fever")
    if st.button("Classify") and sample_text.strip():
        predicted_category, probs = classify_text(clf, sample_text)
        st.success(f"Predicted category: **{predicted_category}**")
        prob_df = pd.DataFrame({"category": list(probs.keys()), "probability": list(probs.values())})
        prob_df = prob_df.sort_values("probability", ascending=False)
        st.bar_chart(prob_df.set_index("category"))

st.divider()
st.caption(
    "EpidemicWatch — Federated Learning SLM for Early Disease Outbreak Detection. "
    "Detection thresholds (CUSUM, Poisson-floored variance) tuned and validated in Phase 5."
)