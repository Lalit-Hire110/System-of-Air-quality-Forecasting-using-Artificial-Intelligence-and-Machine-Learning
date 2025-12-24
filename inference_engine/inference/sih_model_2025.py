# app.py
# =========================
# SIH 2025 Air Quality Prototype (Streamlit) – Robust Live Scoring v2
# - CPCB-first targets, time-aware validation, multi-source covariates
# - Demo (Holdout) and Live Scoring with auto target detection and diagnostics
# - Robust CSV parsing: delimiter/encoding inference, header cleanup, numeric coercion
# =========================

import os
import io
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.expanduser(r"C:\Users\Lalit Hire\SIH 2025 Model")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRED_DIR   = os.path.join(BASE_DIR, "predictions")
EVAL_DIR   = os.path.join(BASE_DIR, "evaluation")
FEAT_DIR   = os.path.join(BASE_DIR, "features")

st.set_page_config(page_title="SIH 2025 Air Quality Prototype", layout="wide")

# -------------------------
# Helpers
# -------------------------
def rmse_compat(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_artifacts():
    # Features
    no2_features = pd.read_csv(os.path.join(FEAT_DIR, "no2_features.csv"))["feature"].tolist()
    o3_features  = pd.read_csv(os.path.join(FEAT_DIR, "o3_features.csv"))["feature"].tolist()
    feat_meta    = load_json(os.path.join(FEAT_DIR, "feature_metadata.json"))

    # Models + imputers
    no2_model   = load_pickle(os.path.join(MODELS_DIR, "no2_model.pkl"))
    no2_imputer = load_pickle(os.path.join(MODELS_DIR, "no2_imputer.pkl"))
    o3_model    = load_pickle(os.path.join(MODELS_DIR, "o3_model.pkl"))
    o3_imputer  = load_pickle(os.path.join(MODELS_DIR, "o3_imputer.pkl"))

    # Quantile models (optional)
    no2_qmods = {}
    o3_qmods  = {}
    no2_qpath = os.path.join(MODELS_DIR, "no2_quantile_models.pkl")
    o3_qpath  = os.path.join(MODELS_DIR, "o3_quantile_models.pkl")
    if os.path.exists(no2_qpath):
        no2_qmods = load_pickle(no2_qpath)
    if os.path.exists(o3_qpath):
        o3_qmods = load_pickle(o3_qpath)

    # Predictions + station metrics
    no2_hold = pd.read_csv(os.path.join(PRED_DIR, "no2_holdout_predictions.csv"))
    o3_hold  = pd.read_csv(os.path.join(PRED_DIR, "o3_holdout_predictions.csv"))
    no2_st   = pd.read_csv(os.path.join(EVAL_DIR, "no2_station_metrics.csv"))
    o3_st    = pd.read_csv(os.path.join(EVAL_DIR, "o3_station_metrics.csv"))
    eval_sum = load_json(os.path.join(EVAL_DIR, "evaluation_summary.json"))

    return {
        "no2_features": no2_features,
        "o3_features":  o3_features,
        "feat_meta":    feat_meta,
        "no2_model":    no2_model,
        "no2_imputer":  no2_imputer,
        "o3_model":     o3_model,
        "o3_imputer":   o3_imputer,
        "no2_qmods":    no2_qmods,
        "o3_qmods":     o3_qmods,
        "no2_hold":     no2_hold,
        "o3_hold":      o3_hold,
        "no2_station":  no2_st,
        "o3_station":   o3_st,
        "eval_sum":     eval_sum
    }

def sanitize_name(s: str) -> str:
    s2 = s.replace("(", "").replace(")", "").replace("/", "_per_")
    s2 = "_".join(s2.split())
    return s2

def build_resolver(cols):
    # Map original, stripped, and sanitized variants back to original column
    resolver = {}
    for c in cols:
        resolver[c] = c
        resolver[c.strip()] = c
        resolver[sanitize_name(c)] = c
    return resolver

def align_from_incoming(df_in: pd.DataFrame, required: list) -> pd.DataFrame:
    # Normalize incoming headers (strip only; keep original for resolver)
    df_norm = df_in.copy()
    df_norm.columns = [str(c).strip() for c in df_norm.columns]
    resolver = build_resolver(list(df_norm.columns))

    X = pd.DataFrame(index=df_norm.index)
    for f in required:
        if f in df_norm.columns:
            src = f
        else:
            san = sanitize_name(f)
            src = resolver.get(f, resolver.get(san, None))
        if src is not None and src in df_norm.columns:
            X[f] = df_norm[src]
        else:
            X[f] = np.nan
        # Coerce to numeric to avoid object->NaN later; commas handled
        if X[f].dtype == object:
            X[f] = X[f].astype(str).str.replace(",", "").str.strip()
        X[f] = pd.to_numeric(X[f], errors="coerce")

    return X[required]

def kpi_block(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = rmse_compat(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (μg/m³)", f"{mae:.2f}")
    c2.metric("RMSE (μg/m³)", f"{rmse:.2f}")
    c3.metric("R²", f"{r2:.3f}")

def line_chart_with_band(df, time_col="time_utc", y_true="y_true", y_pred="y_pred", q10="q10", q90="q90", title=""):
    import plotly.graph_objects as go
    fig = go.Figure()
    if y_true in df.columns:
        fig.add_trace(go.Scatter(x=df[time_col], y=df[y_true], mode="lines", name="Observed", line=dict(color="black")))
    if y_pred in df.columns:
        fig.add_trace(go.Scatter(x=df[time_col], y=df[y_pred], mode="lines", name="Predicted", line=dict(color="blue")))
    if q10 in df.columns and q90 in df.columns and df[q10].notna().any():
        fig.add_trace(go.Scatter(x=df[time_col], y=df[q90], mode="lines", name="q90", line=dict(color="lightblue"), showlegend=False))
        fig.add_trace(go.Scatter(x=df[time_col], y=df[q10], mode="lines", name="q10", line=dict(color="lightblue"), fill="tonexty", fillcolor="rgba(30,144,255,0.2)", showlegend=False))
    fig.update_layout(title=title, xaxis_title="Time (UTC)", yaxis_title="μg/m³", height=420)
    st.plotly_chart(fig, use_container_width=True)

def detect_best_target(cols, no2_feats, o3_feats):
    s = set([str(c).strip() for c in cols])
    i_no2 = len(s.intersection(set(no2_feats)))
    i_o3  = len(s.intersection(set(o3_feats)))
    if i_no2 == 0 and i_o3 == 0:
        return None, i_no2, i_o3
    return ("NO2" if i_no2 >= i_o3 else "O3"), i_no2, i_o3

def predict_quantiles_from_uploaded(inp_df, qmods, align_func):
    if not qmods:
        n = len(inp_df)
        return np.full(n, np.nan), np.full(n, np.nan)
    q10 = np.full(len(inp_df), np.nan)
    q90 = np.full(len(inp_df), np.nan)
    for a, art in qmods.items():
        feats_q = art["features"]
        imp_q   = art["imputer"]
        mdl_q   = art["model"]
        Xq = align_func(inp_df, feats_q)
        Xq_imp = pd.DataFrame(imp_q.transform(Xq), columns=feats_q, index=Xq.index)
        preds = mdl_q.predict(Xq_imp)
        if abs(a - 0.1) < 1e-6:
            q10 = preds
        elif abs(a - 0.9) < 1e-6:
            q90 = preds
    return q10, q90

def read_uploaded_csv(file):
    # Robust reader: handle BOM, auto delimiter, and semicolon fallback; drop unnamed index cols
    read_attempts = [
        dict(sep=None, engine="python", encoding="utf-8-sig"),
        dict(sep=",", encoding="utf-8-sig"),
        dict(sep=";", encoding="utf-8-sig")
    ]
    last_err = None
    for kw in read_attempts:
        try:
            df = pd.read_csv(file, **kw)
            # Drop unnamed/index columns introduced by Excel/exports
            drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            return df
        except Exception as e:
            last_err = e
            file.seek(0)
            continue
    raise last_err if last_err else RuntimeError("Failed to parse uploaded CSV")

# -------------------------
# App UI
# -------------------------
st.title("Smart India Hackathon 2025 – Air Quality Prototype")
st.caption("CPCB-first targets, time-aware validation, multi-source covariates (ERA5/MERRA-2/satellite/ground).")

art = load_artifacts()

mode = st.sidebar.radio("Mode", ["Demo (Holdout)", "Live Scoring"], index=0)

if mode == "Demo (Holdout)":
    target = st.sidebar.selectbox("Target", ["NO2", "O3"], index=0)

    hold_df = art["no2_hold"] if target == "NO2" else art["o3_hold"]
    station_col = "station_id" if "station_id" in hold_df.columns else None
    time_col = "time_utc" if "time_utc" in hold_df.columns else hold_df.columns[0]

    st.subheader(f"{target} – Holdout Demo")
    stations = ["All"] + sorted(hold_df[station_col].dropna().unique().tolist()) if station_col else ["All"]
    station_sel = st.selectbox("Station", stations, index=0)

    plot_df = hold_df.copy()
    if station_col and station_sel != "All":
        plot_df = plot_df[plot_df[station_col] == station_sel]

    if "y_true" in plot_df.columns:
        st.markdown("#### KPIs")
        kpi_block(plot_df["y_true"].astype(float), plot_df["y_pred"].astype(float))

    st.markdown("#### Time Series")
    line_chart_with_band(
        plot_df.sort_values(time_col),
        time_col=time_col, y_true="y_true", y_pred="y_pred",
        q10="q10", q90="q90",
        title=f"{target} predictions with q10–q90 uncertainty"
    )

    st.markdown("#### Per-Station Performance (Holdout)")
    st_df = art["no2_station"] if target == "NO2" else art["o3_station"]
    st.dataframe(st_df.sort_values("rmse").reset_index(drop=True))

    st.markdown("#### Model Card")
    card = art["eval_sum"]["model_cards"]["NO2" if target == "NO2" else "O3"]
    st.json(card)

else:
    st.subheader("Live Scoring")
    target_mode = st.selectbox("Target mode", ["Auto (detect)", "NO2", "O3"], index=0)

    up = st.file_uploader("Upload CSV (recommend holdout_*_live_input.csv for exact evaluation slice)", type=["csv"])
    if up is not None:
        # Robust parse
        try:
            inp = read_uploaded_csv(up)
        finally:
            up.seek(0)
        st.write("Uploaded columns:", list(inp.columns))
        st.write("Input preview:", inp.head(5))

        # Auto-detect target
        auto_choice, i_no2, i_o3 = detect_best_target(inp.columns, art["no2_features"], art["o3_features"])
        if target_mode == "Auto (detect)":
            if auto_choice is None:
                st.error("Could not detect target: no overlap with required feature sets. Check headers or delimiter.")
                st.stop()
            target = auto_choice
            st.info(f"Auto-detected target: {target} (overlap: NO2={i_no2}, O3={i_o3})")
        else:
            target = target_mode

        if target == "NO2":
            model   = art["no2_model"]
            imputer = art["no2_imputer"]
            feats   = art["no2_features"]
            qmods   = art["no2_qmods"]
        else:
            model   = art["o3_model"]
            imputer = art["o3_imputer"]
            feats   = art["o3_features"]
            qmods   = art["o3_qmods"]

        # Align and coerce
        X_aligned = align_from_incoming(inp, feats)

        # Coverage diagnostics
        per_feat_cov = X_aligned.notna().mean().sort_values(ascending=True)
        nonnull_frac = float(X_aligned.notna().mean().mean())
        st.caption(f"Non-null coverage across required features: {nonnull_frac*100:.1f}%")
        st.markdown("Top missing features:")
        st.write(per_feat_cov.head(15))

        if nonnull_frac < 0.5:
            st.warning("Low feature coverage detected. This usually means a delimiter/encoding/header mismatch. "
                       "Try saving CSV with comma delimiter and UTF-8 (no Excel re-export), or upload the generated holdout_*_live_input.csv directly.")

        # Impute and predict
        X_imp = pd.DataFrame(imputer.transform(X_aligned), columns=feats, index=X_aligned.index)
        yhat = model.predict(X_imp)
        q10, q90 = predict_quantiles_from_uploaded(inp, qmods, lambda d, f: align_from_incoming(d, f))

        # Meta echo
        meta_cols = [c for c in ["time_utc","timestamp_utc","station_id","station"] if c in inp.columns]
        pred_df = pd.DataFrame({"y_pred": yhat, "q10": q10, "q90": q90}, index=inp.index)
        out = pd.concat([inp[meta_cols].reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

        # Diversity check
        n_unique_rows = int(pd.DataFrame(X_imp).drop_duplicates().shape[0])
        st.caption(f"Unique feature rows after imputation: {n_unique_rows} of {len(X_imp)}")

        st.success(f"Inference complete for {len(out)} rows.")
        st.write(out.head(10))

        # Download results
        buff = io.StringIO()
        out.to_csv(buff, index=False)
        st.download_button("Download predictions CSV", buff.getvalue(),
                           file_name=f"{target}_live_predictions.csv", mime="text/csv")
