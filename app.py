import os
import warnings

import altair as alt
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")  # scikit-learn version notices when unpickling
st.set_page_config(page_title="Battery Health Predictor", page_icon="🔋", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(ROOT, "battery_health_project")
MODEL_DIR = os.path.join(PROJ, "models")
PLOT_DIR = os.path.join(PROJ, "plots")
DATA_DIR = os.path.join(ROOT, "data")

FEATURES = {
    "cycle": ("Cycle number", "Charge/discharge cycle index of the cell", "%d"),
    "charge_ah": ("Charge capacity (Ah)", "Capacity measured during the charge step", "%.4f"),
    "charge_ah_reported": ("Reported charge capacity (Ah)", "Capacity value logged by the cycler", "%.3f"),
    "v_max": ("Max voltage (V)", "Highest voltage reached in the cycle", "%.4f"),
    "v_min": ("Min voltage (V)", "Lowest voltage reached in the cycle", "%.4f"),
    "duration_s": ("Cycle duration (s)", "Length of the full cycle", "%.1f"),
}
FEAT = list(FEATURES)

# Settings are the exact ones used in evprediction.ipynb.
MODELS = {
    "Random Forest": {
        "file": "random_forest.pkl", "plot": "random_forest_prediction.png", "scaled": False,
        "family": "Tree ensemble · bagging",
        "summary": "Averages 300 decision trees, each trained on a random sample of cycles and features.",
        "how": "Every tree learns threshold rules such as *charge capacity below 1.80 Ah → lower SOH*. "
               "Because each tree sees a different bootstrap sample and a random subset of features at every split, "
               "their individual errors partly cancel when the predictions are averaged.",
        "strengths": ["Best accuracy in this project", "Follows the non-linear capacity fade",
                      "Works on raw values, no scaling", "Provides feature importance"],
        "limits": ["Cannot predict SOH outside the range seen in training", "Largest file (~55 MB), slowest to load"],
        "params": {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2,
                   "max_features": "sqrt", "random_state": 42},
    },
    "XGBoost": {
        "file": "xgboost.pkl", "plot": "xgboost_prediction.png", "scaled": False,
        "family": "Tree ensemble · gradient boosting",
        "summary": "Builds 500 small trees in sequence, each one correcting the errors left by the previous trees.",
        "how": "Gradient boosting fits each new tree to the residuals of the current ensemble, scaled by a learning "
               "rate of 0.03. Row and column subsampling (80%) and XGBoost's built-in regularisation limit overfitting.",
        "strengths": ["Second-best accuracy", "Captures interactions between features", "Fast predictions, small file"],
        "limits": ["More hyperparameters to tune", "The saved file needs a matching XGBoost version"],
        "params": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.8,
                   "colsample_bytree": 0.8, "objective": "reg:squarederror", "random_state": 42},
    },
    "SVR": {
        "file": "svr.pkl", "plot": "svr_prediction.png", "scaled": True,
        "family": "Kernel method",
        "summary": "Support Vector Regression with an RBF kernel, trained on standardised inputs.",
        "how": "The RBF kernel lets SVR fit a smooth curved function. It ignores errors smaller than epsilon (0.1 pp) "
               "and only the training points outside that band, the support vectors, shape the model. "
               "C = 100 sets how strongly larger errors are penalised.",
        "strengths": ["Models smooth non-linear trends", "Robust to small measurement noise", "Compact file"],
        "limits": ["Needs the StandardScaler from training", "Training slows down sharply on large datasets",
                   "Sensitive to C and gamma"],
        "params": {"kernel": "rbf", "C": 100, "gamma": "scale", "epsilon": 0.1},
    },
    "Linear Regression": {
        "file": "linear_regression.pkl", "plot": "linear_regression_prediction.png", "scaled": True,
        "family": "Linear model",
        "summary": "A weighted sum of the six standardised features, fitted by ordinary least squares.",
        "how": "Finds one coefficient per feature that minimises squared error. Because inputs are standardised, "
               "each coefficient is the change in SOH (pp) for a one-standard-deviation change in that feature.",
        "strengths": ["Fastest and smallest model", "Coefficients are directly interpretable", "Useful baseline"],
        "limits": ["Cannot follow the curved, accelerating capacity fade", "Lowest accuracy of the four"],
        "params": {"fit_intercept": True},
    },
}

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:4.5rem}
h1,h2,h3{letter-spacing:-.01em}
.kicker{font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#1f9d74}
.big{font-size:3rem;font-weight:700;line-height:1.05;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:.12rem .6rem;border-radius:999px;font-size:.78rem;font-weight:600}
.good{background:rgba(31,157,116,.16);color:#1f9d74}
.fair{background:rgba(214,150,20,.18);color:#b77d06}
.bad{background:rgba(214,69,69,.15);color:#d64545}
.step-n{font-size:.72rem;font-weight:700;color:#1f9d74;letter-spacing:.08em}
.step-v{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums}
.step-l{font-size:.82rem;opacity:.72;line-height:1.3;padding-bottom:.35rem}
</style>
""", unsafe_allow_html=True)


# ---------------- data & models ----------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner="Loading model…")
def load_pickle(path):
    return joblib.load(path)


def path_ok(*parts):
    p = os.path.join(*parts)
    return p if os.path.exists(p) else None


metrics = load_csv(os.path.join(PROJ, "model_comparison.csv"))
metrics = metrics.rename(columns={"R2 Score": "R2"}).sort_values("RMSE").reset_index(drop=True)
metrics["Rank"] = range(1, len(metrics) + 1)
RANK = {r.Model: r for r in metrics.itertuples()}
BEST = metrics.iloc[0]["Model"]
ORDER = list(metrics["Model"])
test_df = load_csv(os.path.join(DATA_DIR, "test_predictions.csv"))
curves = load_csv(os.path.join(DATA_DIR, "soh_curves.csv"))
ranges = load_csv(os.path.join(DATA_DIR, "train_feature_ranges.csv"))
importance = load_csv(os.path.join(PROJ, "feature_importance.csv"))
HAS_SCALER = os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl"))


def predict(name, X):
    m = MODELS[name]
    model = load_pickle(os.path.join(MODEL_DIR, m["file"]))
    if m["scaled"]:
        X = load_pickle(os.path.join(MODEL_DIR, "scaler.pkl")).transform(X)
    return model.predict(X)


def usable(name):
    return os.path.exists(os.path.join(MODEL_DIR, MODELS[name]["file"])) and (HAS_SCALER or not MODELS[name]["scaled"])


def health(soh):
    if soh >= 90:
        return "Good", "good"
    if soh >= 80:
        return "Ageing", "fair"
    return "End of life", "bad"


def model_label(name):
    r = RANK[name]
    return f"#{r.Rank} {name} · RMSE {r.RMSE:.2f}"


def show_image(name, caption=None):
    p = path_ok(PLOT_DIR, name)
    if p:
        st.image(p, caption=caption, width="stretch")
    else:
        st.caption(f"Plot not found: {name}")


# ---------------- page: training ----------------
def page_training():
    st.markdown('<div class="kicker">Step 1 · Training</div>', unsafe_allow_html=True)
    st.title("How the models were trained")
    st.write("Four regression models learn to estimate State of Health from one cycle's measurements. "
             "Each one was tested on 10 battery cells it never saw during training.")

    steps = [
        ("LOAD", "27,642", "cycles from 55 cells, 6 batches (XJTU-2024)"),
        ("FILTER", "8,976", "comparable 1C-discharge cycles from 47 cells"),
        ("LABEL", "SOH", "discharge capacity ÷ 2 Ah nominal × 100"),
        ("SPLIT", "37 / 10", "train / test cells, grouped by cell"),
        ("TRAIN", "4 models", "scored on 1,574 unseen test cycles"),
    ]
    cols = st.columns(len(steps))
    for i, (col, (n, v, l)) in enumerate(zip(cols, steps), 1):
        with col.container(border=True):
            st.markdown(f'<div class="step-n">{i:02d} {n}</div><div class="step-v">{v}</div>'
                        f'<div class="step-l">{l}</div>', unsafe_allow_html=True)

    # --- data
    st.subheader("The data: how batteries age")
    if curves is not None:
        batches = sorted(curves["batch"].unique())
        pick = st.multiselect("Batches", batches, default=batches, key="curve_batches")
        d = curves[curves["batch"].isin(pick)]
        chart = alt.Chart(d).mark_line(strokeWidth=1.2, opacity=.75).encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("SOH:Q", title="SOH (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("batch:N", title="Batch"),
            detail="cell:N",
            tooltip=["cell", "batch", "split", "cycle", alt.Tooltip("SOH:Q", format=".2f")],
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")
        st.caption("Each line is one cell. Capacity fades with use, and faster under harsher charging (Batch-2 uses 3C charge).")
    with st.expander("Exploratory plots from the notebook"):
        show_image("eda_overview.png", "SOH distribution, SOH vs cycle, per-cell curves, SOH by batch")
        show_image("correlation_heatmap.png", "Feature correlation heatmap")

    with st.expander("Why split by cell, and why scale?"):
        st.markdown(
            "- **Split by cell:** consecutive cycles of one cell are almost identical. A random split would leak "
            "near-copies into the test set and inflate scores. `GroupShuffleSplit` keeps every cell entirely in "
            "train *or* test, so the score reflects performance on a new battery.\n"
            "- **Scaling:** features range from ~2.5 V to ~7,000 s. Linear Regression and SVR are sensitive to that, "
            "so they receive inputs standardised with a `StandardScaler` fitted on the training cells only. "
            "The tree models (Random Forest, XGBoost) use raw values."
        )

    # --- results
    st.subheader("Results on the 10 test cells")
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        show = metrics[["Rank", "Model", "MAE", "RMSE", "R2"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "MAE": st.column_config.NumberColumn("MAE (pp)", format="%.3f"),
            "RMSE": st.column_config.NumberColumn("RMSE (pp)", format="%.3f"),
            "R2": st.column_config.ProgressColumn("R²", format="%.3f", min_value=0, max_value=1),
        })
        st.caption("MAE and RMSE are in SOH percentage points; lower is better. R² closer to 1 is better.")
    with c2:
        metric = st.segmented_control("Metric", ["RMSE", "MAE", "R2"], default="RMSE", key="cmp_metric") or "RMSE"
        bar = alt.Chart(metrics).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Model:N", sort=ORDER, title=None),
            x=alt.X(f"{metric}:Q", title=metric),
            color=alt.condition(alt.datum.Model == BEST, alt.value("#1f9d74"), alt.value("#9aa9b5")),
            tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
        ).properties(height=190)
        st.altair_chart(bar, width="stretch")

    st.markdown("**Actual vs predicted SOH**")
    m = st.segmented_control("Model", ORDER, default=BEST, key="avp_model") or BEST
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        show_image(MODELS[m]["plot"])
    with b:
        r = RANK[m]
        st.metric("MAE", f"{r.MAE:.3f} pp")
        st.metric("RMSE", f"{r.RMSE:.3f} pp")
        st.metric("R²", f"{r.R2:.3f}")
        st.caption("Points on the dashed diagonal are perfect predictions. Spread around it is error.")
    with st.expander("All four models side by side, and the metric chart from the notebook"):
        show_image("actual_vs_predicted.png")
        show_image("model_comparison.png")

    st.subheader("What drives the prediction")
    a, b = st.columns([1, 1], gap="large")
    with a:
        if importance is not None:
            imp = importance.sort_values("Importance", ascending=False)
            ch = alt.Chart(imp).mark_bar(cornerRadiusEnd=3, color="#1f9d74").encode(
                y=alt.Y("Feature:N", sort=None, title=None),
                x=alt.X("Importance:Q", axis=alt.Axis(format="%"), title="Random Forest importance"),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
            ).properties(height=220)
            st.altair_chart(ch, width="stretch")
    with b:
        st.markdown(
            "- **Charge capacity** (both columns) accounts for about **72%** of the Random Forest's decisions. "
            "A cell that accepts less charge has less usable capacity.\n"
            "- **Cycle number** adds ~19%: age matters on its own.\n"
            "- **Voltages** barely matter here because the test protocol fixes them (4.2 V / 2.5 V)."
        )
    st.info(f"**{BEST}** has the lowest error, so it is the default on the Predict page.", icon="➡️")


# ---------------- page: predict ----------------
def page_predict():
    st.markdown('<div class="kicker">Step 2 · Prediction</div>', unsafe_allow_html=True)
    st.title("Predict State of Health")

    ready = [n for n in ORDER if usable(n)]
    if not ready:
        st.error("No model files found in battery_health_project/models.")
        return
    if not HAS_SCALER:
        st.warning("Linear Regression and SVR are hidden because models/scaler.pkl is missing. "
                   "Run `python prepare_data.py xjtu-2024-cycles.csv` to rebuild it.")

    chosen = st.pills("Models (ranked by test RMSE)", ready, selection_mode="multi", default=[BEST],
                      format_func=model_label, key="pred_models")
    st.caption("Pick one model, or several to see whether they agree.")

    source = st.segmented_control("Input", ["Real test cycle", "Enter values"], default="Real test cycle",
                                  key="pred_source") or "Real test cycle"
    actual, cell_rows = None, None
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        if source == "Real test cycle" and test_df is not None:
            cells = sorted(test_df["cell"].unique())
            cell = st.selectbox("Test cell (never seen in training)", cells, key="pred_cell")
            cell_rows = test_df[test_df["cell"] == cell].sort_values("cycle").reset_index(drop=True)
            cyc = st.select_slider("Cycle", options=list(cell_rows["cycle"]), key=f"pred_cycle_{cell}")
            row = cell_rows[cell_rows["cycle"] == cyc].iloc[0]
            actual = float(row["actual_soh"])
            X = pd.DataFrame([row[FEAT].astype(float)], columns=FEAT)
            vals = pd.DataFrame({"Measurement": [FEATURES[f][0] for f in FEAT],
                                 "Value": [f"{row[f]:,.4g}" if f != "cycle" else str(int(row[f])) for f in FEAT]})
            st.dataframe(vals, hide_index=True, width="stretch")
            st.caption(f"{row['batch']} · measured SOH {actual:.2f}%")
        else:
            base = test_df.iloc[len(test_df) // 2] if test_df is not None else None
            vals = {}
            a, b = st.columns(2)
            for i, f in enumerate(FEAT):
                label, help_, fmt = FEATURES[f]
                lo, hi = (ranges.set_index(ranges.columns[0]).loc[f, ["min", "max"]] if ranges is not None else (0, 1e6))
                default = float(base[f]) if base is not None else float(lo)
                help_ = f"{help_}. Training range: {lo:,.4g} – {hi:,.4g}"
                col = a if i % 2 == 0 else b
                if f == "cycle":
                    vals[f] = col.number_input(label, 1, 5000, int(default), help=help_, key=f"in_{f}")
                else:
                    vals[f] = col.number_input(label, 0.0, 1e7, default, format=fmt, help=help_, key=f"in_{f}")
            X = pd.DataFrame([vals], columns=FEAT)
            if ranges is not None:
                r = ranges.set_index(ranges.columns[0])
                out = [FEATURES[f][0] for f in FEAT if not (r.loc[f, "min"] <= vals[f] <= r.loc[f, "max"])]
                if out:
                    st.warning("Outside the training range: " + ", ".join(out) + ". Predictions may be unreliable.")

    with right:
        if not chosen:
            st.info("Select at least one model.")
            return
        preds = {n: float(predict(n, X)[0]) for n in chosen}

        if len(preds) == 1:
            name, soh = next(iter(preds.items()))
            label, cls = health(soh)
            with st.container(border=True):
                st.caption(f"{name} · predicted SOH")
                st.markdown(f'<div class="big">{soh:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="pill {cls}">{label}</span>', unsafe_allow_html=True)
                st.progress(max(0.0, min(soh / 100, 1.0)))
                if actual is not None:
                    a, b = st.columns(2)
                    a.metric("Measured SOH", f"{actual:.2f}%")
                    b.metric("Error", f"{soh - actual:+.2f} pp")
        else:
            d = pd.DataFrame({"Model": list(preds), "Predicted SOH": list(preds.values())})
            if actual is not None:
                d["Error (pp)"] = d["Predicted SOH"] - actual
            med = d["Predicted SOH"].median()
            label, cls = health(med)
            a, b, c = st.columns(3)
            a.metric("Median prediction", f"{med:.2f}%")
            if actual is not None:
                b.metric("Measured SOH", f"{actual:.2f}%")
            c.metric("Model spread", f"{d['Predicted SOH'].max() - d['Predicted SOH'].min():.2f} pp")
            st.markdown(f'<span class="pill {cls}">{label}</span>', unsafe_allow_html=True)
            lo = min(d["Predicted SOH"].min(), actual or 100) - 1.5
            hi = max(d["Predicted SOH"].max(), actual or 0) + 1.5
            bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3, color="#1f9d74").encode(
                y=alt.Y("Model:N", sort=ORDER, title=None),
                x=alt.X("Predicted SOH:Q", scale=alt.Scale(domain=[lo, hi]), title="SOH (%)"),
                tooltip=["Model", alt.Tooltip("Predicted SOH:Q", format=".2f")],
            )
            layers = [bars]
            if actual is not None:
                layers.append(alt.Chart(pd.DataFrame({"x": [actual]})).mark_rule(strokeDash=[4, 3], strokeWidth=2)
                              .encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Measured", format=".2f")]))
            st.altair_chart(alt.layer(*layers).properties(height=45 * len(d) + 30), width="stretch")
            st.dataframe(d, hide_index=True, width="stretch", column_config={
                "Predicted SOH": st.column_config.NumberColumn(format="%.2f%%"),
                "Error (pp)": st.column_config.NumberColumn(format="%+.2f")})
            if actual is not None:
                st.caption("Dashed line = measured SOH.")

    if cell_rows is not None:
        st.subheader("Whole-life check for this cell")
        long = [pd.DataFrame({"cycle": cell_rows["cycle"], "SOH": cell_rows["actual_soh"], "Series": "Measured"})]
        for n in chosen:
            long.append(pd.DataFrame({"cycle": cell_rows["cycle"], "SOH": predict(n, cell_rows[FEAT]), "Series": n}))
        long = pd.concat(long)
        series = ["Measured"] + list(chosen)
        palette = ["#7d8896", "#1f9d74", "#e0892b", "#5a7bd8", "#c64f8e"][: len(series)]
        lines = alt.Chart(long).mark_line(strokeWidth=1.8).encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("SOH:Q", title="SOH (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=palette), title=None),
            strokeDash=alt.condition(alt.datum.Series == "Measured", alt.value([1, 0]), alt.value([5, 3])),
            tooltip=["Series", "cycle", alt.Tooltip("SOH:Q", format=".2f")],
        )
        rule = alt.Chart(pd.DataFrame({"cycle": [int(X["cycle"].iloc[0])]})).mark_rule(opacity=.4).encode(x="cycle:Q")
        st.altair_chart((lines + rule).properties(height=300), width="stretch")
        errs = {n: (long[long.Series == n]["SOH"].values - cell_rows["actual_soh"].values) for n in chosen}
        st.caption("Each model predicted every test cycle of this cell. MAE on this cell: " +
                   " · ".join(f"{n} {abs(e).mean():.2f} pp" for n, e in errs.items()))

    st.caption("Good ≥ 90% · Ageing 80–90% · End of life < 80% (common EV replacement threshold). "
               "Research estimate, not a safety assessment.")


# ---------------- page: model guide ----------------
def page_guide():
    st.markdown('<div class="kicker">Reference</div>', unsafe_allow_html=True)
    st.title("Model guide")
    st.write("What each model does, how it was configured in the notebook, and when to use it.")

    overview = pd.DataFrame([{
        "Rank": RANK[n].Rank, "Model": n, "Family": MODELS[n]["family"],
        "Input": "Standardised" if MODELS[n]["scaled"] else "Raw values",
        "MAE": RANK[n].MAE, "RMSE": RANK[n].RMSE, "R²": RANK[n].R2,
    } for n in ORDER])
    st.dataframe(overview, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in ["MAE", "RMSE", "R²"]})

    for n in ORDER:
        m, r = MODELS[n], RANK[n]
        with st.expander(f"#{r.Rank}  {n}  ·  {m['family']}", expanded=(n == BEST)):
            a, b = st.columns([1.4, 1], gap="large")
            with a:
                st.markdown(f"**{m['summary']}**")
                st.markdown(m["how"])
                s, l = st.columns(2)
                s.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in m["strengths"]))
                l.markdown("**Limitations**\n" + "\n".join(f"- {x}" for x in m["limits"]))
            with b:
                x, y, z = st.columns(3)
                x.metric("MAE", f"{r.MAE:.3f}")
                y.metric("RMSE", f"{r.RMSE:.3f}")
                z.metric("R²", f"{r.R2:.3f}")
                st.markdown("**Hyperparameters**")
                st.code("\n".join(f"{k} = {v!r}" for k, v in m["params"].items()), language="python")
                st.caption("Input: " + ("StandardScaler → model" if m["scaled"] else "raw feature values"))

    st.subheader("Which one should I use?")
    st.markdown(
        f"- **Most accurate estimate:** {BEST}. It has the lowest error on unseen cells.\n"
        "- **Second opinion:** XGBoost. When it agrees with Random Forest, the estimate is more trustworthy.\n"
        "- **Explaining the result:** Linear Regression. Each coefficient shows a feature's direct effect.\n"
        "- **Unusual inputs:** be careful with every model. Tree models cannot extrapolate beyond the training range."
    )
    st.subheader("Input features")
    st.dataframe(pd.DataFrame([{"Feature": f, "Meaning": FEATURES[f][1]} for f in FEAT]),
                 hide_index=True, width="stretch")


pg = st.navigation([
    st.Page(page_training, title="Training", icon=":material/model_training:", default=True),
    st.Page(page_predict, title="Predict", icon=":material/battery_charging_full:", url_path="predict"),
    st.Page(page_guide, title="Model guide", icon=":material/menu_book:", url_path="guide"),
], position="top")
pg.run()
