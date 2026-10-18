
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PJM Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DASHBOARD CSS + BACKGROUND IMAGE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-image:
        linear-gradient(
            110deg,
            rgba(3, 13, 28, 0.88),
            rgba(5, 24, 42, 0.90)
        ),
        url("https://images.unsplash.com/photo-1473341304170-971a741c7a1e?auto=format&fit=crop&w=2200&q=85");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020b18, #06203a);
    border-right: 1px solid #2c8ac4;
}

[data-testid="stSidebar"] * {
    color: #f2f8ff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.hero-title {
    font-size: clamp(2rem, 3vw, 3rem);
    font-weight: 800;
    letter-spacing: -1.2px;
    color: #ffffff;
    text-shadow: 0 0 16px rgba(56,189,248,.55);
}

.hero-subtitle {
    color: #7dd3fc;
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 8px;
    margin-bottom: 30px;
}

.section-title {
    color: #ffffff;
    font-weight: 800;
    font-size: 1.55rem;
    margin: 26px 0 16px;
    border-left: 4px solid #38bdf8;
    padding-left: 12px;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.small-muted {
    color: #b7cce0;
    font-size: .9rem;
}

.kpi {
    background: linear-gradient(
        145deg,
        rgba(12,42,72,.98),
        rgba(5,25,46,.98)
    );
    border: 1px solid #2688c2;
    border-radius: 20px;
    padding: 22px;
    min-height: 135px;
    box-shadow: 0 8px 25px rgba(0,0,0,.2);
}

.kpi-label {
    color: #7dd3fc;
    text-transform: uppercase;
    font-size: .82rem;
    font-weight: 800;
    letter-spacing: .8px;
}

.kpi-value {
    color: #ffffff;
    font-size: clamp(1.15rem, 1.7vw, 1.8rem);
    font-weight: 800;
    margin-top: 9px;
    overflow-wrap: anywhere;
}

.kpi-note {
    color: #b7d7f3;
    font-size: .78rem;
    margin-top: 7px;
}

.status {
    background: rgba(5,54,63,.92);
    border: 1px solid #24b8a6;
    padding: 14px 16px;
    border-radius: 13px;
    margin: 10px 0;
    color: #d7fff8;
    font-weight: 600;
}

.panel {
    background: rgba(4,22,41,.88);
    border: 1px solid #1d5278;
    border-radius: 18px;
    padding: 12px;
}

div[data-testid="stMetric"] {
    background: rgba(6,34,58,.92);
    border: 1px solid #277cad;
    padding: 16px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #7dd3fc !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg,#0879b9,#0e9fba);
    color: white;
    border: 1px solid #7dd3fc;
    border-radius: 10px;
    font-weight: 700;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: linear-gradient(90deg,#0e9fba,#22c7d6);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data(show_spinner=False)
def read_dataset(file_bytes, filename):
    import io

    file_buffer = io.BytesIO(file_bytes)

    if filename.lower().endswith(".csv"):
        return pd.read_csv(file_buffer)

    return pd.read_excel(file_buffer)


def prepare_dataset(raw_df, date_col, demand_col):
    df = raw_df[[date_col, demand_col]].copy()
    df.columns = ["Datetime", "Demand"]

    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="coerce"
    )
    df["Demand"] = pd.to_numeric(
        df["Demand"], errors="coerce"
    )

    df = (
        df.dropna()
        .sort_values("Datetime")
        .drop_duplicates(
            subset="Datetime",
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


def create_features(df):
    data = df.copy()
    dt = data["Datetime"]

    data["hour"] = dt.dt.hour
    data["dayofweek"] = dt.dt.dayofweek
    data["day"] = dt.dt.day
    data["month"] = dt.dt.month
    data["quarter"] = dt.dt.quarter
    data["year"] = dt.dt.year
    data["dayofyear"] = dt.dt.dayofyear

    data["is_weekend"] = (
        dt.dt.dayofweek >= 5
    ).astype(int)

    for lag in [1, 2, 3, 24, 48, 168]:
        data[f"lag_{lag}"] = (
            data["Demand"].shift(lag)
        )

    data["rolling_mean_24"] = (
        data["Demand"].shift(1).rolling(24).mean()
    )

    data["rolling_mean_168"] = (
        data["Demand"].shift(1).rolling(168).mean()
    )

    return data.dropna().reset_index(drop=True)


def make_model():
    return XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


def style_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,15,29,.45)",
        font=dict(
            color="#eff8ff",
            family="Inter, sans-serif"
        ),
        margin=dict(l=15, r=15, t=45, b=12),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),
        xaxis=dict(
            gridcolor="rgba(180,215,240,.18)",
            zeroline=False
        ),
        yaxis=dict(
            title="Demand (MW)",
            gridcolor="rgba(180,215,240,.18)",
            zeroline=False
        )
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚡ Energy AI")

    st.markdown(
        "<div style='color:#7dd3fc;font-weight:700;"
        "font-size:13px;'>"
        "PJM DEMAND FORECASTING SYSTEM"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 🎯 Forecast Settings")

    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=30,
        value=30
    )

    st.caption(
        f"Hourly predictions: {forecast_days * 24}"
    )

    st.markdown("---")
    st.markdown("### 🟢 System Status")

    st.markdown(
        '<div class="status">● Dashboard initialized</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="status">● Upload a dataset to train</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="status">● Forecast engine available</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("Supported formats: CSV, XLSX, XLS")


# ============================================================
# DASHBOARD HEADER
# ============================================================
st.markdown("""
<div style="display:flex;align-items:center;gap:18px;">
    <span style="font-size:3rem;">⚡</span>
    <div class="hero-title">PJM Energy Intelligence</div>
</div>

<div class="hero-subtitle">
Machine learning powered electricity demand forecasting
using an XGBoost regression model.
</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD DATASET
# ============================================================
st.markdown(
    '<div class="section-title">📂 Dataset Configuration</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload hourly electricity demand data",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info(
        "Upload your PJM dataset to display metrics, "
        "train the model, and generate forecasts."
    )

    st.markdown("#### Expected dataset format")

    st.dataframe(
        pd.DataFrame({
            "Datetime": [
                "2018-01-01 00:00:00",
                "2018-01-01 01:00:00",
                "2018-01-01 02:00:00"
            ],
            "PJMW_MW": [5728, 5512, 5316]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.stop()

try:
    raw_df = read_dataset(
        uploaded_file.getvalue(),
        uploaded_file.name
    )
except Exception as exc:
    st.error(f"Unable to read the uploaded file: {exc}")
    st.stop()

if raw_df.empty or len(raw_df.columns) < 2:
    st.error(
        "The file must contain a date/time column "
        "and a demand column."
    )
    st.stop()

columns = list(raw_df.columns)

date_matches = [
    i for i, name in enumerate(columns)
    if any(
        word in str(name).lower()
        for word in ["date", "time", "timestamp"]
    )
]

demand_matches = [
    i for i, name in enumerate(columns)
    if any(
        word in str(name).lower()
        for word in ["pjmw", "mw", "demand", "load", "usage"]
    )
]

default_date = date_matches[0] if date_matches else 0

default_demand = (
    demand_matches[0]
    if demand_matches
    else (1 if default_date == 0 else 0)
)

if default_demand == default_date:
    default_demand = 1 if default_date == 0 else 0

left, right = st.columns(2)

with left:
    date_col = st.selectbox(
        "Date/time column",
        columns,
        index=default_date
    )

with right:
    demand_options = [
        c for c in columns if c != date_col
    ]

    preferred = (
        columns[default_demand]
        if columns[default_demand] in demand_options
        else demand_options[0]
    )

    demand_col = st.selectbox(
        "Electricity demand column (MW)",
        demand_options,
        index=demand_options.index(preferred)
    )

df = prepare_dataset(
    raw_df,
    date_col,
    demand_col
)

if len(df) < 180:
    st.error(
        "At least 180 valid observations are needed "
        "for the 168-hour lag features."
    )
    st.stop()


# ============================================================
# FEATURE ENGINEERING + CHRONOLOGICAL SPLIT
# ============================================================
df = df.sort_values("Datetime").reset_index(drop=True)

feature_data = create_features(df)

feature_cols = [
    c for c in feature_data.columns
    if c not in ["Datetime", "Demand"]
]

split_idx = int(len(feature_data) * 0.8)

if split_idx < 10 or len(feature_data) - split_idx < 10:
    st.error(
        "Not enough observations for model evaluation."
    )
    st.stop()

X_train = feature_data[feature_cols].iloc[:split_idx]
y_train = feature_data["Demand"].iloc[:split_idx]

X_test = feature_data[feature_cols].iloc[split_idx:]
y_test = feature_data["Demand"].iloc[split_idx:]


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================
with st.spinner("Training the XGBoost model..."):
    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

current_demand = float(df["Demand"].iloc[-1])
average_demand = float(df["Demand"].mean())
peak_demand = float(df["Demand"].max())
last_observation = df["Datetime"].iloc[-1]


# ============================================================
# KPI OVERVIEW
# ============================================================
st.markdown(
    '<div class="section-title">Historical Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

kpi_items = [
    (
        c1,
        "CURRENT DEMAND",
        f"{current_demand:,.0f} MW",
        "Most recent observation"
    ),
    (
        c2,
        "HISTORICAL AVERAGE",
        f"{average_demand:,.0f} MW",
        "Average recorded demand"
    ),
    (
        c3,
        "PEAK DEMAND",
        f"{peak_demand:,.0f} MW",
        "Maximum recorded demand"
    ),
    (
        c4,
        "LAST OBSERVATION",
        last_observation.strftime("%d %b %Y %H:%M"),
        "Latest timestamp"
    )
]

for col, label, value, note in kpi_items:
    with col:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# HISTORICAL DEMAND CHART
# ============================================================
st.markdown(
    '<div class="section-title">'
    'Historical Demand — Last 7 Days'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="small-muted">'
    'Hourly electricity demand from the most recent '
    'seven-day period.'
    '</div>',
    unsafe_allow_html=True
)

recent = df.tail(min(24 * 7, len(df)))

hist_fig = go.Figure()

hist_fig.add_trace(go.Scatter(
    x=recent["Datetime"],
    y=recent["Demand"],
    mode="lines",
    name="Historical demand",
    line=dict(color="#38bdf8", width=2.7)
))

hist_fig = style_chart(hist_fig, 390)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.plotly_chart(hist_fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================
st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

m1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f} MW")
m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f} MW")
m3.metric("R² Score", f"{r2:.4f}")

evaluation_fig = go.Figure()

evaluation_fig.add_trace(go.Scatter(
    x=feature_data["Datetime"].iloc[split_idx:],
    y=y_test,
    mode="lines",
    name="Actual",
    line=dict(color="#38bdf8", width=2)
))

evaluation_fig.add_trace(go.Scatter(
    x=feature_data["Datetime"].iloc[split_idx:],
    y=predictions,
    mode="lines",
    name="Predicted",
    line=dict(color="#fb923c", width=2)
))

evaluation_fig.update_layout(
    title="Actual vs Predicted Demand"
)

evaluation_fig = style_chart(evaluation_fig, 390)

st.plotly_chart(
    evaluation_fig,
    use_container_width=True
)


# ============================================================
# FUTURE DEMAND FORECAST
# ============================================================
st.markdown(
    '<div class="section-title">Future Demand Forecast</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Forecast horizon: {forecast_days} day(s) · "
    f"{forecast_days * 24} hourly steps"
)

if st.button("⚡ Generate Forecast", type="primary"):

    with st.spinner("Generating future demand forecast..."):

        history = df[
            ["Datetime", "Demand"]
        ].copy().reset_index(drop=True)

        forecast_rows = []

        for _ in range(forecast_days * 24):

            next_time = (
                history["Datetime"].iloc[-1]
                + pd.Timedelta(hours=1)
            )

            values = history["Demand"]

            row = {
                "hour": next_time.hour,
                "dayofweek": next_time.dayofweek,
                "day": next_time.day,
                "month": next_time.month,
                "quarter": next_time.quarter,
                "year": next_time.year,
                "dayofyear": next_time.dayofyear,
                "is_weekend": int(next_time.dayofweek >= 5),

                "lag_1": values.iloc[-1],
                "lag_2": values.iloc[-2],
                "lag_3": values.iloc[-3],
                "lag_24": values.iloc[-24],
                "lag_48": values.iloc[-48],
                "lag_168": values.iloc[-168],

                "rolling_mean_24": values.tail(24).mean(),
                "rolling_mean_168": values.tail(168).mean()
            }

            next_X = pd.DataFrame([row])[feature_cols]

            predicted_demand = float(
                model.predict(next_X)[0]
            )

            forecast_rows.append({
                "Datetime": next_time,
                "Predicted_Demand_MW": predicted_demand
            })

            history.loc[len(history)] = [
                next_time,
                predicted_demand
            ]

        forecast_df = pd.DataFrame(forecast_rows)

        forecast_fig = go.Figure()

        recent_actual = df.tail(72)

        forecast_fig.add_trace(go.Scatter(
            x=recent_actual["Datetime"],
            y=recent_actual["Demand"],
            mode="lines",
            name="Recent actual",
            line=dict(color="#38bdf8", width=2.2)
        ))

        forecast_fig.add_trace(go.Scatter(
            x=forecast_df["Datetime"],
            y=forecast_df["Predicted_Demand_MW"],
            mode="lines",
            name="Forecast",
            line=dict(color="#fb923c", width=2.7)
        ))

        forecast_fig.update_layout(
            title="Hourly Electricity Demand Forecast"
        )

        forecast_fig = style_chart(forecast_fig, 400)

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.markdown("#### Forecast data")

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="pjm_energy_forecast.csv",
            mime="text/csv"
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
with st.expander("🔍 View model feature importance"):

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    ).tail(15)

    importance_fig = go.Figure(go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h",
        marker_color="#38bdf8"
    ))

    importance_fig.update_layout(
        title="Top 15 Important Features"
    )

    importance_fig = style_chart(importance_fig, 450)

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown(
    "<div style='text-align:center;color:#b7cce0;padding:18px;'>"
    "PJM Energy Intelligence · XGBoost Demand Forecasting"
    "</div>",
    unsafe_allow_html=True
)
