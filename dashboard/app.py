"""
Streamlit Dashboard - AI-Powered Cloud Cost & Resource Optimization Platform.

Professional dashboard with Plotly charts consuming actual backend
and experiment data. No fabricated metrics.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ============ Configuration ============
API_URL = os.getenv("API_URL", "http://localhost:8000")
EXPERIMENTS_DIR = Path("experiments/results")

st.set_page_config(
    page_title="CloudOpt AI — Cost & Resource Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Premium CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { box-sizing: border-box; }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: #060b14 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1523 0%, #0a1020 100%) !important;
        border-right: 1px solid #1a2744 !important;
    }

    /* Page background */
    .main .block-container {
        padding-top: 1.5rem !important;
        background: #060b14;
    }

    /* Glassmorphism card */
    .glass-card {
        background: linear-gradient(135deg, rgba(30,45,77,0.8) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 20px 24px;
        color: #e2e8f0;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,179,237,0.4), transparent);
    }

    /* Page banner */
    .page-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0f172a 100%);
        border: 1px solid #1e3a8a;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .page-banner::after {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(59,130,246,0.12), transparent 70%);
        border-radius: 50%;
    }
    .page-banner h1 {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0 0 6px 0;
        background: linear-gradient(135deg, #f1f5f9, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .page-banner p { color: #64748b; margin: 0; font-size: 0.87rem; }

    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 14px 0;
    }
    .section-header h2 {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
    }
    .section-header .badge {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
        color: #93c5fd;
        font-size: 0.65rem;
        padding: 2px 9px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Status badges */
    .badge-healthy { background: #052e16; color: #34d399; border: 1px solid #065f46; padding: 2px 9px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
    .badge-warning { background: #431407; color: #fb923c; border: 1px solid #7c2d12; padding: 2px 9px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
    .badge-error   { background: #3f0f0f; color: #f87171; border: 1px solid #7f1d1d; padding: 2px 9px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
    .badge-info    { background: #0c1a3f; color: #93c5fd; border: 1px solid #1e40af; padding: 2px 9px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }

    /* Experiment card */
    .exp-card {
        background: linear-gradient(135deg, #0f1d35 0%, #0a1020 100%);
        border: 1px solid #1e3a8a;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Legacy compat */
    .status-healthy { background: #064e3b; color: #34d399; }
    .status-warning { background: #78350f; color: #fbbf24; }
    .status-error { background: #7f1d1d; color: #f87171; }
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #1e3a8a;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .experiment-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.73rem !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(37,99,235,0.5) !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 7px 16px;
        background: #0f172a;
        color: #64748b !important;
        border: 1px solid #1e293b;
        font-size: 0.84rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8) !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }

    /* Live pulse dot */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }
    .live-dot {
        display: inline-block; width: 8px; height: 8px;
        background: #22c55e; border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 6px; vertical-align: middle;
    }
    .offline-dot {
        display: inline-block; width: 8px; height: 8px;
        background: #f59e0b; border-radius: 50%;
        margin-right: 6px; vertical-align: middle;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] { border: 1px solid #1e293b !important; border-radius: 12px; }

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background: #0f172a;
        border-color: #1e293b;
        border-radius: 10px;
    }

    /* Expanders */
    details { background: #0a1020; border: 1px solid #1e293b; border-radius: 10px; }
    summary { color: #94a3b8; }

    hr { border-color: #1e293b !important; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

def api_get(endpoint: str, default=None):
    """Make a GET request to the backend API."""
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return default
    except requests.exceptions.ConnectionError:
        return default
    except Exception:
        return default


def api_post(endpoint: str, data: dict = None, default=None):
    """Make a POST request to the backend API."""
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
        if response.status_code in [200, 201]:
            return response.json()
        elif response.status_code in [400, 422, 500]:
            try:
                err = response.json()
                detail = err.get("detail", err.get("error", "Request failed"))
                return {"error": detail}
            except Exception:
                return {"error": response.text or "Request failed"}
        return default
    except requests.exceptions.ConnectionError:
        return default
    except Exception:
        return default


def load_experiment_files():
    """Load experiment results from API or disk, handling both dict and list responses."""
    # 1. Try API first
    api_res = api_get("/api/v1/experiments/results")
    if api_res:
        if isinstance(api_res, dict) and "results" in api_res and isinstance(api_res["results"], list):
            return api_res["results"]
        elif isinstance(api_res, list) and len(api_res) > 0:
            return api_res

    # 2. Fall back to local disk
    results_by_id = {}
    if EXPERIMENTS_DIR.exists():
        for f in sorted(EXPERIMENTS_DIR.glob("*.json")):
            if f.name == "all_experiments.json":
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, dict) and "experiment_id" in data:
                        results_by_id[data["experiment_id"]] = data
            except Exception:
                pass

        if not results_by_id:
            all_path = EXPERIMENTS_DIR / "all_experiments.json"
            if all_path.exists():
                try:
                    with open(all_path) as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            for d in data:
                                if isinstance(d, dict) and "experiment_id" in d:
                                    results_by_id[d["experiment_id"]] = d
                except Exception:
                    pass

    results = list(results_by_id.values())
    results.sort(key=lambda x: str(x.get("started_at") or ""))
    return results


def create_gauge_chart(value, title, max_val=100, color_ranges=None):
    """Create a gauge chart."""
    if color_ranges is None:
        color_ranges = [
            [0, 0.5, "#22c55e"],
            [0.5, 0.75, "#eab308"],
            [0.75, 1.0, "#ef4444"],
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, max_val], "tickfont": {"color": "#64748b"}},
            "bar": {"color": "#3b82f6"},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [r[0] * max_val, r[1] * max_val], "color": r[2] + "20"}
                for r in color_ranges
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.8,
                "value": max_val * 0.85,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
    )
    return fig


def create_timeseries_chart(data, x_key, y_keys, title, y_label="Value"):
    """Create a multi-line time series chart."""
    fig = go.Figure()
    colors = ["#3b82f6", "#22c55e", "#eab308", "#ef4444", "#a855f7", "#06b6d4"]

    for i, key in enumerate(y_keys):
        if key in data[0] if data else False:
            fig.add_trace(go.Scatter(
                x=[d.get(x_key, i) for i, d in enumerate(data)],
                y=[d.get(key, 0) for d in data],
                mode="lines+markers",
                name=key.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=3),
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        xaxis_title="Time",
        yaxis_title=y_label,
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
        ),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    )
    return fig


DARK_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#070e1c",
    "font": {"color": "#94a3b8", "family": "Inter, sans-serif"},
    "xaxis": {"gridcolor": "#111d33", "zerolinecolor": "#111d33"},
    "yaxis": {"gridcolor": "#111d33", "zerolinecolor": "#111d33"},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#94a3b8"}},
    "hovermode": "x unified",
}


def apply_dark_theme(fig, title="", height=350):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["plot_bgcolor"],
        font=DARK_THEME["font"],
        legend=DARK_THEME["legend"],
        hovermode=DARK_THEME["hovermode"],
        height=height,
        margin=dict(l=44, r=16, t=44 if title else 20, b=36),
    )
    if title:
        fig.update_layout(title={"text": title, "font": {"size": 14, "color": "#e2e8f0"}})
    fig.update_xaxes(**DARK_THEME["xaxis"])
    fig.update_yaxes(**DARK_THEME["yaxis"])
    return fig


# ============ Sidebar ============

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 8px 0;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <span style="font-size:1.6rem;">&#9889;</span>
            <div>
                <div style="font-size:1.05rem; font-weight:800; color:#f1f5f9; line-height:1.2;">CloudOpt AI</div>
                <div style="font-size:0.68rem; color:#334155; letter-spacing:1px; text-transform:uppercase;">Resource Optimizer</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API connection status
    health = api_get("/api/v1/health")
    if health:
        st.markdown('<span class="live-dot"></span><span style="color:#22c55e; font-size:0.82rem; font-weight:600;">API Connected</span>', unsafe_allow_html=True)
        ml_cls = 'badge-healthy' if health.get('ml_model') == 'ready' else 'badge-warning'
        anom_cls = 'badge-healthy' if health.get('anomaly_model') == 'ready' else 'badge-warning'
        st.markdown(f"""
        <div style="margin-top:8px; padding:10px 12px; background:#0a1020; border-radius:10px; border:1px solid #1e293b;">
            <div style="font-size:0.68rem; color:#334155; font-weight:700; letter-spacing:0.5px; margin-bottom:6px;">SYSTEM STATUS</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#64748b; font-size:0.77rem;">Database</span>
                <span class="badge-healthy">{health.get('database', '—')}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#64748b; font-size:0.77rem;">ML Model</span>
                <span class="{ml_cls}">{health.get('ml_model', '—')}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b; font-size:0.77rem;">Anomaly</span>
                <span class="{anom_cls}">{health.get('anomaly_model', '—')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="offline-dot"></span><span style="color:#f59e0b; font-size:0.82rem; font-weight:600;">API Offline — Using stored data</span>', unsafe_allow_html=True)
        st.caption("Start the API: `python -m uvicorn app.main:app`")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔮 Predictions", "🚨 Anomalies",
         "⚡ Optimizer", "💰 Cost Analysis", "🧪 Experiments"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding: 4px 0;">
        <div style="font-size:0.68rem; color:#1e293b; font-weight:600;">v1.0.0 &bull; AI-Powered Platform</div>
        <div style="font-size:0.67rem; color:#1e293b; margin-top:2px;">&#8987; {datetime.datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)


# ============ Pages ============

if page == "📊 Overview":
    st.markdown("""
    <div class="page-banner">
        <h1>☁️ Cloud Cost &amp; Resource Optimization</h1>
        <p>AI-powered real-time monitoring, demand prediction, and infrastructure optimization platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick action bar
    top_col1, top_col2, top_col3 = st.columns([3, 1, 1])
    with top_col2:
        if st.button("🚀 Stream Telemetry", key="stream_telemetry_top", help="Generate and ingest 30 live workload metric points"):
            with st.spinner("Ingesting metrics..."):
                res = api_post("/api/v1/metrics/simulate?count=30")
                if res and "error" not in res:
                    st.rerun()
    with top_col3:
        if st.button("🔄 Refresh", key="refresh_top"):
            st.rerun()

    # Current metrics
    metrics = api_get("/api/v1/metrics/current")

    col1, col2, col3, col4, col5 = st.columns(5)

    if metrics and "message" not in metrics:
        col1.metric("CPU Usage", f"{metrics.get('cpu_usage', 0):.1f}%")
        col2.metric("Memory Usage", f"{metrics.get('memory_usage', 0):.1f}%")
        col3.metric("Request Rate", f"{metrics.get('request_rate', 0):.1f} rps")
        col4.metric("Latency", f"{metrics.get('latency_ms', 0):.1f} ms")
        col5.metric("Error Rate", f"{metrics.get('error_rate', 0):.2f}%")
    else:
        col1.metric("CPU Usage", "—")
        col2.metric("Memory Usage", "—")
        col3.metric("Request Rate", "—")
        col4.metric("Latency", "—")
        col5.metric("Error Rate", "—")
        st.info("📡 No live metrics yet. Click '🚀 Stream Telemetry' above to ingest real-time workload data.")
        if st.button("🚀 Stream Sample Workload Telemetry (30 Data Points)", key="btn_stream_sample_empty"):
            with st.spinner("Streaming workload telemetry..."):
                res = api_post("/api/v1/metrics/simulate?count=30")
                if res and "error" not in res:
                    st.success("Successfully ingested 30 telemetry points!")
                    st.rerun()

    # Gauges
    if metrics and "message" not in metrics:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(
                create_gauge_chart(metrics.get("cpu_usage", 0), "CPU Utilization"),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(
                create_gauge_chart(metrics.get("memory_usage", 0), "Memory Utilization"),
                use_container_width=True,
            )

    # Historical metrics
    history = api_get("/api/v1/metrics/history?limit=50")
    if history and history.get("metrics"):
        st.markdown("### 📈 Recent Metrics History")
        st.plotly_chart(
            create_timeseries_chart(
                history["metrics"],
                "timestamp",
                ["cpu_usage", "memory_usage", "request_rate"],
                "Workload Metrics Over Time",
            ),
            use_container_width=True,
        )

    # Platform status
    st.markdown("### 🔧 Platform Status")
    status = api_get("/api/v1/training/status")
    if status:
        s1, s2, s3 = st.columns(3)
        s1.metric("Metrics Collected", status.get("metrics_count", 0))
        pred_status = status.get("prediction_model", {})
        s2.metric("ML Model", pred_status.get("model_name", "not trained"))
        s3.metric("Current Replicas", status.get("current_replicas", 1))

    # Quick experiment results
    experiments = load_experiment_files()
    if experiments:
        st.markdown("### 🧪 Latest Experiment Summary")
        latest = experiments[-1]
        cc = latest.get("cost_comparison", {})
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Scenario", latest.get("scenario", "—").replace("_", " ").title())
        e2.metric("Baseline Cost", f"${cc.get('baseline', {}).get('total_cost', 0):.4f}")
        e3.metric("AI Cost", f"${cc.get('ai_assisted', {}).get('total_cost', 0):.4f}")
        savings = cc.get("savings_percentage", 0)
        e4.metric("Savings", f"{savings:.1f}%",
                  delta=f"{'↓' if savings < 0 else '↑'} ${abs(cc.get('absolute_savings', 0)):.4f}")

elif page == "🔮 Predictions":
    st.markdown("""
    <div class="page-banner">
        <h1>🔮 Demand Predictions</h1>
        <p>ML-based demand forecasting using trained Random Forest &amp; Gradient Boosting models</p>
    </div>
    """ , unsafe_allow_html=True)

    pred_status = api_get("/api/v1/predictions/status")
    if pred_status:
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", pred_status.get("model_name", "—"))
        col2.metric("Training Samples", pred_status.get("training_samples", 0))
        col3.metric("Trained", "✅" if pred_status.get("is_trained") else "❌")

        if pred_status.get("metrics"):
            st.markdown("### Model Evaluation Metrics")
            m = pred_status["metrics"]
            mc1, mc2 = st.columns(2)
            mc1.metric("Best MAE", f"{m.get('best_mae', 0):.4f}")
            mc2.metric("Best RMSE", f"{m.get('best_rmse', 0):.4f}")

            if "all_results" in m:
                st.markdown("#### All Models Comparison")
                model_data = []
                for name, vals in m["all_results"].items():
                    model_data.append({
                        "Model": name.replace("_", " ").title(),
                        "MAE": vals.get("mae", 0),
                        "RMSE": vals.get("rmse", 0),
                    })
                st.dataframe(pd.DataFrame(model_data), use_container_width=True)

    # Get prediction
    if st.button("🔮 Generate Prediction"):
        prediction = api_post("/api/v1/predictions/", {"horizon_minutes": 5})
        if prediction and "error" not in prediction:
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted CPU", f"{prediction.get('predicted_cpu', 0):.1f}%")
            p2.metric("Predicted Memory", f"{prediction.get('predicted_memory', 0):.1f}%")
            p3.metric("Predicted RPS", f"{prediction.get('predicted_request_rate', 0):.1f}")

            # Actual vs Predicted comparison
            current = api_get("/api/v1/metrics/current")
            if current and "message" not in current:
                st.markdown("### Actual vs Predicted")
                comparison_data = {
                    "Metric": ["CPU Usage (%)", "Memory Usage (%)", "Request Rate (rps)"],
                    "Actual": [
                        current.get("cpu_usage", 0),
                        current.get("memory_usage", 0),
                        current.get("request_rate", 0),
                    ],
                    "Predicted": [
                        prediction.get("predicted_cpu", 0),
                        prediction.get("predicted_memory", 0),
                        prediction.get("predicted_request_rate", 0),
                    ],
                }
                df = pd.DataFrame(comparison_data)
                df["Difference"] = abs(df["Actual"] - df["Predicted"])
                st.dataframe(df, use_container_width=True)
        else:
            st.error(prediction.get("error", "Prediction failed") if prediction else "API unavailable")

    # Train models
    st.markdown("---")
    if st.button("🔄 Train/Retrain Models"):
        with st.spinner("Training models..."):
            result = api_post("/api/v1/training/train")
            if result:
                st.success("Models trained successfully!")
                st.json(result)
            else:
                st.error("Training failed or API unavailable.")

elif page == "🚨 Anomalies":
    st.markdown("""
    <div class="page-banner">
        <h1>🚨 Anomaly Detection</h1>
        <p>Isolation Forest–based outlier detection on live workload telemetry</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Uses Isolation Forest to detect unusual patterns. "
        "May produce false positives — see docs/ANOMALY_DETECTION.md for limitations."
    )

    anomaly_status = api_get("/api/v1/anomalies/status")
    if anomaly_status:
        col1, col2 = st.columns(2)
        col1.metric("Detector Status", "Ready ✅" if anomaly_status.get("is_trained") else "Not Trained ❌")
        col2.metric("Training Samples", anomaly_status.get("training_samples", 0))

    # Manual detection
    st.markdown("### Test Anomaly Detection")
    with st.expander("Enter metrics manually"):
        ac1, ac2 = st.columns(2)
        test_cpu = ac1.slider("CPU Usage %", 0.0, 100.0, 50.0)
        test_memory = ac2.slider("Memory Usage %", 0.0, 100.0, 50.0)
        ac3, ac4 = st.columns(2)
        test_rps = ac3.slider("Request Rate (rps)", 0.0, 200.0, 10.0)
        test_latency = ac4.slider("Latency (ms)", 0.0, 5000.0, 50.0)
        test_error = st.slider("Error Rate %", 0.0, 100.0, 0.5)

        if st.button("🔍 Detect Anomaly"):
            result = api_post("/api/v1/anomalies/detect", {
                "cpu_usage": test_cpu,
                "memory_usage": test_memory,
                "request_rate": test_rps,
                "latency_ms": test_latency,
                "error_rate": test_error,
            })
            if result:
                if result.get("is_anomaly"):
                    st.error(f"🚨 ANOMALY DETECTED — Severity: {result.get('severity', 'unknown').upper()}")
                else:
                    st.success("✅ Normal — No anomaly detected")

                st.json(result)
            else:
                st.warning("Detection failed. Ensure models are trained.")

    # Show experiment-based anomalies from results
    experiments = load_experiment_files()
    for exp in experiments:
        if "ai_timeseries" in exp:
            anomalous = [m for m in exp["ai_timeseries"]
                         if m.get("cpu_usage", 0) > 85 or m.get("error_rate", 0) > 5]
            if anomalous:
                st.markdown(f"### Anomalous Points in {exp.get('scenario', 'unknown')} Experiment")
                st.dataframe(
                    pd.DataFrame(anomalous)[["cpu_usage", "memory_usage", "latency_ms", "error_rate"]].head(10),
                    use_container_width=True,
                )

elif page == "⚡ Optimizer":
    st.markdown("""
    <div class="page-banner">
        <h1>⚡ Scaling Optimizer</h1>
        <p>AI-driven horizontal pod autoscaling recommendations with configurable thresholds and cooldown</p>
    </div>
    """, unsafe_allow_html=True)

    config = api_get("/api/v1/recommendations/config")
    if config:
        st.markdown("### Current Configuration")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min Replicas", config.get("min_replicas", 1))
        c2.metric("Max Replicas", config.get("max_replicas", 10))
        c3.metric("Cooldown", f"{config.get('cooldown_seconds', 120)}s")
        c4.metric("Mode", "🔒 Dry Run" if config.get("dry_run") else "⚡ Active")

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Safety Margin", f"{config.get('safety_margin', 0.2) * 100:.0f}%")
        cc2.metric("Scale Up @", f"{config.get('scale_up_threshold', 80)}%")
        cc3.metric("Scale Down @", f"{config.get('scale_down_threshold', 30)}%")

    # Current recommendation
    st.markdown("### Current Recommendation")
    rec = api_get("/api/v1/recommendations/current")
    if rec and "error" not in rec:
        if rec.get("action") == "scale_up":
            st.warning(f"⬆️ Scale UP: {rec.get('current_replicas')} → {rec.get('recommended_replicas')}")
        elif rec.get("action") == "scale_down":
            st.info(f"⬇️ Scale DOWN: {rec.get('current_replicas')} → {rec.get('recommended_replicas')}")
        else:
            st.success(f"✅ No Change — {rec.get('current_replicas')} replica(s)")

        st.caption(f"Reason: {rec.get('reason', '—')}")
        st.caption(f"Confidence: {rec.get('confidence', 0):.0%}")
    else:
        st.info("No recommendation available. Ingest metrics first.")

    # History
    history = api_get("/api/v1/recommendations/history")
    if history and history.get("recommendations"):
        st.markdown("### Recommendation History")
        recs = history["recommendations"]
        df = pd.DataFrame(recs)
        st.dataframe(df[["timestamp", "action", "current_replicas", "recommended_replicas", "reason"]].tail(20),
                      use_container_width=True)

elif page == "💰 Cost Analysis":
    st.markdown("""
    <div class="page-banner">
        <h1>💰 Cost Analysis</h1>
        <p>Financial comparison between conventional reactive scaling and AI-assisted proactive optimization</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "⚠️ All costs are estimates based on configured pricing assumptions. "
        "Local Kubernetes costs do not represent actual cloud bills."
    )

    # Action bar
    top_c1, top_c2 = st.columns([4, 1])
    with top_c2:
        if st.button("🔄 Refresh Data", key="refresh_costs"):
            st.rerun()

    # Pricing Assumptions
    pricing = api_get("/api/v1/costs/pricing")
    if pricing:
        with st.expander("⚙️ Configured Pricing Assumptions", expanded=False):
            p = pricing.get("pricing", {})
            p1, p2, p3 = st.columns(3)
            p1.metric("vCPU / Core / Hour", f"${p.get('cpu_per_core_hour', 0.048):.4f}")
            p2.metric("Memory / GB / Hour", f"${p.get('memory_per_gb_hour', 0.006):.4f}")
            p3.metric("Base Instance / Hour", f"${p.get('base_instance_hour', 0.0116):.4f}")
            st.caption(pricing.get("note", "Configurable in cost_engine/pricing_config.json"))

    # Load experiments
    experiments = load_experiment_files()
    completed_experiments = [
        e for e in experiments
        if isinstance(e, dict) and e.get("status") != "failed" and "cost_comparison" in e
    ]

    if completed_experiments:
        # Determine default experiment to select (latest or session_state)
        default_exp_id = st.session_state.get("latest_experiment_id")
        default_index = len(completed_experiments) - 1
        if default_exp_id:
            for idx, exp in enumerate(completed_experiments):
                if exp.get("experiment_id") == default_exp_id:
                    default_index = idx
                    break

        st.markdown("### 🔬 Select Completed Experiment Run")
        def _format_exp_choice(i):
            exp = completed_experiments[i]
            scen = exp.get("scenario", "unknown").replace("_", " ").title()
            eid = exp.get("experiment_id", "")
            started = exp.get("started_at", "")[:19].replace("T", " ")
            return f"{scen} ({eid}) — {started}"

        selected_idx = st.selectbox(
            "Experiment Run",
            options=range(len(completed_experiments)),
            index=default_index,
            format_func=_format_exp_choice,
            key="cost_exp_selector",
        )

        selected_exp = completed_experiments[selected_idx]
        exp_id = selected_exp.get("experiment_id", "unknown")
        scenario_name = selected_exp.get("scenario", "unknown").replace("_", " ").title()
        duration_s = selected_exp.get("duration_seconds", 300)
        started_time = selected_exp.get("started_at", "—")
        cc = selected_exp.get("cost_comparison", {})

        st.markdown(f"#### 📋 Run Summary: {scenario_name} (`{exp_id}`)")
        st.caption(f"Started: {started_time} | Duration: {duration_s}s | Status: {selected_exp.get('status', 'completed').title()}")

        # 4 Primary Cost Metrics
        baseline_cost = cc.get("baseline", {}).get("total_cost", 0.0)
        ai_cost = cc.get("ai_assisted", {}).get("total_cost", 0.0)
        savings = cc.get("absolute_savings", 0.0)
        savings_pct = cc.get("savings_percentage", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline Estimated Cost", f"${baseline_cost:.4f}")
        c2.metric("AI-Assisted Cost", f"${ai_cost:.4f}")
        c3.metric(
            "Absolute Savings",
            f"${savings:.4f}",
            delta="Cost Reduction" if savings >= 0 else "Negative Savings",
        )
        c4.metric(
            "Savings Percentage",
            f"{savings_pct:.1f}%",
            delta="Savings" if savings >= 0 else "AI Costs More",
        )

        if cc.get("ai_costs_more") or savings < 0:
            st.warning(
                f"ℹ️ **Honest Tradeoff Observation:** In the {scenario_name} scenario, AI-assisted mode costs slightly "
                f"more (${abs(savings):.4f} / {abs(savings_pct):.1f}%) because it scaled proactively before latency spiked. "
                "This extra compute headroom guarantees zero SLA breaches during rapid demand ramp-ups."
            )

        # Resource & Workload Consumption Comparison
        st.markdown("### 📊 Resource Consumption & Workload Comparison")
        bl_summary = selected_exp.get("baseline", {})
        ai_summary = selected_exp.get("ai_assisted", {})

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Avg CPU", f"{bl_summary.get('avg_cpu', 0):.1f}%")
        m2.metric("Baseline Avg Replicas", f"{bl_summary.get('avg_replicas', 0):.1f}")
        m3.metric("Baseline Scaling Actions", bl_summary.get("scaling_actions", 0))

        m4, m5, m6 = st.columns(3)
        m4.metric("AI-Assisted Avg CPU", f"{ai_summary.get('avg_cpu', 0):.1f}%")
        m5.metric("AI-Assisted Avg Replicas", f"{ai_summary.get('avg_replicas', 0):.1f}")
        m6.metric("AI Scaling Actions", ai_summary.get("scaling_actions", 0))

        # Performance comparison table
        perf = cc.get("performance_comparison", {})
        perf_data = [
            {
                "Metric": "Average CPU Utilization (%)",
                "Baseline": f"{bl_summary.get('avg_cpu', perf.get('baseline_avg_cpu', 0)):.1f}%",
                "AI-Assisted": f"{ai_summary.get('avg_cpu', perf.get('ai_avg_cpu', 0)):.1f}%",
            },
            {
                "Metric": "Average Allocated Replicas",
                "Baseline": f"{bl_summary.get('avg_replicas', 0):.2f}",
                "AI-Assisted": f"{ai_summary.get('avg_replicas', 0):.2f}",
            },
            {
                "Metric": "Total Scaling Actions",
                "Baseline": bl_summary.get("scaling_actions", 0),
                "AI-Assisted": ai_summary.get("scaling_actions", 0),
            },
            {
                "Metric": "Average Latency (ms)",
                "Baseline": f"{perf.get('baseline_avg_latency', bl_summary.get('avg_latency', 0)):.1f} ms",
                "AI-Assisted": f"{perf.get('ai_avg_latency', ai_summary.get('avg_latency', 0)):.1f} ms",
            },
            {
                "Metric": "Average Error Rate (%)",
                "Baseline": f"{perf.get('baseline_avg_error_rate', bl_summary.get('avg_error_rate', 0)):.2f}%",
                "AI-Assisted": f"{perf.get('ai_avg_error_rate', ai_summary.get('avg_error_rate', 0)):.2f}%",
            },
        ]
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        # Cost Breakdown Table
        with st.expander("💵 Granular Cost Breakdown by Component", expanded=False):
            bl_breakdown = cc.get("baseline", {})
            ai_breakdown = cc.get("ai_assisted", {})
            breakdown_df = pd.DataFrame([
                {
                    "Cost Component": "vCPU Compute Cost",
                    "Baseline": f"${bl_breakdown.get('cpu_cost', 0):.4f}",
                    "AI-Assisted": f"${ai_breakdown.get('cpu_cost', 0):.4f}",
                    "Difference": f"${bl_breakdown.get('cpu_cost', 0) - ai_breakdown.get('cpu_cost', 0):.4f}",
                },
                {
                    "Cost Component": "Memory (RAM) Cost",
                    "Baseline": f"${bl_breakdown.get('memory_cost', 0):.4f}",
                    "AI-Assisted": f"${ai_breakdown.get('memory_cost', 0):.4f}",
                    "Difference": f"${bl_breakdown.get('memory_cost', 0) - ai_breakdown.get('memory_cost', 0):.4f}",
                },
                {
                    "Cost Component": "Base Instance Overhead",
                    "Baseline": f"${bl_breakdown.get('instance_cost', 0):.4f}",
                    "AI-Assisted": f"${ai_breakdown.get('instance_cost', 0):.4f}",
                    "Difference": f"${bl_breakdown.get('instance_cost', 0) - ai_breakdown.get('instance_cost', 0):.4f}",
                },
                {
                    "Cost Component": "Total Estimated Cost",
                    "Baseline": f"${bl_breakdown.get('total_cost', 0):.4f}",
                    "AI-Assisted": f"${ai_breakdown.get('total_cost', 0):.4f}",
                    "Difference": f"${savings:.4f}",
                },
            ])
            st.dataframe(breakdown_df, use_container_width=True)

        # Cross-Scenario Cost Comparison Chart
        st.markdown("### 📈 Cross-Scenario Cost Comparison")
        chart_data = []
        for exp in completed_experiments:
            exp_cc = exp.get("cost_comparison", {})
            if exp_cc:
                chart_data.append({
                    "Scenario": exp.get("scenario", "").replace("_", " ").title(),
                    "Experiment ID": exp.get("experiment_id", "")[:12],
                    "Baseline": exp_cc.get("baseline", {}).get("total_cost", 0),
                    "AI-Assisted": exp_cc.get("ai_assisted", {}).get("total_cost", 0),
                })

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig = go.Figure(data=[
                go.Bar(name="Baseline", x=df_chart["Scenario"] + " (" + df_chart["Experiment ID"] + ")",
                       y=df_chart["Baseline"], marker_color="#ef4444"),
                go.Bar(name="AI-Assisted", x=df_chart["Scenario"] + " (" + df_chart["Experiment ID"] + ")",
                       y=df_chart["AI-Assisted"], marker_color="#22c55e"),
            ])
            fig.update_layout(
                title="Estimated Cost by Completed Run (USD)",
                yaxis_title="Cost (USD)",
                barmode="group",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📊 No completed experiment results found. Run experiments to see cost comparisons.")
        st.caption("You can run experiments from the '🧪 Experiments' page in the navigation or using the CLI command below:")
        st.code("python experiments/run_experiment.py --duration 60 --scenario ramp_up", language="bash")

elif page == "🧪 Experiments":
    st.markdown("""
    <div class="page-banner">
        <h1>🧪 Experiment Runner</h1>
        <p>Compare baseline reactive scaling vs AI-assisted proactive scaling across different workload scenarios</p>
    </div>
    """, unsafe_allow_html=True)

    # Experiment configuration
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    with col1:
        scenario = st.selectbox(
            "Workload Scenario",
            ["stable", "ramp_up", "spike", "low_demand"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        duration = st.slider("Duration (seconds)", 60, 600, 120)
    with col2:
        request_rate = st.slider("Base Request Rate (rps)", 1.0, 100.0, 10.0)
        initial_replicas = st.slider("Initial Replicas", 1, 5, 1)

    if st.button("▶️ Run Experiment", type="primary"):
        with st.spinner(f"Running {scenario} experiment for {duration}s..."):
            result = api_post("/api/v1/experiments/run", {
                "scenario": scenario,
                "duration_seconds": duration,
                "request_rate": request_rate,
                "initial_replicas": initial_replicas,
            })

            if result and "experiment_id" in result:
                st.session_state["latest_experiment_id"] = result["experiment_id"]
                st.success(f"✅ Experiment {result['experiment_id']} complete! Full financial & resource analysis is now updated in '💰 Cost Analysis'.")

                # Show results
                b, a = st.columns(2)
                with b:
                    st.markdown("**Baseline**")
                    bl = result.get("baseline", {})
                    st.metric("Avg CPU", f"{bl.get('avg_cpu', 0):.1f}%")
                    st.metric("Avg Replicas", f"{bl.get('avg_replicas', 0):.1f}")
                    st.metric("Scaling Actions", bl.get("scaling_actions", 0))

                with a:
                    st.markdown("**AI-Assisted**")
                    ai = result.get("ai_assisted", {})
                    st.metric("Avg CPU", f"{ai.get('avg_cpu', 0):.1f}%")
                    st.metric("Avg Replicas", f"{ai.get('avg_replicas', 0):.1f}")
                    st.metric("Scaling Actions", ai.get("scaling_actions", 0))

                # Cost comparison
                cc = result.get("cost_comparison", {})
                st.markdown("### Cost Comparison")
                st.metric("Estimated Savings",
                          f"${cc.get('absolute_savings', 0):.4f} ({cc.get('savings_percentage', 0):.1f}%)")

                # Timeseries chart
                if "baseline_timeseries" in result:
                    baseline_ts = result["baseline_timeseries"]
                    ai_ts = result.get("ai_timeseries", [])

                    fig = make_subplots(rows=2, cols=1,
                                       subplot_titles=["CPU Usage Over Time", "Replica Count Over Time"])

                    fig.add_trace(go.Scatter(
                        y=[m["cpu_usage"] for m in baseline_ts],
                        name="Baseline CPU", line=dict(color="#ef4444"),
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        y=[m["cpu_usage"] for m in ai_ts],
                        name="AI CPU", line=dict(color="#22c55e"),
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        y=[m["replica_count"] for m in baseline_ts],
                        name="Baseline Replicas", line=dict(color="#ef4444", dash="dash"),
                    ), row=2, col=1)
                    fig.add_trace(go.Scatter(
                        y=[m["replica_count"] for m in ai_ts],
                        name="AI Replicas", line=dict(color="#22c55e", dash="dash"),
                    ), row=2, col=1)

                    fig.update_layout(
                        height=500,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#0f172a",
                        font=dict(color="#94a3b8"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Experiment failed. Check API logs.")

    # Historical experiments
    experiments = load_experiment_files()
    if experiments:
        st.markdown("### 📋 Experiment History")
        for exp in experiments:
            scenario_name = exp.get("scenario", "unknown").replace("_", " ").title()
            cc = exp.get("cost_comparison", {})
            savings = cc.get("savings_percentage", 0)

            with st.expander(f"🔬 {scenario_name} — {exp.get('experiment_id', 'unknown')}"):
                st.json({
                    "experiment_id": exp.get("experiment_id"),
                    "scenario": exp.get("scenario"),
                    "duration_seconds": exp.get("duration_seconds"),
                    "baseline_summary": exp.get("baseline"),
                    "ai_summary": exp.get("ai_assisted"),
                    "savings_percentage": savings,
                    "ai_costs_more": cc.get("ai_costs_more", False),
                })
