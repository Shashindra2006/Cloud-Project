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
    page_title="Cloud Cost Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Custom CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #e2e8f0;
        margin-bottom: 10px;
    }

    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    .metric-card .delta {
        font-size: 0.8rem;
        color: #22c55e;
        margin-top: 4px;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-healthy { background: #064e3b; color: #34d399; }
    .status-warning { background: #78350f; color: #fbbf24; }
    .status-error { background: #7f1d1d; color: #f87171; }

    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-bottom: 1px solid #334155;
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

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
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
        return default
    except requests.exceptions.ConnectionError:
        return default
    except Exception:
        return default


def load_experiment_files():
    """Load experiment results from disk."""
    results = []
    if EXPERIMENTS_DIR.exists():
        for f in sorted(EXPERIMENTS_DIR.glob("*.json")):
            if f.name == "all_experiments.json":
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            results.extend(data)
                        continue
                except Exception:
                    continue
            try:
                with open(f) as fh:
                    results.append(json.load(fh))
            except Exception:
                pass
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


# ============ Sidebar ============

with st.sidebar:
    st.markdown("## ☁️ Cloud Cost Optimizer")
    st.markdown("---")

    # API connection status
    health = api_get("/api/v1/health")
    if health:
        st.success("🟢 API Connected")
        st.caption(f"DB: {health.get('database', 'unknown')}")
        st.caption(f"ML: {health.get('ml_model', 'unknown')}")
        st.caption(f"Anomaly: {health.get('anomaly_model', 'unknown')}")
    else:
        st.warning("🟡 API Offline — Using stored data")
        st.caption("Start the API: `python -m uvicorn app.main:app`")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔮 Predictions", "🚨 Anomalies",
         "⚡ Optimizer", "💰 Cost Analysis", "🧪 Experiments"],
        index=0,
    )

    st.markdown("---")
    st.caption("v1.0.0 | AI-Powered Cloud Cost Optimizer")
    st.caption(f"⏱️ {datetime.datetime.now().strftime('%H:%M:%S')}")


# ============ Pages ============

if page == "📊 Overview":
    st.markdown("""
    <div class="header-container">
        <h1 style="color: #f1f5f9; margin: 0;">☁️ Cloud Cost & Resource Optimization</h1>
        <p style="color: #94a3b8; margin: 4px 0 0 0;">AI-powered monitoring, prediction, and optimization platform</p>
    </div>
    """, unsafe_allow_html=True)

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
        st.info("📡 No live metrics. Ingest metrics via the API or run the workload simulator.")

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
    st.markdown("## 🔮 Demand Predictions")

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
    st.markdown("## 🚨 Anomaly Detection")
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
    st.markdown("## ⚡ Scaling Optimizer")

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
    st.markdown("## 💰 Cost Analysis")
    st.caption(
        "⚠️ All costs are estimates based on configured pricing assumptions. "
        "Local Kubernetes costs do not represent actual cloud bills."
    )

    # Pricing
    pricing = api_get("/api/v1/costs/pricing")
    if pricing:
        st.markdown("### Pricing Assumptions")
        p = pricing.get("pricing", {})
        p1, p2, p3 = st.columns(3)
        p1.metric("CPU/Core/Hour", f"${p.get('cpu_per_core_hour', 0):.4f}")
        p2.metric("Memory/GB/Hour", f"${p.get('memory_per_gb_hour', 0):.4f}")
        p3.metric("Base Instance/Hour", f"${p.get('base_instance_hour', 0):.4f}")

    # Experiment cost comparison
    experiments = load_experiment_files()
    if experiments:
        st.markdown("### 📊 Cost Comparison — Baseline vs AI-Assisted")

        for exp in experiments:
            cc = exp.get("cost_comparison", {})
            if not cc:
                continue

            scenario = exp.get("scenario", "unknown").replace("_", " ").title()
            st.markdown(f"#### {scenario}")

            col1, col2, col3, col4 = st.columns(4)
            baseline_cost = cc.get("baseline", {}).get("total_cost", 0)
            ai_cost = cc.get("ai_assisted", {}).get("total_cost", 0)
            savings = cc.get("absolute_savings", 0)
            savings_pct = cc.get("savings_percentage", 0)

            col1.metric("Baseline Cost", f"${baseline_cost:.4f}")
            col2.metric("AI-Assisted Cost", f"${ai_cost:.4f}")
            col3.metric("Savings", f"${savings:.4f}")
            col4.metric("Savings %", f"{savings_pct:.1f}%",
                        delta="Positive" if savings > 0 else "Negative (AI costs more)")

            if cc.get("ai_costs_more"):
                st.warning(f"⚠️ In the {scenario} scenario, AI-assisted mode costs MORE than baseline.")

            # Performance comparison
            perf = cc.get("performance_comparison", {})
            if perf:
                st.markdown("**Performance Comparison**")
                perf_df = pd.DataFrame([
                    {"Metric": "Avg Latency (ms)",
                     "Baseline": perf.get("baseline_avg_latency", 0),
                     "AI-Assisted": perf.get("ai_avg_latency", 0)},
                    {"Metric": "Avg Error Rate (%)",
                     "Baseline": perf.get("baseline_avg_error_rate", 0),
                     "AI-Assisted": perf.get("ai_avg_error_rate", 0)},
                    {"Metric": "Avg CPU (%)",
                     "Baseline": perf.get("baseline_avg_cpu", 0),
                     "AI-Assisted": perf.get("ai_avg_cpu", 0)},
                ])
                st.dataframe(perf_df, use_container_width=True)

        # Cost comparison bar chart
        if len(experiments) > 0:
            chart_data = []
            for exp in experiments:
                cc = exp.get("cost_comparison", {})
                if cc:
                    chart_data.append({
                        "Scenario": exp.get("scenario", "").replace("_", " ").title(),
                        "Baseline": cc.get("baseline", {}).get("total_cost", 0),
                        "AI-Assisted": cc.get("ai_assisted", {}).get("total_cost", 0),
                    })

            if chart_data:
                df = pd.DataFrame(chart_data)
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=df["Scenario"], y=df["Baseline"],
                           marker_color="#ef4444"),
                    go.Bar(name="AI-Assisted", x=df["Scenario"], y=df["AI-Assisted"],
                           marker_color="#22c55e"),
                ])
                fig.update_layout(
                    title="Estimated Cost Comparison by Scenario",
                    yaxis_title="Cost (USD)",
                    barmode="group",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No experiment results found. Run experiments to see cost comparisons.")
        st.code("python -m experiments.run_experiment", language="bash")

elif page == "🧪 Experiments":
    st.markdown("## 🧪 Experiment Runner")

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
                st.success(f"✅ Experiment {result['experiment_id']} complete!")

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
