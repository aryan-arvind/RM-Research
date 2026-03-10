import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
VIS_DIR = OUTPUTS_DIR / "visualizations"
RESULTS_PATH = EVAL_DIR / "results.json"


def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def summary_table(results: dict) -> list[dict]:
    rows = []
    summary = results.get("summary", {}).get("per_corruption", {})
    for name, item in summary.items():
        curves = item.get("severity_curves", {})
        recoveries = [v.get("recovery_rate", 0.0) for v in curves.values()]
        drops = [v.get("det_drop_pct", 0.0) for v in curves.values()]
        rows.append(
            {
                "Corruption": name,
                "Diagnosis Accuracy": round(100 * item.get("diagnosis_accuracy", 0.0), 1),
                "Avg Detection Drop %": round(sum(drops) / max(len(drops), 1), 1),
                "Best Recovery %": round(max(recoveries) if recoveries else 0.0, 1),
            }
        )
    return rows


def run_pipeline(max_images: int, epochs: int, batch_size: int) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "full_pipeline",
        "--image_dir",
        "./datasets/eurosat/eurosat/2750",
        "--image_size",
        "64",
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--max_images",
        str(max_images),
        "--num_workers",
        "0",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, output


def open_folder(path: Path):
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", str(path)])


def metric_card(title: str, value: str, hint: str = ""):
    st.markdown(
        (
            "<div class='kpi-card'>"
            f"<div class='kpi-title'>{title}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-hint'>{hint}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def verdict_text(best_recovery: float) -> tuple[str, str]:
    if best_recovery >= 30:
        return "Good MVP", "The model shows measurable recovery on at least one corruption."
    if best_recovery >= 5:
        return "Baseline MVP", "Pipeline works; recovery is modest and needs tuning."
    return "Early MVP", "Pipeline works end-to-end; quality improvements are next-step work."


def render_simple_table(rows: list[dict]):
    if not rows:
        st.info("No summary rows to display.")
        return

    headers = ["Corruption", "Diagnosis Accuracy", "Avg Detection Drop %", "Best Recovery %"]
    html = [
        "<div style='overflow-x:auto;border:1px solid #d6e2dd;border-radius:12px;background:#fff;'>",
        "<table style='width:100%;border-collapse:collapse;font-size:0.95rem;'>",
        "<thead><tr style='background:#f3f7f5;'>",
    ]
    for h in headers:
        html.append(
            f"<th style='text-align:left;padding:10px 12px;border-bottom:1px solid #d6e2dd;color:#0f172a;'>{h}</th>"
        )
    html.append("</tr></thead><tbody>")

    for r in rows:
        html.append("<tr>")
        html.append(f"<td style='padding:10px 12px;border-bottom:1px solid #eef2f1;'>{r['Corruption']}</td>")
        html.append(f"<td style='padding:10px 12px;border-bottom:1px solid #eef2f1;'>{r['Diagnosis Accuracy']:.1f}%</td>")
        html.append(f"<td style='padding:10px 12px;border-bottom:1px solid #eef2f1;'>{r['Avg Detection Drop %']:.1f}%</td>")
        html.append(f"<td style='padding:10px 12px;border-bottom:1px solid #eef2f1;'>{r['Best Recovery %']:.1f}%</td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


st.set_page_config(page_title="Satellite Preprocessing Research Dashboard", page_icon="ST", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #eef7ff 0%, #f7f7f4 45%, #f2f8ef 100%);
        color: #0b1220 !important;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div, th, td {
        color: #0b1220 !important;
    }
    [data-testid="stMarkdownContainer"] * {
        color: #0b1220 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #334155 !important;
    }
    [data-testid="stMetricValue"] * {
        color: #0b1220 !important;
    }
    [data-testid="stNumberInput"] input {
        color: #0b1220 !important;
        background: #ffffff !important;
    }
    [data-testid="stNumberInput"] button {
        color: #0b1220 !important;
    }
    .stButton > button {
        border: 1px solid #b7c4bf !important;
        color: #0b1220 !important;
        background: #ffffff !important;
    }
    .stButton > button[kind="primary"] {
        background: #ef4444 !important;
        color: #ffffff !important;
        border-color: #ef4444 !important;
    }
    .hero {
        background: #0f172a;
        color: #ffffff !important;
        border-radius: 14px;
        padding: 20px 24px;
        border: 1px solid #1f2937;
        margin-bottom: 16px;
    }
    .hero h1, .hero h2, .hero h3, .hero p, .hero span, .hero div {
        color: #ffffff !important;
    }
    .hero-sub {
        color: #cbd5e1 !important;
        margin-top: 6px;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #dbe5e1;
        border-radius: 12px;
        padding: 14px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #d6e2dd;
        border-left: 6px solid #0f766e;
        border-radius: 12px;
        padding: 12px 14px;
        min-height: 110px;
    }
    .kpi-title {
        color: #334155;
        font-size: 0.95rem;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .kpi-hint {
        color: #64748b;
        font-size: 0.85rem;
    }
    code {
        color: #0f172a !important;
        background: #e8f0ec !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
    <h1 style='margin:0;'>Adaptive Preprocessing Hardening: Research Dashboard</h1>
      <div class='hero-sub'>Simple demo view for faculty: run, read, and present outputs in one place.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2.2, 1])
with left:
    st.markdown("### 1) Run Experiment Pipeline")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_images = st.number_input("Max Images", min_value=5, max_value=100, value=10, step=5)
    with c2:
        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=1, step=1)
    with c3:
        batch_size = st.number_input("Batch Size", min_value=2, max_value=16, value=4, step=1)

    if st.button("Run Full Pipeline", type="primary"):
        with st.spinner("Running pipeline. This may take several minutes..."):
            ok, logs = run_pipeline(max_images=max_images, epochs=epochs, batch_size=batch_size)
        if ok:
            st.success("Pipeline completed successfully. Outputs refreshed.")
        else:
            st.error("Pipeline failed. Check logs below.")
        with st.expander("Execution Logs", expanded=not ok):
            st.code(logs, language="text")

with right:
    st.markdown("### 2) Output Folders")
    if st.button("Open Evaluation Folder"):
        open_folder(EVAL_DIR)
    if st.button("Open Visualizations Folder"):
        open_folder(VIS_DIR)
    st.caption("Evaluation path")
    st.code(str(EVAL_DIR), language="text")
    st.caption("Visualization path")
    st.code(str(VIS_DIR), language="text")

results = load_results()

st.markdown("### 3) Results Snapshot")
if results is None:
    st.warning("No results found yet. Run the MVP pipeline first.")
else:
    clean = results.get("summary", {}).get("clean_avg_detections", 0.0)
    table = summary_table(results)

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Clean Avg Detections", f"{clean:.2f}", "Higher means YOLO finds more objects on clean images.")
    with m2:
        best_diag = max(table, key=lambda x: x["Diagnosis Accuracy"]) if table else None
        label = best_diag["Corruption"] if best_diag else "N/A"
        score = best_diag["Diagnosis Accuracy"] if best_diag else 0.0
        metric_card("Best Diagnosis", f"{label} ({score:.1f}%)", "Most confidently identified corruption type.")
    with m3:
        best_rec = max((row["Best Recovery %"] for row in table), default=0.0)
        metric_card("Best Recovery", f"{best_rec:.1f}%", "Best gain after hardening vs corrupted input.")

    status, note = verdict_text(best_rec)
    st.info(f"Experiment Status: {status}. {note}")

    render_simple_table(table)

st.markdown("### 4) Visual Outputs")
tab1, tab2 = st.tabs(["Evaluation Charts", "Corruption Samples"])

with tab1:
    fig1, fig2 = st.columns(2)
    with fig1:
        dashboard = EVAL_DIR / "dashboard.png"
        if dashboard.exists():
            st.image(str(dashboard), caption="Evaluation Dashboard", use_column_width=True)
        else:
            st.warning("Missing: outputs/evaluation/dashboard.png")
    with fig2:
        stress = EVAL_DIR / "stress_curves.png"
        if stress.exists():
            st.image(str(stress), caption="Stress Curves", use_column_width=True)
        else:
            st.warning("Missing: outputs/evaluation/stress_curves.png")

with tab2:
    if VIS_DIR.exists():
        sample_images = sorted(VIS_DIR.glob("*.png"))[:9]
        cols = st.columns(3)
        for idx, img_path in enumerate(sample_images):
            cols[idx % 3].image(str(img_path), caption=img_path.name, use_column_width=True)
    else:
        st.info("No visualization images found yet.")

st.markdown("---")
st.caption("Start dashboard: streamlit run demo_ui.py")
