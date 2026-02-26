from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Optional, List, Tuple

# Flat repo imports (no core/ folder)
try:
    from validation import validate_and_clean
    from peer_groups import assign_peer_groups
    from modeling_gap import adjusted_gap_ols
    from modeling_outliers import score_outliers
    from reporting import outlier_budget
    from exports import export_excel_pack
except ImportError:  # pragma: no cover
    from .validation import validate_and_clean
    from .peer_groups import assign_peer_groups
    from .modeling_gap import adjusted_gap_ols
    from .modeling_outliers import score_outliers
    from .reporting import outlier_budget
    from .exports import export_excel_pack


# ----------------------------
# Branding
# ----------------------------
PRIMARY = "#0074C7"
BLACK = "#000000"
WHITE = "#FFFFFF"
BG = "#F7F9FC"
CARD = "#FFFFFF"
BORDER = "#D9E2EC"
MUTED = "#6B7280"
LIGHT_BLUE = "#EAF4FF"
GREEN = "#15803D"
AMBER = "#B45309"
RED = "#B91C1C"


# ----------------------------
# Page config + styling
# ----------------------------
st.set_page_config(page_title="Global Pay Auditor", layout="wide")
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG}; }}
        .block-container {{ padding-top: 1.0rem; padding-bottom: 2rem; }}
        .brand-title {{ font-size: 1.95rem; font-weight: 800; color: {BLACK}; margin-bottom: 0.1rem; }}
        .brand-subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 0.8rem; }}
        .section-title {{ font-size: 1.1rem; font-weight: 700; color: {BLACK}; margin: 0.1rem 0 0.7rem 0; }}
        .panel {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 14px 16px 12px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            margin-bottom: 0.9rem;
        }}
        .kpi-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-left: 6px solid {PRIMARY};
            border-radius: 16px;
            padding: 14px 16px;
            min-height: 92px;
        }}
        .kpi-label {{ color: {MUTED}; font-size: 0.82rem; margin-bottom: 0.2rem; }}
        .kpi-value {{ color: {BLACK}; font-size: 1.45rem; font-weight: 800; line-height: 1.2; }}
        .kpi-note {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.2rem; }}
        .pill {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid {BORDER};
            margin-right: 6px;
            margin-bottom: 6px;
            background: {WHITE};
            color: {BLACK};
        }}
        .insight-box {{
            background: {LIGHT_BLUE};
            border: 1px solid {PRIMARY};
            border-radius: 16px;
            padding: 12px 14px;
            color: {BLACK};
            font-size: 0.92rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
def draw_section_header(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_num(x: Any, decimals: int = 0) -> str:
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return str(x)


def fmt_pct(x: Any, decimals: int = 1) -> str:
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):,.{decimals}f}%"
    except Exception:
        return str(x)


def read_file(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    raise ValueError("Unsupported file type")


def normalize_validate_output(result) -> Tuple[pd.DataFrame, Any]:
    if isinstance(result, tuple):
        df_clean = result[0]
        report = result[1] if len(result) > 1 else None
        return df_clean, report
    return result, None


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalize_adjusted_output(model_output: dict[str, Any]) -> Tuple[pd.DataFrame, Optional[str], Optional[int], Optional[float]]:
    effects_df = pd.DataFrame(model_output.get("effects", [])) if isinstance(model_output, dict) else pd.DataFrame()
    summary_text = model_output.get("model_summary_text") if isinstance(model_output, dict) else None
    n = model_output.get("n") if isinstance(model_output, dict) else None
    r2 = model_output.get("r2") if isinstance(model_output, dict) else None
    return effects_df, summary_text, n, r2


def make_bar(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: str = ""):
    fig = px.bar(df, x=x, y=y, color=color, text_auto=True)
    fig.update_layout(
        title=title,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font_color=BLACK,
        showlegend=bool(color),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_heatmap(df: pd.DataFrame, idx: str, col: str, val: str, title: str):
    pivot = pd.pivot_table(df, index=idx, columns=col, values=val, aggfunc="sum", fill_value=0)
    if pivot.empty:
        return None
    fig = px.imshow(pivot, text_auto=True, aspect="auto")
    fig.update_layout(
        title=title,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font_color=BLACK,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str], hover_cols: List[str], title: str):
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover_cols)
    x_min, x_max = df[x].min(), df[x].max()
    if pd.notna(x_min) and pd.notna(x_max):
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[x_min, x_max], mode="lines", name="Actual = Expected"))
    fig.update_layout(
        title=title,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font_color=BLACK,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def infer_rag(gap_pct: float, budget: float, outlier_rate: float) -> str:
    gap_pct = 0 if pd.isna(gap_pct) else float(gap_pct)
    budget = 0 if pd.isna(budget) else float(budget)
    outlier_rate = 0 if pd.isna(outlier_rate) else float(outlier_rate)
    if gap_pct <= -5 or budget > 0 or outlier_rate >= 0.15:
        return "Red"
    if gap_pct <= -2 or outlier_rate >= 0.08:
        return "Amber"
    return "Green"


def status_color(status: str) -> str:
    return {"Green": GREEN, "Amber": AMBER, "Red": RED}.get(status, BLACK)


def build_expected_pay(df: pd.DataFrame, pay_col: str, protected_col: str, comparator_cols: List[str]) -> pd.DataFrame:
    """Create peer benchmark excluding the protected attribute from the comparator set."""
    work = df.copy()
    work[pay_col] = pd.to_numeric(work[pay_col], errors="coerce")
    work = work.dropna(subset=[pay_col]).copy()

    comparator_cols = [c for c in comparator_cols if c in work.columns and c != protected_col]
    if comparator_cols:
        work["pred_pay"] = work.groupby(comparator_cols, dropna=False)[pay_col].transform("mean")
    else:
        work["pred_pay"] = float(work[pay_col].mean()) if len(work) else np.nan

    work["gap_pct_emp"] = np.where(
        work["pred_pay"].ne(0),
        ((work[pay_col] - work["pred_pay"]) / work["pred_pay"]) * 100,
        np.nan,
    )
    return work


def build_unadjusted_summary(
    df: pd.DataFrame,
    pay_col: str,
    protected_col: str,
    comparator_cols: List[str],
    underpaid_threshold_pct: int,
    min_n: int,
) -> pd.DataFrame:
    """Group-level summary using peer benchmark excluding the protected attribute."""
    if pay_col not in df.columns or protected_col not in df.columns or "pred_pay" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    by = [c for c in comparator_cols if c in work.columns] + [protected_col]
    if not by:
        work["scope"] = "All filtered employees"
        by = ["scope"]

    low_factor = 1 + underpaid_threshold_pct / 100.0  # -10 => 0.90
    high_factor = 1 - underpaid_threshold_pct / 100.0 # -10 => 1.10

    work["min_acceptable_pay"] = work["pred_pay"] * low_factor
    work["max_expected_pay"] = work["pred_pay"] * high_factor
    work["underpaid_amt"] = np.maximum(work["min_acceptable_pay"] - work[pay_col], 0)
    work["overpaid_amt"] = np.maximum(work[pay_col] - work["max_expected_pay"], 0)
    work["flag_underpaid"] = work["gap_pct_emp"] < underpaid_threshold_pct

    summary = (
        work.groupby(by, dropna=False)
        .agg(
            n=(pay_col, "size"),
            actual_mean=(pay_col, "mean"),
            pred_mean=("pred_pay", "mean"),
            median_pay=(pay_col, "median"),
            flagged_count=("flag_underpaid", "sum"),
            underpaid_budget=("underpaid_amt", "sum"),
            overpaid_budget=("overpaid_amt", "sum"),
        )
        .reset_index()
    )

    summary["unadj_gap_pct"] = np.where(
        summary["pred_mean"].ne(0),
        ((summary["actual_mean"] - summary["pred_mean"]) / summary["pred_mean"]) * 100,
        np.nan,
    )
    summary["adj_gap_pct"] = np.nan
    summary["meets_min_n"] = summary["n"] >= int(min_n)
    summary["group_key"] = summary.apply(lambda r: " | ".join([f"{c}={r[c]}" for c in by]), axis=1)

    order = by + [
        "group_key", "n", "actual_mean", "pred_mean", "median_pay", "unadj_gap_pct",
        "adj_gap_pct", "flagged_count", "underpaid_budget", "overpaid_budget", "meets_min_n"
    ]
    return summary[order].sort_values(by).reset_index(drop=True)


# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="brand-title">Pay Equity Auditor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-subtitle">Branded compensation diagnostics dashboard with validation, peer benchmarking, pay gap review, outlier detection, and export pack.</div>',
    unsafe_allow_html=True,
)


# ----------------------------
# Upload
# ----------------------------
with st.container():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload compensation snapshot (CSV / XLSX)", type=["csv", "xlsx"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    draw_section_header("Recommended flow")
    st.markdown(
        """
        1. **Upload and validate** the compensation snapshot  
        2. **Select pay measure and protected attribute**  
        3. **Apply scope filters** for country, grade, and job family  
        4. **Review KPI cards and hotspot visuals**  
        5. **Inspect group summary, regression results, and outliers**  
        6. **Export the Excel pack and tables**
        """
    )
    st.stop()

try:
    df_raw = read_file(uploaded)
except Exception as e:
    st.error(f"File read failed: {e}")
    st.stop()

raw_rows = len(df_raw)

try:
    validated = validate_and_clean(df_raw)
    df, validation_report = normalize_validate_output(validated)
except Exception as e:
    st.error(f"Validation failed: {e}")
    st.stop()

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Validated dataset is empty.")
    st.stop()

clean_rows = len(df)
rows_excluded = raw_rows - clean_rows
usability_pct = (clean_rows / raw_rows * 100) if raw_rows else np.nan

# Column detection
country_col = first_existing(df, ["country"])
grade_col = first_existing(df, ["grade"])
job_family_col = first_existing(df, ["job_family"])
employee_col = first_existing(df, ["employee_id"])
pay_default = first_existing(df, ["base_pay_annual", "total_cash_annual", "base_pay"])
protected_default = first_existing(df, ["gender", "nationality", "ethnicity", "expat_local"])

# Sidebar controls
st.sidebar.markdown("## Analysis Controls")
pay_col = st.sidebar.selectbox(
    "Pay measure",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(pay_default) if pay_default in df.columns else 0,
)

protected_options = [c for c in df.columns if str(df[c].dtype) == "object" or str(df[c].dtype).startswith("category")]
if not protected_options:
    protected_options = df.columns.tolist()
protected_col = st.sidebar.selectbox(
    "Protected attribute",
    options=protected_options,
    index=protected_options.index(protected_default) if protected_default in protected_options else 0,
)

threshold = st.sidebar.slider("Underpaid threshold (%) vs expected", min_value=-25, max_value=-1, value=-10, step=1)
min_n = st.sidebar.slider("Minimum sample size", min_value=2, max_value=15, value=5, step=1)

filtered = df.copy()
filter_summary: List[Tuple[str, int]] = []
for label, col_name in [("Country", country_col), ("Grade", grade_col), ("Job Family", job_family_col)]:
    if col_name and col_name in filtered.columns:
        values = sorted(filtered[col_name].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, options=values, default=values)
        if selected:
            filtered = filtered[filtered[col_name].astype(str).isin(selected)].copy()
            filter_summary.append((label, len(selected)))

if filtered.empty:
    st.warning("No rows remain after applying filters.")
    st.stop()

# Scored datasets
filtered = ensure_numeric(filtered, [pay_col])
filtered = filtered.dropna(subset=[pay_col]).copy()
comparator_cols = [c for c in [country_col, grade_col, job_family_col] if c and c != protected_col]
scored = build_expected_pay(filtered, pay_col=pay_col, protected_col=protected_col, comparator_cols=comparator_cols)
unadj = build_unadjusted_summary(scored, pay_col, protected_col, comparator_cols, threshold, min_n)

# Adjusted model
controls = [c for c in ["tenure_years", "time_in_grade_years", "performance_rating", grade_col, job_family_col, country_col] if c]
adj_effects = pd.DataFrame()
adj_text = None
adj_n = None
adj_r2 = None
adjusted_error = None
try:
    model_output = adjusted_gap_ols(
        df=scored,
        pay_col=pay_col,
        protected_col=protected_col,
        controls=controls,
        min_n=max(30, min_n),
    )
    adj_effects, adj_text, adj_n, adj_r2 = normalize_adjusted_output(model_output)
except Exception as e:
    adjusted_error = str(e)

# Outlier scoring
outlier_df = pd.DataFrame()
outlier_summary = pd.DataFrame()
outlier_error = None
try:
    df_pg = assign_peer_groups(scored) if "peer_group" not in scored.columns else scored.copy()
    out = score_outliers(df_pg, pay_col=pay_col, group_cols=["peer_group"] if "peer_group" in df_pg.columns else comparator_cols)
    outlier_df = out.get("flagged", pd.DataFrame()).copy()
    outlier_summary = out.get("summary", pd.DataFrame()).copy()
    if not outlier_df.empty and employee_col and employee_col in outlier_df.columns and employee_col in scored.columns:
        extra = scored[[employee_col, "pred_pay", "gap_pct_emp"]].drop_duplicates()
        outlier_df = outlier_df.merge(extra, on=employee_col, how="left")
except Exception as e:
    outlier_error = str(e)

# KPI values
flagged_emp_count = int((scored["gap_pct_emp"] < threshold).fillna(False).sum()) if "gap_pct_emp" in scored.columns else 0
estimated_budget = float(unadj["underpaid_budget"].sum()) if not unadj.empty else 0.0
overall_gap = np.nan
if "pred_pay" in scored.columns and scored["pred_pay"].notna().any():
    p = scored["pred_pay"].mean()
    a = scored[pay_col].mean()
    overall_gap = ((a - p) / p * 100) if p else np.nan
outlier_rate = float(outlier_df["outlier_flag"].mean()) if (not outlier_df.empty and "outlier_flag" in outlier_df.columns) else np.nan
status = infer_rag(overall_gap, estimated_budget, outlier_rate)

# Insight text
insight_lines = [
    f"{len(scored):,} employees analyzed after validation and filters.",
    f"Protected attribute selected: {protected_col}.",
    f"Pay measure selected: {pay_col}.",
    f"Overall unadjusted gap vs peer benchmark: {fmt_pct(overall_gap)}.",
    f"Employees below threshold ({threshold}%): {flagged_emp_count:,}.",
    f"Estimated underpaid budget: {fmt_num(estimated_budget)}.",
]
if adjusted_error:
    insight_lines.append(f"Adjusted model note: {adjusted_error}")
if outlier_error:
    insight_lines.append(f"Outlier note: {outlier_error}")

# Scope/status strip
st.markdown('<div class="panel">', unsafe_allow_html=True)
left, right = st.columns([2.2, 1])
with left:
    draw_section_header("Current scope")
    pills = "".join([f'<span class="pill">{name}: {count} selected</span>' for name, count in filter_summary]) or '<span class="pill">No scope filters applied</span>'
    st.markdown(pills, unsafe_allow_html=True)
with right:
    draw_section_header("Status")
    st.markdown(f'<span class="pill" style="border-color:{status_color(status)}; color:{status_color(status)};">Overall Equity Status: {status}</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Data Quality",
    "Pay Gap Analysis",
    "Outliers & Budget",
    "Methodology & Export",
])

with tab1:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Employees Analyzed", fmt_num(len(scored)), "After validation + filters")
    with k2:
        kpi_card("Overall Gap %", fmt_pct(overall_gap), "Actual mean vs expected peer mean")
    with k3:
        kpi_card("Flagged Employees", fmt_num(flagged_emp_count), f"Below {threshold}% threshold")
    with k4:
        kpi_card("Estimated Budget", fmt_num(estimated_budget), "Potential remediation exposure")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        kpi_card("Protected Attribute", protected_col, "Selected lens")
    with k6:
        kpi_card("Pay Measure", pay_col, "Selected metric")
    with k7:
        kpi_card("Outlier Rate", fmt_pct(outlier_rate * 100 if pd.notna(outlier_rate) else np.nan), "Outlier flag rate")
    with k8:
        kpi_card("Data Usability", fmt_pct(usability_pct), "Clean rows / uploaded rows")

    st.markdown('<div class="insight-box">' + "<br>".join([f"• {x}" for x in insight_lines]) + '</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Top groups by underpaid budget")
        if not unadj.empty:
            top_budget = unadj.sort_values("underpaid_budget", ascending=False).head(10)
            fig = make_bar(top_budget, x="group_key", y="underpaid_budget", title="Highest remediation exposure")
            fig.update_xaxes(title="Peer group")
            fig.update_yaxes(title="Underpaid budget")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No unadjusted summary available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Population mix")
        if protected_col in scored.columns:
            mix = scored[protected_col].astype(str).value_counts().reset_index()
            mix.columns = [protected_col, "count"]
            fig = px.pie(mix, names=protected_col, values="count", hole=0.45)
            fig.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE, font_color=BLACK, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Protected attribute not available for mix chart.")
        st.markdown('</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    with h1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Hotspot heatmap")
        if not unadj.empty and grade_col and job_family_col and grade_col in unadj.columns and job_family_col in unadj.columns:
            fig = make_heatmap(unadj, idx=grade_col, col=job_family_col, val="underpaid_budget", title="Underpaid budget by Grade × Job Family")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for heatmap.")
        else:
            st.info("Heatmap requires grade and job family columns.")
        st.markdown('</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Top negative gap groups")
        if not unadj.empty:
            top_gap = unadj.sort_values("unadj_gap_pct").head(10)
            fig = make_bar(top_gap, x="group_key", y="unadj_gap_pct", title="Most negative unadjusted gaps")
            fig.update_xaxes(title="Peer group")
            fig.update_yaxes(title="Gap %")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gap summary available.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        kpi_card("Uploaded Rows", fmt_num(raw_rows), "Rows in input file")
    with q2:
        kpi_card("Clean Rows", fmt_num(clean_rows), "Rows after validation")
    with q3:
        kpi_card("Rows Excluded", fmt_num(rows_excluded), "Potential quality drops")
    with q4:
        kpi_card("Columns", fmt_num(len(df.columns)), "Fields available for analysis")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Missing values by field")
        missing_df = df.isna().sum().reset_index()
        missing_df.columns = ["field", "missing_count"]
        missing_df = missing_df.sort_values("missing_count", ascending=False).head(15)
        fig = make_bar(missing_df, x="field", y="missing_count", title="Top missing fields")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Validation report")
        if validation_report is None:
            st.success("No validation warnings returned by the validation module.")
        elif isinstance(validation_report, dict):
            st.json(validation_report)
        else:
            st.write(validation_report)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Data preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Actual vs expected pay")
        if not scored.empty and "pred_pay" in scored.columns:
            hover_cols = [c for c in [employee_col, country_col, grade_col, job_family_col, protected_col] if c]
            fig = make_scatter(scored.dropna(subset=[pay_col, "pred_pay"]), x="pred_pay", y=pay_col, color=protected_col if protected_col in scored.columns else None, hover_cols=hover_cols, title="Employee-level actual vs expected")
            fig.update_xaxes(title="Expected pay")
            fig.update_yaxes(title="Actual pay")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Expected pay could not be derived.")
        st.markdown('</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Unadjusted group comparison")
        if not unadj.empty and protected_col in unadj.columns:
            compare = unadj[[protected_col, "actual_mean", "pred_mean"]].copy()
            compare = compare.groupby(protected_col, dropna=False).mean(numeric_only=True).reset_index()
            fig = px.bar(compare.melt(id_vars=[protected_col], value_vars=["actual_mean", "pred_mean"]), x=protected_col, y="value", color="variable", barmode="group")
            fig.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE, font_color=BLACK, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unadjusted summary not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Unadjusted Pay (by group)")
    if not unadj.empty:
        st.dataframe(unadj, use_container_width=True)
    else:
        st.info("No unadjusted summary available.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Adjusted pay gap / regression audit")
    if adjusted_error:
        st.warning(adjusted_error)
    else:
        st.success(f"Model fitted. n={adj_n} | R²={adj_r2:.3f}" if adj_n is not None and adj_r2 is not None else "Model fitted.")
        if not adj_effects.empty:
            st.dataframe(adj_effects, use_container_width=True)
        if adj_text:
            with st.expander("Model summary"):
                st.text(adj_text)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    o1, o2 = st.columns(2)
    with o1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Budget by group")
        if not unadj.empty:
            budget_view = unadj.sort_values("underpaid_budget", ascending=False).head(15)
            fig = make_bar(budget_view, x="group_key", y="underpaid_budget", title="Remediation budget by group")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No budget summary available.")
        st.markdown('</div>', unsafe_allow_html=True)
    with o2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        draw_section_header("Employee gap distribution")
        if "gap_pct_emp" in scored.columns:
            fig = px.histogram(scored, x="gap_pct_emp")
            fig.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE, font_color=BLACK, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Employee-level gap distribution unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Outlier summary")
    if not outlier_summary.empty:
        st.dataframe(outlier_summary, use_container_width=True)
    elif outlier_error:
        st.warning(outlier_error)
    else:
        st.info("No outlier summary available.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Flagged employees / outliers")
    if not outlier_df.empty:
        st.dataframe(outlier_df.head(250), use_container_width=True)
        try:
            bud = outlier_budget(outlier_df, pay_col=pay_col, target_col="pred_pay" if "pred_pay" in outlier_df.columns else None)
            st.caption("Outlier budget summary")
            st.dataframe(bud, use_container_width=True)
        except Exception:
            pass
    else:
        flagged = scored.loc[scored["gap_pct_emp"] < threshold].copy() if "gap_pct_emp" in scored.columns else pd.DataFrame()
        if not flagged.empty:
            st.dataframe(flagged.head(250), use_container_width=True)
        else:
            st.info("No outlier or flagged employee table available.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Methodology")
    methodology = [
        f"Pay measure used: {pay_col}",
        f"Protected attribute used: {protected_col}",
        f"Comparator dimensions used for expected pay: {', '.join(comparator_cols) if comparator_cols else 'All filtered employees'}",
        f"Minimum sample size rule: {min_n}",
        f"Underpaid threshold rule: below {threshold}% vs expected pay",
        "Expected pay in the unadjusted summary is based on peer mean excluding the protected attribute.",
        "Adjusted model uses log(pay) OLS with available controls.",
        "Outlier detection uses peer-group logic and robust rules from the connected outlier module.",
    ]
    st.markdown("<br>".join([f"• {m}" for m in methodology]), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    draw_section_header("Downloads")
    export_tables = {
        "unadjusted_summary": unadj,
        "adjusted_effects": adj_effects,
        "outlier_summary": outlier_summary,
        "outlier_flagged": outlier_df,
    }
    try:
        excel_bytes = export_excel_pack(scored, outputs=export_tables, filename_prefix="pay_equity_audit")
        st.download_button(
            label="Download Excel pack",
            data=excel_bytes,
            file_name="pay_equity_audit_pack.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.warning(f"Excel pack could not be generated: {e}")

    st.download_button(
        label="Download cleaned dataset (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_compensation_data.csv",
        mime="text/csv",
    )
    if not unadj.empty:
        st.download_button(
            label="Download unadjusted summary (CSV)",
            data=unadj.to_csv(index=False).encode("utf-8"),
            file_name="unadjusted_pay_summary.csv",
            mime="text/csv",
        )
    if not outlier_df.empty:
        st.download_button(
            label="Download outlier / flagged records (CSV)",
            data=outlier_df.to_csv(index=False).encode("utf-8"),
            file_name="outlier_flagged_records.csv",
            mime="text/csv",
        )
    st.markdown('</div>', unsafe_allow_html=True)
