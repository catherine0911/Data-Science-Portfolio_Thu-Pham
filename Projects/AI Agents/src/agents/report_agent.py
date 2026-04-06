import io
import logging
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from jinja2 import Template
from statsmodels.tsa.seasonal import seasonal_decompose

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import Flowable

from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"

# Colour palette
NAVY   = colors.HexColor("#1e1b4b")
BLUE   = colors.HexColor("#4361EE")
INDIGO = colors.HexColor("#3A0CA3")
PINK   = colors.HexColor("#F72585")
TEAL   = colors.HexColor("#0891b2")
SLATE  = colors.HexColor("#64748b")
LIGHT  = colors.HexColor("#f0f4ff")
LIGHT2 = colors.HexColor("#f8fafc")
WHITE  = colors.white
BORDER = colors.HexColor("#e2e8f0")
GREEN  = colors.HexColor("#065f46")
GREEN_BG = colors.HexColor("#d1fae5")

W, H   = A4
MARGIN = 18 * mm
PW     = W - 2 * MARGIN   # usable page width ≈ 555 pt

MPL = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
}


# Step 1: Human review passthrough

def human_review_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))
    messages.append({"node": "human_review", "status": "approved",
                     "msg": state.get("human_feedback") or "Approved"})
    return {**state, "current_node": "human_review", "messages": messages}


# Step 2: Write report.md and forecast_12m.csv

REPORT_TEMPLATE = """# Multi-Agent Sales Analysis Report
*Generated: {{ generated_at }}*

---

## Pipeline Run Summary

| Agent | Status | Notes |
|---|---|---|
{% for msg in messages %}| `{{ msg.node }}` | {{ msg.status }} | {{ msg.msg[:120] }} |
{% endfor %}

---

## Data Quality

| Metric | Value |
|---|---|
| Total rows | {{ dq.total_rows }} |
| Date range | {{ dq.date_range[0] }} → {{ dq.date_range[1] }} |
| Missing values | {{ dq.missing_by_col }} |
| Duplicates removed | {{ dq.duplicate_rows }} |
| Outliers flagged (z > 3σ) | {{ dq.outlier_rows }} |

{% if dq.warnings %}**Warnings:** {{ dq.warnings | join(' · ') }}{% endif %}

---

## Analysis & Insights

{{ insights }}

---

## Forecast Model Comparison

**Winner: {{ comparison.winner }}**

{{ comparison.rationale }}

| Model | MAE ($) | RMSE ($) | MAPE (%) | AIC |
|---|---|---|---|---|
{% for m in comparison.metrics_table %}| {{ m.model }} | {{ m.mae }} | {{ m.get('rmse', 'N/A') }} | {{ m.mape }} | {{ m.aic or 'N/A' }} |
{% endfor %}

**Business recommendation:** {{ comparison.recommendation }}

---

## Forecast Data

- [Download 12-month forecast CSV](forecast_12m.csv)

---

## Quality Gate

- **Critic score:** {{ critic_score }}/10

{% if human_feedback %}
## Human Reviewer Feedback
{{ human_feedback }}
{% endif %}

---

## Errors (non-fatal)
{% if errors %}{% for e in errors %}- {{ e }}
{% endfor %}{% else %}None.{% endif %}

---
*Pipeline complete · LangGraph multi-agent system*
"""


def _write_markdown(state: AgentState) -> str:
    template   = Template(REPORT_TEMPLATE)
    comparison = state.get("model_comparison") or {}
    dq         = state.get("data_quality")     or {}
    feedback   = state.get("critic_feedback")  or {}

    rendered = template.render(
        generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M"),
        messages      = state.get("messages", []),
        dq            = dq,
        insights      = state.get("insights", ""),
        comparison    = comparison,
        human_feedback= state.get("human_feedback"),
        critic_score  = feedback.get("score", "N/A"),
        errors        = state.get("errors", []),
    )
    path = os.path.join(RESULTS_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rendered)
    return path


def _save_forecast_csv(state: AgentState) -> str:
    comparison = state.get("model_comparison") or {}
    winner     = comparison.get("winner", "Prophet")
    fc         = (state.get("prophet_result") if winner != "SARIMAX"
                  else state.get("sarima_result")) or {}
    rows       = fc.get("forecast_df", [])
    if not rows:
        return ""
    df_fc = pd.DataFrame(rows)
    df_fc["model"] = winner
    path = os.path.join(RESULTS_DIR, "forecast_12m.csv")
    df_fc.to_csv(path, index=False)
    return path


# STEP 3 — Build PDF

# Parsers

def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",       r"\1", text)
    return text.strip()


def _parse_md_table(block: str) -> list[list[str]]:
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or re.match(r"^[\|\-\s:]+$", line):
            continue
        cells = [_strip_md(c.strip()) for c in line.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _parse_report(md_path: str) -> dict:
    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    def _get(pattern, default=""):
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else default

    # Metadata 
    generated    = _get(r"\*Generated:\s*(.+?)\*")
    critic_score = _get(r"\*\*Critic score:\*\*\s*(\S+)", "N/A")

    # Pipeline table 
    m = re.search(r"## Pipeline Run Summary\n+((?:\|.+\n)+)", text)
    pipeline_rows = _parse_md_table(m.group(1)) if m else []

    # Data quality table 
    m = re.search(r"## Data Quality\n+((?:\|.+\n)+)", text)
    dq_rows = _parse_md_table(m.group(1)) if m else []
    m = re.search(r"\*\*Warnings:\*\*(.+)", text)
    dq_warnings = m.group(1).strip() if m else ""

    # Insights block
    m = re.search(
        r"## Analysis & Insights\n+([\s\S]+?)(?=\n---\n## Forecast Model Comparison)",
        text
    )
    insights_raw = m.group(1).strip() if m else ""

    #  Model comparison 
    winner      = _get(r"\*\*Winner:\s*(.+?)\*\*", "N/A")
    rationale   = _get(r"\*\*Winner:.+?\*\*\n\n(.+?)(?=\n\n\||\n\n\*\*Business)", "")
    m = re.search(r"## Forecast Model Comparison\n+[\s\S]+?\n((?:\|.+\n)+)", text)
    model_rows  = _parse_md_table(m.group(1)) if m else []
    biz_rec     = _get(r"\*\*Business recommendation:\*\*(.+?)(?=\n---|\n\n##|\Z)", "")

    #  Strategic Recommendations 
    recs_raw = _get(r"## Strategic Recommendations\n+([\s\S]+?)(?=\n---|\n## |\Z)")
    recs = []
    for m in re.finditer(
        r"\d+\.\s+\*\*(.+?)\*\*\s*([^\n].*?)(?=\n\d+\.\s+\*\*|\n---|\n## |\Z)",
        recs_raw, re.DOTALL
    ):
        title = m.group(1).strip()
        body  = re.sub(r"\s+", " ", m.group(2)).strip()
        if title:
            recs.append((title, body))

    # Human feedback 
    human_feedback = _get(r"## Human Reviewer Feedback\n+([\s\S]+?)(?=\n---|\Z)")

    return {
        "generated":      generated,
        "critic_score":   critic_score,
        "pipeline_rows":  pipeline_rows,
        "dq_rows":        dq_rows,
        "dq_warnings":    dq_warnings,
        "insights_raw":   insights_raw,
        "winner":         winner,
        "rationale":      rationale,
        "model_rows":     model_rows,
        "biz_rec":        biz_rec,
        "recs":           recs,
        "human_feedback": human_feedback,
    }


def _split_sections(insights_raw: str) -> dict[str, str]:
    sections = {}
    headings = [(m.start(), m.group(1))
                for m in re.finditer(r"^#{1,3}\s+(.+)$", insights_raw, re.MULTILINE)]
    for i, (start, title) in enumerate(headings):
        end  = headings[i+1][0] if i + 1 < len(headings) else len(insights_raw)
        body = re.sub(r"^#{1,3}.+\n", "", insights_raw[start:end], count=1).strip()
        sections[title.strip()] = body
    return sections


def _tables_in_section(text: str) -> list[list[list[str]]]:
    tables = []
    for m in re.finditer(r"((?:\|.+\n)+)", text, re.MULTILINE):
        rows = _parse_md_table(m.group(1))
        if len(rows) >= 2:
            tables.append(rows)
    return tables


# ReportLab styles

def _styles() -> dict:
    def s(n, **kw): return ParagraphStyle(n, **kw)
    return {
        "cover_title": s("CT", fontSize=28, textColor=WHITE,
                          fontName="Helvetica-Bold", leading=34, spaceAfter=8),
        "cover_sub":   s("CS", fontSize=13, textColor=colors.HexColor("#c7d2fe"),
                          fontName="Helvetica", leading=18),
        "cover_meta":  s("CM", fontSize=10, textColor=colors.HexColor("#a5b4fc"),
                          fontName="Helvetica", spaceAfter=4),
        "section":     s("SH", fontSize=13, textColor=NAVY, fontName="Helvetica-Bold",
                          spaceBefore=14, spaceAfter=6, leading=18),
        "subsection":  s("SS", fontSize=11, textColor=INDIGO, fontName="Helvetica-Bold",
                          spaceBefore=10, spaceAfter=4, leading=14),
        "body":        s("BD", fontSize=9.5, textColor=colors.HexColor("#1e293b"),
                          fontName="Helvetica", leading=15, spaceAfter=5),
        "caption":     s("CP", fontSize=8, textColor=SLATE,
                          fontName="Helvetica-Oblique", leading=11,
                          spaceAfter=6, alignment=TA_CENTER),
        "footer":      s("FT", fontSize=7.5, textColor=SLATE,
                          fontName="Helvetica", alignment=TA_CENTER),
        "bullet":      s("BU", fontSize=9.5, textColor=colors.HexColor("#1e293b"),
                          fontName="Helvetica", leading=14, spaceAfter=3,
                          leftIndent=12, firstLineIndent=-8),
        "table_hdr":   s("TH", fontSize=8, textColor=WHITE, fontName="Helvetica-Bold",
                          alignment=TA_CENTER),
        "small":       s("SM", fontSize=8.5, textColor=SLATE,
                          fontName="Helvetica", leading=12),
    }


class Banner(Flowable):
    def __init__(self, text, bg=NAVY, height=28, fontsize=12):
        super().__init__()
        self.text = text; self.bg = bg; self.bh = height; self.fs = fontsize
        self.width = PW; self.height = height

    def draw(self):
        self.canv.setFillColor(self.bg)
        self.canv.roundRect(0, 0, self.width, self.bh, 5, fill=1, stroke=0)
        self.canv.setFillColor(WHITE)
        self.canv.setFont("Helvetica-Bold", self.fs)
        self.canv.drawCentredString(
            self.width / 2, self.bh / 2 - self.fs * 0.35, self.text
        )


class KpiRow(Flowable):
    def __init__(self, items, card_h=54):
        super().__init__()
        self.items = items; self.width = PW; self.height = card_h + 4

    def draw(self):
        n, cw, ch = len(self.items), self.width / len(self.items), self.height - 4
        for i, (val, label, col) in enumerate(self.items):
            x = i * cw
            self.canv.setFillColor(col)
            self.canv.roundRect(x + 2, 2, cw - 4, ch, 5, fill=1, stroke=0)
            self.canv.setFillColor(WHITE)
            self.canv.setFont("Helvetica-Bold", 14)
            self.canv.drawCentredString(x + cw / 2, ch / 2 + 4, str(val))
            self.canv.setFont("Helvetica", 7.5)
            self.canv.drawCentredString(x + cw / 2, ch / 2 - 9, str(label))


def _tbl(rows: list, col_fracs: list = None, hdr_color=NAVY) -> Table | None:
    if not rows:
        return None
    n = len(rows[0])
    if col_fracs is None:
        col_fracs = [1 / n] * n
    # Ensure fractions sum to ≤ 1
    total = sum(col_fracs)
    col_widths = [PW * (f / total) for f in col_fracs]

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  hdr_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, BORDER),
        ("WORDWRAP",      (0, 0), (-1, -1), True),
    ]))
    return tbl


def _section_flowables(text: str, ST: dict) -> list:
    result   = []
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            result.append(Spacer(1, 4))
            in_table = False
            continue
        if s.startswith("|"):
            in_table = True
            continue
        if in_table and re.match(r"^[\-|:\s]+$", s):
            continue
        in_table = False

        if s.startswith("### "):
            result.append(Paragraph(s[4:], ST["subsection"]))
        elif s.startswith("## "):
            result.append(Paragraph(s[3:], ST["section"]))
        elif re.match(r"^\d+\.\s", s) or s.startswith(("- ", "* ")):
            body = re.sub(r"^[\d]+\.\s|^[-\*]\s", "", s)
            body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
            result.append(Paragraph(f"• {body}", ST["bullet"]))
        else:
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            clean = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", clean)
            result.append(Paragraph(clean, ST["body"]))
    return result


# Chart builders

def _fig_to_img(fig, width_pt: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    img = Image(buf)
    img.drawWidth  = width_pt
    img.drawHeight = width_pt * (img.imageHeight / img.imageWidth)
    return img


def _chart_monthly_sales(analysis: dict) -> Image | None:
    monthly = analysis.get("monthly_sales", {})
    if not monthly or not monthly.get("period"):
        return None
    periods = monthly["period"]
    values  = monthly["value"]
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(8, 3))
        x = np.arange(len(periods))
        ax.fill_between(x, values, alpha=0.12, color="#4361EE")
        ax.plot(x, values, color="#4361EE", lw=2, marker="o", ms=3)
        # Annotate peak
        peak_i = int(np.argmax(values))
        ax.scatter([peak_i], [values[peak_i]], color="#F72585", s=60, zorder=5)
        ax.annotate(f"Peak\n${values[peak_i]/1000:.0f}K",
                    xy=(peak_i, values[peak_i]),
                    xytext=(max(0, peak_i - 4), values[peak_i] * 0.88),
                    fontsize=7.5, color="#F72585", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#F72585", lw=1))
        tick_step = max(1, len(periods) // 10)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(periods[::tick_step], rotation=35, ha="right", fontsize=7.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        ax.set_title("Monthly Sales Trend (2015–2018)", fontweight="bold", pad=8)
        ax.set_ylabel("Sales ($)")
        fig.tight_layout()
    return _fig_to_img(fig, PW)


def _chart_category(analysis: dict) -> Image | None:
    breakdown = analysis.get("category_breakdown", [])
    if not breakdown:
        return None
    df = pd.DataFrame(breakdown)
    val_col = "Sales" if "Sales" in df.columns else df.columns[-1]
    cat_col = df.columns[0]
    df = df.sort_values(val_col)
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(7, max(2, len(df) * 0.55)))
        colors_list = ["#3A0CA3", "#4361EE", "#748FCA"][:len(df)]
        bars = ax.barh(df[cat_col], df[val_col], color=colors_list, height=0.5)
        for bar, val in zip(bars, df[val_col]):
            ax.text(val + df[val_col].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"${val/1000:.0f}K", va="center", fontsize=8.5, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        ax.set_title("Total Sales by Category (2015–2018)", fontweight="bold", pad=8)
        ax.set_xlim(0, df[val_col].max() * 1.18)
        fig.tight_layout()
    return _fig_to_img(fig, PW * 0.72)


def _chart_seasonality(analysis: dict) -> Image | None:
    idx = analysis.get("seasonality_index", {})
    if not idx:
        return None
    months = list(idx.keys())
    values = [float(v) for v in idx.values()]
    bar_colors = ["#3A0CA3" if v >= 1.2 else
                  "#4361EE" if v >= 1.0 else
                  "#748FCA" if v >= 0.75 else
                  "#F72585" for v in values]
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        bars = ax.bar(months, values, color=bar_colors, width=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(1.0, color="#64748b", lw=1.2, ls="--", alpha=0.7)
        ax.text(len(months) - 0.4, 1.03, "avg", fontsize=7, color="#64748b")
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_title("Monthly Seasonality Index  (>1.0 = above average)", fontweight="bold", pad=8)
        ax.set_ylabel("Index")
        patches = [
            mpatches.Patch(color="#3A0CA3", label="Strong peak (≥1.2)"),
            mpatches.Patch(color="#4361EE", label="Above avg"),
            mpatches.Patch(color="#748FCA", label="Below avg"),
            mpatches.Patch(color="#F72585", label="Weak trough"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
        fig.tight_layout()
    return _fig_to_img(fig, PW)


def _chart_forecast(fc_df: pd.DataFrame, winner: str) -> Image | None:
    if fc_df.empty or "yhat" not in fc_df.columns:
        return None
    x      = np.arange(len(fc_df))
    months = fc_df["ds"].astype(str).tolist()
    yhat   = fc_df["yhat"].tolist()
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        if "yhat_lower" in fc_df.columns and "yhat_upper" in fc_df.columns:
            ax.fill_between(x, fc_df["yhat_lower"], fc_df["yhat_upper"],
                            alpha=0.18, color="#F72585", label="95% CI")
        ax.plot(x, yhat, color="#F72585", lw=2.5, ls="--", marker="D", ms=4.5,
                label=f"{winner} Forecast", zorder=4)
        peak_i = int(np.argmax(yhat))
        ax.scatter([peak_i], [yhat[peak_i]], color="#3A0CA3", s=65, zorder=5)
        ax.annotate(f"Peak: ${yhat[peak_i]:,.0f}",
                    xy=(peak_i, yhat[peak_i]),
                    xytext=(max(0, peak_i - 2.5), yhat[peak_i] * 0.86),
                    fontsize=7.5, color="#3A0CA3", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#3A0CA3", lw=1))
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=40, ha="right", fontsize=7.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        ax.set_title(f"12-Month Sales Forecast — {winner}", fontweight="bold", pad=8)
        ax.set_ylabel("Sales ($)")
        ax.legend(fontsize=7.5, loc="upper left")
        fig.tight_layout()
    return _fig_to_img(fig, PW)


def _chart_model_comparison(model_rows: list) -> Image | None:
    if not model_rows or len(model_rows) < 2:
        return None
    headers = model_rows[0]
    rows    = model_rows[1:]

    def _num(s):
        cleaned = re.sub(r"[^\d.]", "", str(s))
        return float(cleaned) if cleaned else 0.0

    mae_idx  = next((i for i, h in enumerate(headers) if "MAE"  in h.upper()), 1)
    mape_idx = next((i for i, h in enumerate(headers) if "MAPE" in h.upper()), 3)
    names, maes, mapes = [], [], []
    for row in rows:
        if len(row) > max(mae_idx, mape_idx):
            names.append(row[0]); maes.append(_num(row[mae_idx])); mapes.append(_num(row[mape_idx]))
    if not names:
        return None

    pal = ["#4361EE", "#F72585", "#748FCA"]
    with plt.style.context(MPL):
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
        for ax, vals, title, fmt in zip(
            axes,
            [maes, mapes],
            ["MAE — hold-out ($)", "MAPE — hold-out (%)"],
            [lambda v: f"${v:,.0f}", lambda v: f"{v:.1f}%"],
        ):
            b = ax.bar(names, vals, color=pal[:len(names)], width=0.45)
            for bar, val in zip(b, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val * 1.03,
                        fmt(val), ha="center", fontsize=8.5, fontweight="bold")
            ax.set_title(title, fontweight="bold")
        fig.tight_layout(w_pad=3)
    return _fig_to_img(fig, PW)

def _plot_decomposition(df: pd.DataFrame) -> Image:
    ts = df.set_index("Order_Date").resample("W")["Sales"].sum().fillna(0)
    decomp = seasonal_decompose(ts, model="additive", period=52)
    with plt.style.context(MPL):
        fig, axes = plt.subplots(3, 1, figsize=(8, 4.5), sharex=True)
        axes[0].plot(ts.index, ts.values, color="#118AB2"); axes[0].set_title("Observed Weekly Sales", pad=3)
        axes[1].plot(decomp.trend.index, decomp.trend.values, color="#EF476F", lw=2.5); axes[1].set_title("Underlying Trend", pad=3)
        axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color="#06D6A0"); axes[2].set_title("Seasonality", pad=3)
        fig.tight_layout()
    return _fig_to_img(fig, PW)

def _plot_holiday_impact(df: pd.DataFrame) -> Image:
    if "is_holiday" not in df.columns: return None
    daily = df.groupby(["Order_Date", "is_holiday"])["Sales"].sum().reset_index()
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(6, 3))
        data = [daily[daily["is_holiday"] == True]["Sales"], daily[daily["is_holiday"] == False]["Sales"]]
        ax.boxplot(data, labels=["Holiday", "Normal Day"], patch_artist=True, boxprops=dict(facecolor="#4361EE", color="#4361EE"))
        ax.set_title("Sales Distribution: Holidays vs Normal Days", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        fig.tight_layout()
    return _fig_to_img(fig, PW * 0.7)

def _plot_pareto(df: pd.DataFrame) -> Image:
    top50 = df.groupby("Product_Name")["Sales"].sum().sort_values(ascending=False).head(50)
    cum_pct = (top50.cumsum() / top50.sum()) * 100
    with plt.style.context(MPL):
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax1.bar(range(len(top50)), top50.values, color="#118AB2")
        ax1.set_xticks([]); ax1.set_ylabel("Revenue ($)")
        ax2 = ax1.twinx()
        ax2.plot(range(len(top50)), cum_pct.values, color="#EF476F", lw=2)
        ax2.set_ylabel("Cumulative %")
        ax1.set_title("Pareto: Top 50 Products by Revenue", fontweight="bold")
        fig.tight_layout()
    return _fig_to_img(fig, PW)

def _plot_seasonality_heatmap(df: pd.DataFrame) -> Image:
    df = df.copy()
    df["DayOfWeek"] = df["Order_Date"].dt.day_name()
    df["Month"] = df["Order_Date"].dt.month_name()
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    pivot = pd.pivot_table(df, values="Sales", index="DayOfWeek", columns="Month", aggfunc="sum").reindex(days)[months]
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(8, 3))
        cax = ax.imshow(pivot, cmap="Blues", aspect="auto")
        ax.set_xticks(range(12)); ax.set_xticklabels([m[:3] for m in months])
        ax.set_yticks(range(7)); ax.set_yticklabels([d[:3] for d in days])
        ax.set_title("Revenue Heatmap: Day of Week × Month", fontweight="bold")
        fig.colorbar(cax, ax=ax, format="$%d")
        fig.tight_layout()
    return _fig_to_img(fig, PW)

def _plot_category_trends(df: pd.DataFrame) -> Image:
    monthly_cat = df.groupby([pd.Grouper(key="Order_Date", freq="ME"), "Category"])["Sales"].sum().unstack().fillna(0)
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.stackplot(monthly_cat.index, monthly_cat.T.values, labels=monthly_cat.columns, colors=["#3A0CA3", "#4361EE", "#748FCA", "#06D6A0"][:len(monthly_cat.columns)])
        ax.set_title("Category Revenue Over Time", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        fig.tight_layout()
    return _fig_to_img(fig, PW)

def _plot_anomalies(df: pd.DataFrame) -> Image:
    ts = df.set_index("Order_Date").resample("W")["Sales"].sum().fillna(0)
    q1, q3 = ts.quantile(0.25), ts.quantile(0.75)
    iqr = q3 - q1
    anomalies = ts[(ts < q1 - 1.5 * iqr) | (ts > q3 + 1.5 * iqr)]
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ts.index, ts.values, color="#118AB2", label="Weekly Sales")
        ax.scatter(anomalies.index, anomalies.values, color="#EF476F", s=60, marker="x", label="Anomaly", zorder=5)
        ax.set_title("Sales Anomaly Detection (IQR Method)", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _fig_to_img(fig, PW)

def _chart_top_products(analysis: dict) -> Image | None:
    top = analysis.get("top_products", [])
    if not top: return None
    df = pd.DataFrame(top).sort_values("Sales", ascending=True)
    with plt.style.context(MPL):
        fig, ax = plt.subplots(figsize=(7, max(2, len(df) * 0.35)))
        ax.barh(df["Product_Name"].str.slice(0, 40) + '...', df["Sales"], color="#0891b2", height=0.6)
        ax.set_title("Top 10 Products by Revenue", fontweight="bold", pad=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        fig.tight_layout()
    return _fig_to_img(fig, PW * 0.8)


# PDF assembly 

def _build_pdf(md_path: str, csv_path: str, out_path: str, state: AgentState):
    parsed   = _parse_report(md_path)
    sections = _split_sections(parsed["insights_raw"])
    winner   = parsed["winner"]
    ST       = _styles()

    fc_df = pd.DataFrame()
    if csv_path and os.path.exists(csv_path):
        fc_df = pd.read_csv(csv_path)
        fc_df.columns = [c.lower() for c in fc_df.columns]

    # Load full dataset for EDA charting
    clean_path = state.get("df_clean_path")
    df = pd.read_csv(clean_path, parse_dates=["Order_Date"]) if clean_path and os.path.exists(clean_path) else None

    # Load holidays for Box Plot
    hols_path = state.get("holidays_path")
    if df is not None and hols_path and os.path.exists(hols_path):
        hols = pd.read_csv(hols_path)
        hols.columns = [c.lower().strip() for c in hols.columns]
        date_col = next(c for c in hols.columns if "date" in c)
        df["is_holiday"] = df["Order_Date"].dt.normalize().isin(pd.to_datetime(hols[date_col]).dt.normalize())

    analysis   = state.get("analysis")   or {}
    dq         = state.get("data_quality") or {}
    comparison = state.get("model_comparison") or {}

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Superstore Multi-Agent Sales Report",
    )
    story = []


    # Page 1: Cover 
    story += [
        Spacer(1, 58 * mm),
        Paragraph("Superstore Sales Analysis", ST["cover_title"]),
        Paragraph("Multi-Agent Pipeline Report", ST["cover_sub"]),
        Spacer(1, 8 * mm),
        Paragraph(f"Generated: {parsed['generated']}", ST["cover_meta"]),
        Paragraph(f"Winning model: {winner}  ·  Critic score: {parsed['critic_score']}", ST["cover_meta"]),
        PageBreak(),
    ]

    # Page 2: Dataset Overview
    story.append(Banner("Dataset Overview")); story.append(Spacer(1, 8))
    stats_rows = [
        ["Metric", "Value"],
        ["Total order lines",    f"{dq.get('total_rows', 'N/A'):,}"],
        ["Date range",           f"{dq.get('date_range', ['?','?'])[0]}  →  {dq.get('date_range', ['?','?'])[1]}"],
        ["Sales mean / median",  f"${analysis.get('summary_stats', {}).get('Sales', {}).get('mean', 0):.2f} / ${analysis.get('summary_stats', {}).get('Sales', {}).get('50%', 0):.2f}"],
        ["Outliers / Missing",   f"{dq.get('outlier_rows', '0')} / {dq.get('missing_by_col', {}) or 'None'}"],
    ]
    story.append(_tbl(stats_rows, [0.4, 0.6]))
    story.append(Spacer(1, 10))

    mc = _chart_monthly_sales(analysis)
    if mc: story += [mc, Spacer(1, 8)]
    tc = _chart_top_products(analysis)
    if tc: story.append(tc)
    story.append(PageBreak())

    # Page 3: Pipeline & DQ
    story.append(Banner("Pipeline Run Summary")); story.append(Spacer(1, 5))
    if parsed["pipeline_rows"]: story.append(_tbl(parsed["pipeline_rows"], [0.22, 0.13, 0.65]))
    story.append(Spacer(1, 10))
    story.append(Banner("Data Quality Gate", bg=colors.HexColor("#0f172a"), height=24, fontsize=11)); story.append(Spacer(1, 5))
    if parsed["dq_rows"]: story.append(_tbl(parsed["dq_rows"], [0.5, 0.5]))
    story.append(PageBreak())

    # Page 4: Executive Summary
    story.append(Banner("Executive Summary"))
    story.append(Spacer(1, 8))

    total_fc = fc_df["yhat"].sum()  if "yhat"  in fc_df.columns else 0
    peak_fc  = fc_df["yhat"].max()  if "yhat"  in fc_df.columns else 0
    model_lbl= fc_df["model"].iloc[0] if "model" in fc_df.columns else winner
    n_months = len(fc_df)

    story.append(KpiRow([
        (f"${total_fc/1000:.0f}K", "2019 total forecast",  INDIGO),
        (f"${peak_fc/1000:.0f}K",  "Peak month forecast",  BLUE),
        (str(n_months),             "Months forecast",      TEAL),
        (model_lbl,                 "Winning model",        colors.HexColor("#0f766e")),
        (parsed["critic_score"],    "Critic score",         colors.HexColor("#7c3aed")),
    ]))
    story.append(Spacer(1, 10))
    exec_text = sections.get("Executive Summary", "")
    if exec_text:
        story += _section_flowables(exec_text, ST)
    story.append(PageBreak())

    # Page 5-6 Exploratory Data Analysis (EDA)
    story.append(Banner("Exploratory Data Analysis (EDA)")); story.append(Spacer(1, 8))
    if df is not None:
        story.append(_plot_decomposition(df)); story.append(Spacer(1, 8))
        story.append(_plot_pareto(df)); story.append(Spacer(1, 8))
        story.append(_plot_category_trends(df)); story.append(Spacer(1, 8))
        story.append(PageBreak())
        
        story.append(Paragraph("Seasonality & Anomalies", ST["section"]))
        story.append(_plot_seasonality_heatmap(df)); story.append(Spacer(1, 8))
        story.append(_plot_anomalies(df)); story.append(Spacer(1, 8))
        if "is_holiday" in df.columns:
            story.append(_plot_holiday_impact(df))
        story.append(PageBreak())

    # Page 7: Forecast Outlook 
    story.append(Banner(f"12-Month Forecast — {winner}"))
    story.append(Spacer(1, 6))

    # Forecast chart
    fc_chart = _chart_forecast(fc_df, winner)
    if fc_chart:
        story.append(fc_chart)
        story.append(Paragraph(
            f"Figure 4 — {winner} model forecast for Jan–Dec 2019 with 95% confidence interval.",
            ST["caption"]))
        story.append(Spacer(1, 8))

    # Forecast Outlook narrative section
    fo_sec = sections.get("Forecast Outlook", "")
    if fo_sec:
        story += _section_flowables(fo_sec, ST)
        story.append(Spacer(1, 8))

    # 12-month forecast table from CSV
    if not fc_df.empty:
        present   = [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper", "model"] if c in fc_df.columns]
        fc_header = [c.upper().replace("_", " ") for c in present]
        fc_data   = [fc_header]
        for _, row in fc_df.iterrows():
            fc_data.append([
                str(row[c]) if c in ("ds", "model") else f"${row[c]:,.0f}"
                for c in present
            ])
        if "yhat" in fc_df.columns:
            tot      = [""] * len(present)
            tot[0]   = "TOTAL"
            yi       = present.index("yhat")
            tot[yi]  = f"${fc_df['yhat'].sum():,.0f}"
            fc_data.append(tot)
        fc_tbl = Table(fc_data, colWidths=[PW / len(present)] * len(present), repeatRows=1)
        st_fc  = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -2), [WHITE, LIGHT]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("LINEBELOW",     (0, 1), (-1, -2), 0.3, BORDER),
            # Total row styling
            ("BACKGROUND",    (0, -1), (-1, -1), LIGHT),
            ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ])
        fc_tbl.setStyle(st_fc)
        story.append(fc_tbl)
    story.append(PageBreak())

    # Page 8-9: Model Comparison
    story.append(Banner("Forecast Model Comparison"))
    story.append(Spacer(1, 6))

    # Chart
    mc_chart = _chart_model_comparison(parsed["model_rows"])
    if mc_chart:
        story.append(mc_chart)
        story.append(Paragraph("Figure 5 — MAE and MAPE on hold-out evaluation set.", ST["caption"]))
        story.append(Spacer(1, 8))

    # Winner + rationale
    if parsed["rationale"]:
        story.append(Paragraph(f"<b>Winner: {winner}</b>", ST["subsection"]))
        story.append(Paragraph(parsed["rationale"], ST["body"]))
        story.append(Spacer(1, 6))

    # Metrics table
    if parsed["model_rows"]:
        story.append(_tbl(parsed["model_rows"]))
        story.append(Spacer(1, 8))

    if parsed["biz_rec"]:
        story.append(Paragraph(
            f"<b>Business recommendation:</b> {parsed['biz_rec']}", ST["body"]))
    story.append(PageBreak())

    # Page 10: Strategic Recommendations 
    story.append(Banner("Strategic Recommendations"))
    story.append(Spacer(1, 8))

    recs = parsed["recs"]
    if recs:
        for i, (title, body) in enumerate(recs, 1):
            rec_tbl = Table(
                [[
                    Paragraph(str(i), ParagraphStyle(
                        "RN", fontSize=13, textColor=WHITE,
                        fontName="Helvetica-Bold", leading=16, alignment=TA_CENTER
                    )),
                    Paragraph(
                        f"<b>{title}</b><br/>{body}",
                        ParagraphStyle(
                            "RB", fontSize=9.5, fontName="Helvetica",
                            leading=14, textColor=colors.HexColor("#1e293b"),
                            spaceAfter=2
                        )
                    ),
                ]],
                colWidths=[24, PW - 24]
            )
            rec_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (0, 0),   BLUE),
                ("BACKGROUND",    (1, 0), (1, 0),   LIGHT),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING",   (1, 0), (1, 0),   12),
                ("RIGHTPADDING",  (1, 0), (1, 0),   10),
            ]))
            story.append(rec_tbl)
            story.append(Spacer(1, 7))
    else:
        story.append(Paragraph(
            "No recommendations were parsed — check that the insight agent output "
            "numbered items in the format '1. **Title** body text'.", ST["body"]
        ))

    # Footer
    story += [
        Spacer(1, 16),
        HRFlowable(width=PW, thickness=0.5, color=BORDER),
        Spacer(1, 5),
        Paragraph(
            f"Superstore Multi-Agent Pipeline · LangGraph · Winning model: {winner} · "
            f"Critic score: {parsed['critic_score']} · Generated: {parsed['generated']}",
            ST["footer"]
        ),
    ]

    doc.build(story)
    logger.info("PDF saved → %s", out_path)


# Main agent node

def report_agent_node(state: AgentState) -> AgentState:
    logger.info("=== REPORT AGENT starting ===")
    messages = list(state.get("messages", []))
    errors   = list(state.get("errors",   []))

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        forecast_csv = _save_forecast_csv(state)
        md_path      = _write_markdown(state)
        pdf_path     = os.path.join(RESULTS_DIR, "report.pdf")

        # Pass state so the PDF builder can access analysis data for charts
        _build_pdf(md_path, forecast_csv, pdf_path, state)

        messages.append({
            "node":   "report_agent",
            "status": "success",
            "msg":    f"report.pdf + report.md + forecast_12m.csv → {RESULTS_DIR}/",
        })

        return {
            **state,
            "report_path":       pdf_path,
            "forecast_csv_path": forecast_csv,
            "current_node":      "report_agent",
            "messages":          messages,
            "errors":            errors,
        }

    except Exception as exc:
        logger.exception("Report agent failed: %s", exc)
        errors.append(f"report_agent: {exc}")
        messages.append({"node": "report_agent", "status": "error", "msg": str(exc)})
        return {**state, "errors": errors, "messages": messages}