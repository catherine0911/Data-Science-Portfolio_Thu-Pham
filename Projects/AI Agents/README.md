# Superstore Multi-Agent Analysis Pipeline

A production-grade multi-agent system built with **LangGraph** that autonomously analyses retail sales data, trains two competing forecasting models (Prophet vs. SARIMA), selects the best one per business segment using an LLM reasoner, and produces a full business intelligence report — with a human-in-the-loop review gate before any output is written.

---

## Architecture

```
START
  └─► Supervisor ──► [route(state)]
                        │
                        ├─► Data Agent            load · validate · clean
                        ├─► Analysis Agent         LLM + 7 EDA tools
                        ├─► Forecast Agent          Prophet + SARIMA + per-segment CV
                        ├─► Model Selector Agent    LLM + 5 comparison tools
                        ├─► Insight Agent           LLM + 5 query tools
                        ├─► Critic Agent            LLM quality gate → can retry ↩
                        ├─► Human Review            interrupt() checkpoint
                        └─► Report Agent            Plotly · Jinja2 · CSV
                                └─► END
```

The **Supervisor** is a pure router — it inspects `AgentState` and returns the name of the next node via a conditional edge function. Every worker writes its output back into the shared `AgentState` TypedDict. No agent talks directly to another; all coordination happens through state.

---

## Agents

### Data Agent
Validates and cleans the raw Superstore CSV using 4 registered `@tool` functions (`check_missing_values`, `check_duplicates`, `check_outliers`, `get_date_range`). Applies cleaning steps: date parsing, deduplication, postal code imputation via state modal value, and time-feature engineering (Year, Month, Quarter, DayOfWeek). Uses an LLM to summarise the quality report in plain English.

### Analysis Agent *(LLM-driven tool calling)*
The LLM receives a dataset overview and 7 analysis tools. It **autonomously decides** which tools to call and in what order, then synthesises a structured `AnalysisResults` object. Tools: `get_summary_stats`, `get_monthly_trend`, `get_top_n`, `get_category_breakdown`, `get_yoy_growth`, `get_seasonality_index`, `detect_anomalies`.

### Forecast Agent
Trains **both models** on the overall sales series and per business segment (Category):

**Prophet:**
- Multiplicative seasonality (appropriate for a growing retail series)
- US public holiday effects via a dedicated holidays dataframe
- Changepoint detection with `changepoint_prior_scale=0.1`
- Hold-out evaluation on last 6 months
- Returns trend decomposition components

**SARIMA (auto-selected via pmdarima):**
- `auto_arima` selects optimal (p,d,q)(P,D,Q)[12] order via AIC
- Hold-out evaluation on last 6 months
- Returns AIC score for model comparison

Both models produce: 12-month forecast, 95% confidence intervals, MAE, RMSE, MAPE.

### Model Selector Agent *(LLM-driven tool calling)*
The most architecturally distinctive agent. Uses 5 comparison tools to gather evidence — `compare_overall_metrics`, `compare_forecast_trajectories`, `compare_segment_metrics`, `compute_ensemble_forecast`, `get_model_selection_criteria` — then reasons about which model (Prophet / SARIMA / Ensemble) is best **overall and per segment**. Does not simply pick the lowest MAE: it considers series characteristics, holiday sensitivity, confidence interval width, segment-level differences, and whether an ensemble would reduce variance.

### Insight Agent *(LLM-driven tool calling)*
Generates a full business intelligence report by calling 5 query tools before writing a single word. Every claim is grounded in actual numbers retrieved via tool calls. Produces: Executive Summary, Revenue Trends, Category Performance, Seasonality Insights, Anomalies & Risks, Forecast Outlook, and 5 Strategic Recommendations.

### Critic Agent
Scores pipeline output on a multi-dimension rubric (data quality, analysis completeness, forecast accuracy, insight depth). Score below 6/10 → rejects and sets `retry_node` so the Supervisor re-routes to the specific failing agent. Capped at 2 retries to prevent infinite loops.

### Human Review *(interrupt checkpoint)*
LangGraph `interrupt()` pauses the graph before the final report. In interactive mode, it shows the critic score and insight preview, then waits for human input. In CI/automated mode (`--no-hitl`), it auto-approves.

### Report Agent
Produces five Plotly interactive HTML charts (monthly trend, 12-month forecast with CI band, category sales, model comparison, seasonality index), a forecast CSV, and a Jinja2-rendered Markdown report covering all pipeline outputs.

---

## Datasets

### 1. Superstore Sales
**Source:** [Kaggle — rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

Download `train.csv` → place at `data/train.csv`.

9,800 order-line records (Jan 2015 – Dec 2018), covering orders, products, customers, geography, and sales across the US. Strongly right-skewed sales distribution with clear year-end seasonality — well-suited for both Prophet and SARIMA.

### 2. US Public Holidays
**Source:** [Kaggle — donnetew/us-holiday-dates-2004-2021](https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021)

Download `USHolidays.csv` → place at `data/us_holidays.csv`.

**Why:** Prophet uses a holiday calendar to learn whether sales are systematically higher or lower around holidays (pre-Christmas spike, post-Thanksgiving tail). Without this, the model treats holiday effects as random noise. The dataset covers 2004–2021, which fully spans the Superstore series (2015–2018) with room to spare.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Open .env and fill in OPENAI_API_KEY and ANTHROPIC_API_KEY

# 4. Download datasets from Kaggle (links above) and place in data/
mkdir data
# → data/train.csv
# → data/us_holidays.csv

# 5. Run the pipeline
python run.py
```

### API keys required

`OPENAI_API_KEY`

### Optional: LangSmith tracing

```bash
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=superstore-agents
```

No code changes required — tracing activates automatically via environment variables. Every LLM call, tool invocation, and state transition appears in your LangSmith dashboard at [smith.langchain.com](https://smith.langchain.com).

### CLI options

```bash
python run.py                                         # standard run
python run.py --no-hitl                               # skip human review (CI/demo mode)
python run.py --goal "Focus on Technology forecasts"  # custom analysis goal
python run.py --superstore data/train.csv --holidays data/us_holidays.csv

# After a run, evaluate outputs and fetch LangSmith traces:
python evaluate.py
python evaluate.py --compare results/forecast_12m_baseline.csv  # regression check
```

---

## Project structure

```
superstore_agents/
├── run.py                      # Pipeline entry point
├── evaluate.py                 # Post-run evaluation + LangSmith trace fetcher
├── requirements.txt
├── .env.example
├── data/
│   ├── train.csv               # Superstore dataset (download from Kaggle)
│   └── us_holidays.csv         # US holidays dataset (download from Kaggle)
├── results/                    # Auto-created on first run
│   ├── report.md               # Full Markdown report
│   ├── forecast_12m.csv        # 12-month forecast (winning model)
│   ├── monthly_sales.html      # Interactive Plotly chart
│   ├── forecast.html           # Forecast + confidence interval chart
│   ├── category_sales.html     # Sales by category
│   ├── model_comparison.html   # Prophet vs SARIMA metrics
│   └── seasonality.html        # Monthly seasonality index
└── src/
    ├── workflow.py             # LangGraph StateGraph assembly
    ├── state/
    │   └── agent_state.py      # TypedDict state
    └── agents/
        ├── supervisor.py       # Conditional routing logic
        ├── data_agent.py       # Load, validate, clean
        ├── analysis_agent.py   # LLM-driven EDA with 7 tools
        ├── forecast_agent.py   # Prophet + SARIMA + per-segment evaluation
        ├── model_selector.py   # LLM-driven model comparison with 5 tools
        ├── insight_agent.py    # LLM-driven narrative with 5 query tools
        ├── critic_agent.py     # Quality gate with structured retry routing
        └── report_agent.py     # Plotly charts · Jinja2 report · human review
```

## Key design decisions

**Why LangGraph over CrewAI or AutoGen?**
LangGraph exposes the full state graph explicitly — you define nodes, edges, and routing functions directly. This means every state transition is inspectable, breakpointable, and resumable from any checkpoint. CrewAI abstracts this away (convenient for demos, limiting for production debugging). AutoGen's actor model is powerful but less suited to the strictly ordered, state-gated flow this pipeline requires.

**Why TypedDict for shared state?**
Type-safe, IDE-friendly, and self-documenting. Every agent reads from and writes to the same `AgentState` object — adding a new field immediately makes it available across the entire graph with no plumbing required. LangGraph's checkpointer serialises this automatically.

**Why multiplicative seasonality in Prophet?**
The Superstore series shows increasing variance over time: monthly swings in 2018 are roughly twice those in 2015. Multiplicative seasonality scales the seasonal component proportionally to the trend level, which models this correctly. Additive seasonality assumes fixed-amplitude swings and would systematically underfit the later years.

**Why a Critic agent with structured retry?**
Real pipelines fail silently — the LLM produces something plausible, you accept it, and only later realise the insight section was thin or the forecast only covered 3 months. The Critic creates a documented quality gate with a scoreable rubric, specific issue descriptions, and a named retry target so the Supervisor knows exactly where to re-enter the graph.
