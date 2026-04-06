# Superstore Multi-Agent Analysis Pipeline

A multi-agent system built with **LangGraph** that autonomously analyses four years of retail sales data, trains two competing forecasting models (Prophet vs SARIMAX), selects the best one using a combination of metrics and LLM reasoning, and generates a professional PDF report with a human-in-the-loop review gate before any output is written.

---

## Architecture

```
                    ┌─────────────────────────────────┐
  START ──► Supervisor │ route(state) → next node     │
            (stateless  └─────────────────────────────┘
             router)              │
                    ┌─────────────┼──────────────────────────┐
                    │             │                            │
                    ▼             ▼                            ▼
             Data Agent    Analysis Agent           Forecast Agent
             load/clean    6 EDA charts             Prophet + SARIMAX
             LLM summary   STL decomposition        hold-out evaluation
                           holiday impact           AIC order selection
                    │             │                            │
                    └─────────────┴────────────────────────────┘
                                  │ (all write to AgentState)
                                  ▼
                         Model Selector Agent
                         deterministic winner (MAPE)
                         LLM explains why
                                  │
                                  ▼
                          Insight Agent
                          single LLM call
                          grounded in state data
                                  │
                                  ▼
                          Critic Agent ◄──── retry (max 2x)
                          scores 1–10
                          approved ≥ 6
                                  │
                          ┌───────┘
                          ▼
                    Human Review          ← interrupt() — run.py handles interaction
                    (state passthrough)
                          │
                          ▼
                    Report Agent
                    writes report.md
                    saves forecast_12m.csv
                    builds report.pdf
                          │
                         END
```

**Key architectural rule:** Every agent reads from and writes to the shared `AgentState` TypedDict. No agent calls another agent directly. All coordination flows through state. The Supervisor is the only node with routing logic — workers just do their task and return.

---

## Project Features

| Feature | This project | Typical demo |
|---|---|---|
| Conditional graph routing | ✅ Supervisor reads state, routes dynamically | ❌ Sequential pipeline |
| Feedback loop with retry | ✅ Critic rejects output and re-routes to specific node | ❌ No retry logic |
| Human-in-the-loop | ✅ `interrupt()` checkpoint before report generation | ❌ Fully automated |
| Typed shared state | ✅ `TypedDict` flows through every node | ❌ Arguments passed between functions |
| Competing model evaluation | ✅ AIC grid search + LLM-generated rationale | ❌ Single model |

---

## Agents

### Data Agent
Cleans the raw Superstore CSV deterministically (date parsing, deduplication, postal code imputation, time-feature engineering). The LLM is used only at the end to write a plain-English summary of the quality report for the pipeline log — it does not make any cleaning decisions.

### Analysis Agent
Generates 6 Plotly HTML charts saved to `results/eda/`:
1. STL decomposition (trend + seasonality + residual)
2. Holiday vs normal-day sales distribution (boxplot)
3. Pareto chart — which products drive 80% of revenue
4. Revenue heatmap by day-of-week × month
5. Stacked area chart of category revenue over time
6. Anomaly detection (IQR method)

Also computes and stores structured numerical results in state: monthly sales series, year-over-year growth, seasonality index (month × index value), and holiday lift percentage. These are what the Insight Agent reads to write its narrative.

### Forecast Agent
Trains both models on monthly aggregated sales. Both are evaluated on a **held-out last-6-months split** — not on training data — which is the only honest way to compare models.

**Prophet:**
- `seasonality_mode='multiplicative'` — correct for a series with growing variance
- `weekly_seasonality=False` — monthly data has no within-week pattern to model
- Holiday effects from the US holidays CSV with ±3-day windows
- `changepoint_prior_scale=0.05` — conservative, avoids overfitting trend noise

**SARIMAX:**
- Order selected by AIC grid search over (p,q) ∈ {0,1,2} with fixed seasonal order (1,1,1,12)
- Binary monthly holiday flag as the exogenous variable — built from the real holidays CSV for both training and the forecast period
- `d=1` fixed (upward trend, non-stationary); `D=1, s=12` fixed (annual seasonality)

### Model Selector Agent
Picks the winner by comparing MAPE on the hold-out set. If the difference is within 1 percentage point, Prophet is preferred — a domain heuristic: for retail data with growing seasonal amplitude and holiday effects, Prophet's structural advantages are worth a marginal accuracy tie. An LLM then generates the rationale, citing actual metric values, so the business has a written explanation for the model choice.

### Insight Agent
Single LLM call (GPT-4o-mini, temperature=0.2) with all computed metrics injected into the context — monthly sales, YoY growth, seasonality index, category breakdown, anomalies, forecast results. The system prompt enforces a specific report structure with numbered recommendations, and requires every claim to cite a specific number. The Critic validates this.

### Critic Agent
Scores the insight narrative on 5 dimensions (accuracy, relevance, completeness, tone, groundedness), each out of 2 points. Approves if total ≥ 6/10. On rejection, sets `retry_node="insight_agent"` so the Supervisor re-routes there specifically. Capped at 2 retries. If the Critic itself crashes, it approves by default — a deliberate fail-safe so a broken quality gate doesn't block the pipeline.

### Human Review
An `interrupt()` checkpoint handled entirely in `run.py`. The graph pauses, the CLI shows the critic score and an insight preview, and the user either approves or types feedback. The feedback is stored in state and appears in the final report. `human_review_node` in `report_agent.py` is a passthrough — the interaction belongs in the CLI entry point, not inside a graph node.

### Report Agent
Three sequential steps: (1) saves `forecast_12m.csv` from the winning model's forecast data, (2) renders `report.md` using a Jinja2 template with all pipeline outputs, (3) builds `report.pdf` by parsing `report.md` and `forecast_12m.csv`. The PDF is generated from the real output files — no numbers are duplicated or hardcoded.

---

## Datasets

### 1. Superstore Sales
**Source:** [Kaggle — rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

Download `train.csv` → place at `data/train.csv`.

9,800 order-line records (Jan 2015 – Dec 2018). Strongly right-skewed sales distribution with clear year-end seasonality and consistent growth — well-suited to both Prophet and SARIMAX.

### 2. US Public Holidays
**Source:** [Kaggle — donnetew/us-holiday-dates-2004-2021](https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021)

Download `USHolidays.csv` → place at `data/us_holidays.csv`.

**Why this matters:** Without a holiday calendar, both Prophet and SARIMAX treat the November–December revenue spike as unexplained variance. With it, the models learn that the spike is holiday-driven and can predict it reliably. The `run.py` entry point will exit with a clear error if this file is missing, because the forecast agent requires it.

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
# Open .env and set OPENAI_API_KEY

# 4. Download datasets (links above) and place in data/
mkdir data
# → data/train.csv
# → data/us_holidays.csv

# 5. Run
python run.py
```

### API keys

| Key | Required by |
|---|---|
| `OPENAI_API_KEY` | All LLM agents (supervisor, data, analysis, insight, critic, model_selector) |
| `LANGCHAIN_API_KEY` | Optional — LangSmith tracing only |

### Optional: LangSmith tracing

Uncomment the three LangSmith lines in `.env`. No code changes required. Every LLM call, tool invocation, and state transition will appear at [smith.langchain.com](https://smith.langchain.com). Recommended for demos — it shows the full agent reasoning trace visually.

### CLI options

```bash
python run.py                          # standard interactive run
python run.py --no-hitl                # skip human review (CI / demo mode)
python run.py --goal "Focus on Technology category forecast for 2019"
python run.py --data data/train.csv --holidays data/us_holidays.csv

# After a run — check forecast quality and optionally compare to a baseline:
python evaluate.py
python evaluate.py --baseline results/forecast_12m_v1.csv
```

---

## Project structure

```
superstore_agents/
├── run.py                       # Pipeline entry point + human review interaction
├── evaluate.py                  # Post-run forecast quality audit
├── requirements.txt
├── .env.example
├── data/
│   ├── train.csv                # Superstore dataset (download from Kaggle)
│   └── us_holidays.csv          # US holidays dataset (download from Kaggle)
├── results/
│   ├── eda/                     # 6 Plotly charts from the Analysis Agent
│   ├── report.md                # Markdown report (Jinja2 rendered)
│   ├── report.pdf               # Final PDF report (ReportLab generated)
│   └── forecast_12m.csv         # 12-month forecast from the winning model
└── src/
    ├── workflow.py              # LangGraph StateGraph assembly
    ├── state/
    │   └── agent_state.py       # TypedDict — single source of truth for all agents
    └── agents/
        ├── supervisor.py        # Stateless router — route() function only
        ├── data_agent.py        # Deterministic cleaning + LLM quality summary
        ├── analysis_agent.py    # 6 EDA charts + numerical results to state
        ├── forecast_agent.py    # Prophet + SARIMAX with hold-out evaluation
        ├── model_selector.py    # MAPE-based winner + LLM rationale
        ├── insight_agent.py     # Single LLM call with injected metrics context
        ├── critic_agent.py      # Scored rubric quality gate with retry routing
        └── report_agent.py      # report.md + forecast CSV + report.pdf
```

---

## Key design decisions

**Why LangGraph over CrewAI or AutoGen?**
LangGraph requires you to define nodes, edges, and routing functions explicitly. That means every state transition is visible, testable, and debuggable. CrewAI handles orchestration for you, which is fast to set up but makes it hard to reason about what's actually happening. For a project where the design decisions need to be explainable, explicit control is more valuable than convenience.

**Why TypedDict for shared state?**
Every agent reads from and writes to the same `AgentState` object. TypedDict makes this contract explicit and IDE-checkable — if you try to read a field that doesn't exist, your editor catches it. LangGraph's checkpointer also serialises this automatically, which is what makes the human_review interrupt and resume possible.

**Why multiplicative seasonality in Prophet?**
Monthly sales variance grows over time, the amplitude of seasonal swings in 2018 is roughly twice that of 2015. Multiplicative mode scales the seasonal component proportionally to the trend level, which matches this behaviour. Additive seasonality assumes constant swing amplitude and would underfit the later years.

**Why the Critic Agent?**
LLMs produce plausible-sounding text even when the numbers are wrong or the analysis is shallow. The Critic formalises the review step that a human analyst would do before sending a report to a client. It creates an audit trail (score + issue list in state) and a named retry target, so when it rejects, the Supervisor knows exactly which agent to re-run, rather than restarting the whole pipeline.

**Why does human interaction happen in run.py, not in human_review_node?**
The graph's `interrupt()` mechanism pauses execution and returns control to whoever called `app.stream()`. That's `run.py`. Putting `input()` inside `human_review_node` would mix CLI logic into the graph layer, making it impossible to replace with a web UI later without modifying the agent. The node is a passthrough; the interaction is in the entry point.