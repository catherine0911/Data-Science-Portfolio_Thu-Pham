# Superstore Multi-Agent Analysis Pipeline

A stateful multi-agent system built with **LangGraph** that autonomously analyzes retail sales data, runs competing forecasting models (Prophet vs. SARIMA), and selects the best model per business segment. Features a human-in-the-loop review gate before generating final reports.

## Architecture

```
START
  └─► Supervisor ──► [route(state)]
                        │
                        ├─► Data Agent         (load, validate, clean)
                        ├─► Analysis Agent     (LLM + 7 EDA tools)
                        ├─► Forecast Agent     (Prophet + SARIMA)
                        ├─► Model Selector     (LLM + 5 comparison tools)
                        ├─► Insight Agent      (LLM + 7 query tools)
                        ├─► Critic Agent       (LLM quality gate → can retry)
                        ├─► Human Review       (interrupt() checkpoint)
                        └─► Report Agent       (Plotly + Jinja2 + CSV)
                                └─► END
```

The **Supervisor** is a stateless router — it reads `AgentState` and returns the name of the next node. Every worker writes its output back to the shared `AgentState`, which is a `TypedDict` that flows through the entire graph.


## Agents

1. **Data Agent**: Validates and cleans raw CSV data, handling missing values and outlier detection.
2. **Analysis Agent**: Performs LLM-driven EDA using specialized tools (summary stats, trends, YOY growth).
3. **Forecast Agent**: Trains Prophet and SARIMA models, evaluating both overall and per-segment data via cross-validation.
4. **Model Selector Agent**: Evaluates model performance and selects the optimal approach (Prophet, SARIMA, or Ensemble) based on data characteristics.
5. **Insight Agent**: Generates a business intelligence narrative grounded in the generated data.
6. **Critic Agent**: Reviews the pipeline output. Re-routes state execution if data quality or analysis depth fails to meet thresholds.
7. **Human Review**: `interrupt()` checkpoint allowing human approval before report generation.
8. **Report Agent**: Outputs interactive Plotly HTML charts, a forecast CSV, and a markdown summary.


## Datasets

### 1. Superstore Sales
**Source:** [Kaggle — rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

9,800 order-line records (Jan 2015 – Dec 2018), covering orders, products, customers, geography, and sales across the US.

### 2. US Public Holidays
**Source:** [Kaggle — donnetew/us-holiday-dates-2004-2021](https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021)


**Why:** Prophet uses a holiday calendar to learn whether sales are systematically higher/lower around holidays (e.g. pre-Christmas boost, post-Thanksgiving spike).


## Setup

```bash
# 1. Clone / download the project
cd superstore_agents

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Download datasets (see above) and place in data/

# 6. Run
python run.py
```

### Optional: LangSmith tracing
```bash
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=superstore-agents
```
This gives you a full trace of every LLM call, tool invocation, and state transition in the LangSmith dashboard — essential for debugging and demonstrating observability.

### CLI options
```bash
python run.py --goal "Focus on Technology category forecasts for Q1 2019"
python run.py --no-hitl           # skip human review (good for demos)
python run.py --superstore path/to/your.csv
```


## Project structure

```
superstore_agents/
├── run.py                          # Entry point
├── requirements.txt
├── .env.example
├── data/
│   ├── train.csv                   # Superstore (download from Kaggle)
│   └── us_holidays.csv             # US holidays (download from Kaggle)
├── results/                        # Auto-created on first run
│   ├── report.md
│   ├── forecast_12m.csv
│   ├── monthly_sales.html
│   ├── forecast.html
│   ├── category_sales.html
│   └── model_comparison.html
└── src/
    ├── workflow.py                 # LangGraph graph assembly
    ├── state/
    │   └── agent_state.py          # TypedDict — single source of truth
    └── agents/
        ├── supervisor.py           # Routing logic
        ├── data_agent.py           # Load + validate + clean
        ├── analysis_agent.py       # LLM-driven EDA with 7 tools
        ├── forecast_agent.py       # Prophet + SARIMA + per-segment
        ├── model_selector.py       # LLM-driven model comparison
        ├── insight_agent.py        # LLM-driven narrative generation
        ├── critic_agent.py         # Quality gate with retry routing
        └── report_agent.py         # Plotly + Jinja2 + human review
```


## Key design decisions

**Why LangGraph over CrewAI/AutoGen?**
LangGraph gives you explicit control over state and routing. You can inspect every state transition, add breakpoints, and resume from any checkpoint. CrewAI abstracts this away — fine for demos, limiting for production.

**Why TypedDict for state?**
Type-safe, IDE-friendly, and self-documenting. Adding a new field to `AgentState` immediately makes it available to all nodes — no plumbing required.

**Why multiplicative seasonality in Prophet?**
The Superstore series shows increasing variance over time (revenue in 2018 swings more than in 2015). Multiplicative seasonality models this correctly; additive would underfit the later years.
