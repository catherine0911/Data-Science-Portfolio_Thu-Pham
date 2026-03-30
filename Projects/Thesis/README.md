Thesis/
├─ data/                 
│  ├─ raw/               # Original TweetEval and iSarcasm files
│  └─ processed/         # JSONL files with LLM Rationales and saved .pt Embeddings
├─ models/               # Saved model checkpoints (.pt)
├─ notebooks/            # EDA and Error Analysis
├─ outputs/              # Results, confusion matrices, and training logs
├─ src/
│  ├─ baseline_model.py  # Standard RoBERTa (1 Head: Sentiment)
│  ├─ mtl_model.py       # Multi-Head RoBERTa (2 Head: Sentiment + Sarcasm)
│  ├─ config.py          # Add Loss weights (alpha/beta) and API keys
│  ├─ preprocessing.py   # Tokenization and handling of iSarcasm dataset
│  ├─ data_loader.py     # Loads Tweet text + Labels + Rationale Embeddings
│  ├─ evaluation.py      # Metric computation for all 3 tasks
│  └─ utils.py           # Custom Masked Loss function for Combined training
├─ run_baseline.py       # Train Model 1 (Baseline)
├─ run_mtl.py            # Train Model 2 (MTL: Sentiment + Sarcasm)
├─ run_rs.py             # Train Model 3 (RS: Sentiment + Rationale Prediction)
├─ run_combined.py       # Train Model 4 (Combined: All 3 heads + Masked Loss)
├─ run_mtl_tuning.py     # Hyperparameter tuning for MTL
├─ run_rs_tuning.py      # Tune Rationale Loss weight (beta)
├─ run_combined_tuning.py # Final fine-tuning of all weights
├─ generate_rationales.py # Script to call OpenAI/Gemini for reasoning data
├─ README.md
└─ requirements.txt