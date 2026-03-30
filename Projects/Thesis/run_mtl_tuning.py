import torch
import os
import torch.nn as nn
import numpy as np
import itertools
import gc
import optuna
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score

from src.config import MODEL_NAME, DEVICE, EPOCHS, MODEL_DIR
from src.utils import set_seed
from src.data_loader import load_and_process_data, get_dataloaders
from src.mtl_model import RobertaMTL_Advanced
from src.evaluation import get_preds

# Fixed values for tuning
TUNE_SEED = 42
FIXED_LAMBDA = 0.2
best_overall_f1 = -1.0
def train_for_trial(trial, dataframes):
    """Modified training loop that accepts Optuna trial suggestions and pruning."""
    global best_overall_f1
    set_seed(TUNE_SEED)
    df_train, df_val, df_test, df_sarc, df_golden = dataframes
    
    # 1. Ask Optuna for hyperparameters (UPDATED RANGES)
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.25, step=0.05)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15, step=0.05)
    
    train_sent, val_sent, test_sent, train_sarc = get_dataloaders(df_train, df_val, df_test, df_sarc, TUNE_SEED)
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = RobertaMTL_Advanced(MODEL_NAME, config, dropout_prob=dropout_prob).to(DEVICE)
    
    # LLRD Implementation using the trial's Learning Rate
    optimizer_grouped_parameters = [
        {'params': model.roberta.embeddings.parameters(), 'lr': lr * 0.1},
        {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': lr * 0.2},
        {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': lr * 0.5},
        {'params': model.roberta.encoder.layer[10:].parameters(), 'lr': lr},
        {'params': model.sentiment_head.parameters(), 'lr': lr * 2},
        {'params': model.sarcasm_head.parameters(), 'lr': lr * 2}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    total_steps = len(train_sent) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * warmup_ratio), 
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_f1 = -1.0
    best_state_dict = None  # UPDATED: Keep best weights in memory
    sarc_iterator = itertools.cycle(train_sarc)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for batch_sent in tqdm(train_sent, desc=f"Trial {trial.number} | Ep {epoch+1}", leave=False):
            optimizer.zero_grad()
            
            # Sentiment Forward
            out_sent = model(batch_sent['input_ids'].to(DEVICE), batch_sent['attention_mask'].to(DEVICE), task='sentiment')
            loss_sent = loss_fn(out_sent.logits, batch_sent['labels'].to(DEVICE))
            
            # Sarcasm Forward
            batch_sarc = next(sarc_iterator)
            out_sarc = model(batch_sarc['input_ids'].to(DEVICE), batch_sarc['attention_mask'].to(DEVICE), task='sarcasm')
            loss_sarc = loss_fn(out_sarc.logits, batch_sarc['labels'].to(DEVICE))
            
            # Joint Backward
            total_loss = loss_sent + (FIXED_LAMBDA * loss_sarc)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_sent:
                out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), task='sentiment')
                val_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                val_labels.extend(batch['labels'].numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # UPDATED: Save best validation model in memory
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if val_f1 > best_overall_f1:
                best_overall_f1 = val_f1
                torch.save(best_state_dict, os.path.join(MODEL_DIR, "best_tuned_mtl_model.pt"))

        # UPDATED: Report to Optuna and Prune if terrible
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # --- EVALUATION ON UNSEEN DATA ---
    # UPDATED: Load weights from memory instead of disk
    model.load_state_dict(best_state_dict)
    model.eval()

    # Test Set Evaluation
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_sent:
            out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), task='sentiment')
            test_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            test_labels.extend(batch['labels'].numpy())
    
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    # Eval on Golden Set
    golden_preds = get_preds(model, df_golden['text'], is_mtl=True)
    golden_f1 = f1_score(df_golden['label'], golden_preds, average='macro')

    # Save extra metrics to Optuna dashboard
    trial.set_user_attr("test_f1", test_f1)
    trial.set_user_attr("golden_f1", golden_f1)

    # Cleanup
    del model, optimizer, scheduler, best_state_dict
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_f1

if __name__ == "__main__":
    dataframes = load_and_process_data()
    
    # Create Optuna study to maximize the validation F1
    # Adding a pruner explicitly (MedianPruner is very reliable)
    study = optuna.create_study(
        direction="maximize", 
        study_name="mtl_hyperparam_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1) # Don't prune on epoch 0
    )
    
    # Run 15 trials 
    study.optimize(lambda trial: train_for_trial(trial, dataframes), n_trials=15)

    # 2. Extract Best Results
    print("Optuna Tuning Result: ")
    print(f"Best Validation F1: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\nPerformance on Test set and Golden set for this best model:")
    print(f"  Test F1:   {study.best_trial.user_attrs.get('test_f1', 'N/A'):.4f}")
    print(f"  Golden F1: {study.best_trial.user_attrs.get('golden_f1', 'N/A'):.4f}")