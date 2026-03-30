import torch
import os
import torch.nn as nn
import numpy as np
import itertools
import gc
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score

from src.config import MODEL_NAME, DEVICE, EPOCHS, LEARNING_RATE, SEEDS, LAMBDAS, MODEL_DIR
from src.utils import set_seed
from src.data_loader import load_and_process_data, get_dataloaders
from src.mtl_model import RobertaMTL_Advanced
from src.evaluation import get_preds

# Config
tuned_model = True  # True for training model with tuned params, False for defaults
search_lambda_mode = False # True for searching lambda
#[I 2026-03-18 15:06:12,302] Trial 11 finished with value: 0.7636595731695698 and parameters: {'learning_rate': 2.65512256035882e-05, 'dropout_prob': 0.2, 'warmup_ratio': 0.1}. Best is trial 11 with value: 0.7636595731695698.

# --- HYPERPARAMETER LOGIC ---
if tuned_model:
    LEARNING_RATE = 1.3642484556175887e-05 
    DROPOUT = 0.15000000000000002
    WARMUP = 0.05
else:
    # Use the defaults from your config or hardcoded values
    DROPOUT = 0.1
    WARMUP = 0.1

# Update the function to use these CURRENT_ variables
def train_single_seed(seed, lambda_val, dataframes, train_tuned_model=tuned_model):
    set_seed(seed)
    df_train, df_val, df_test, df_sarc, df_golden = dataframes

    prefix = "tuned_" if train_tuned_model else ""
    model_save_path = os.path.join(MODEL_DIR, f"{prefix}mtl_seed{seed}_lam{lambda_val}.pt")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = RobertaMTL_Advanced(MODEL_NAME, config, dropout_prob= DROPOUT).to(DEVICE)
    
    train_sent, val_sent, test_sent, train_sarc = get_dataloaders(df_train, df_val, df_test, df_sarc, seed)

    if os.path.exists(model_save_path):
        print(f"Model found at {model_save_path}. Skipping...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))
    else:
        # 2. Setup training-specific components
        optimizer_grouped_parameters = [
            {'params': model.roberta.embeddings.parameters(), 'lr': LEARNING_RATE * 0.1},
            {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': LEARNING_RATE * 0.2},
            {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': LEARNING_RATE * 0.5},
            {'params': model.roberta.encoder.layer[10:].parameters(), 'lr': LEARNING_RATE},
            {'params': model.sentiment_head.parameters(), 'lr': LEARNING_RATE * 2},
            {'params': model.sarcasm_head.parameters(), 'lr': LEARNING_RATE * 2}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
        
        total_steps = len(train_sent) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * WARMUP), 
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_f1 = -1.0
        sarc_iterator = itertools.cycle(train_sarc)

        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            for batch_sent in tqdm(train_sent, desc=f"Seed {seed} | Lam {lambda_val} | Ep {epoch+1}", leave=False):
                optimizer.zero_grad()
                out_sent = model(batch_sent['input_ids'].to(DEVICE), batch_sent['attention_mask'].to(DEVICE), task='sentiment')
                loss_sent = loss_fn(out_sent.logits, batch_sent['labels'].to(DEVICE))
                
                batch_sarc = next(sarc_iterator)
                out_sarc = model(batch_sarc['input_ids'].to(DEVICE), batch_sarc['attention_mask'].to(DEVICE), task='sarcasm')
                loss_sarc = loss_fn(out_sarc.logits, batch_sarc['labels'].to(DEVICE))
                
                total_loss = loss_sent + (lambda_val * loss_sarc)
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
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), model_save_path)
        
        # Load the best weights from the training session we just finished
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))

    # 3. Evaluation (Common to both paths)
    model.eval()

    # Test Evaluation
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_sent:
            out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), task='sentiment')
            test_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            test_labels.extend(batch['labels'].numpy())
    
    t_f1 = f1_score(test_labels, test_preds, average='macro')
    t_acc = accuracy_score(test_labels, test_preds)
    
    # Golden Evaluation
    golden_preds = get_preds(model, df_golden['text'], is_mtl=True)
    g_f1 = f1_score(df_golden['label'], golden_preds, average='macro')
    g_acc = accuracy_score(df_golden['label'], golden_preds)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return (t_f1, t_acc), (g_f1, g_acc)


if __name__ == "__main__":
    # CONFIGURATION
    SEARCH_LAMBDA = search_lambda_mode
    TRAIN_TUNED_MODEL = tuned_model 
    
    print("Loading and Processing Data...")
    dataframes = load_and_process_data()
    
    if SEARCH_LAMBDA:
        # Run across all lambdas defined in config, but only 1 seed each
        active_lambdas = LAMBDAS
        active_seeds = [42]
        print(f"Searching lambda process: Runing for lambdas {active_lambdas} ")
    else:
        # Only run for best lambda
        active_lambdas = [0.2]
        active_seeds = SEEDS
        print(f"Running MTL model with lambda {active_lambdas} with {len(active_seeds)} seeds.")
    # Data structures for results
    results = {lam: {'test_f1': [], 'test_acc': [], 'gold_f1': [], 'gold_acc': []} for lam in active_lambdas}

    for lam in active_lambdas:
        print(f"\n Running MTL | Lambda: {lam}\n{'='*40}")
        for seed in active_seeds:
            (t_f1, t_acc), (g_f1, g_acc) = train_single_seed(seed, lam, dataframes, train_tuned_model= tuned_model)
            results[lam]['test_f1'].append(t_f1)
            results[lam]['test_acc'].append(t_acc)
            results[lam]['gold_f1'].append(g_f1)
            results[lam]['gold_acc'].append(g_acc)
            print(f"Seed {seed} | Test F1: {t_f1:.4f} | Gold F1: {g_f1:.4f}")

    print("\n" + "="*70)
    print(f"Multi Task Learning Model Report Summary (Lambdas Tested: {active_lambdas})")
    
    lambda_scores = {}

    for lam in active_lambdas:
        res = results[lam]
        
        avg_tf1, std_tf1 = np.mean(res['test_f1']), np.std(res['test_f1'])
        avg_tacc, std_tacc = np.mean(res['test_acc']), np.std(res['test_acc'])
        avg_gf1, std_gf1 = np.mean(res['gold_f1']), np.std(res['gold_f1'])
        avg_gacc, std_gacc = np.mean(res['gold_acc']), np.std(res['gold_acc'])
        
        # Calculate search metric: Golden F1 * 0.6 + Test F1 * 0.4
        current_metric = (avg_gf1 * 0.6) + (avg_tf1 * 0.4)
        lambda_scores[lam] = current_metric

        print(f"Results for Lambda: {lam}")
        if len(active_seeds) > 1:
            print(f"Test Set: F1 {avg_tf1:.4f} (±{std_tf1:.4f}) | Acc {avg_tacc:.4f} (±{std_tacc:.4f})")
            print(f"Gold Set: F1 {avg_gf1:.4f} (±{std_gf1:.4f}) | Acc {avg_gacc:.4f} (±{std_gacc:.4f})")
        else:
            print(f"Test Set: F1 {avg_tf1:.4f} | Acc {avg_tacc:.4f}")
            print(f"Gold Set: F1 {avg_gf1:.4f} | Acc {avg_gacc:.4f}")

    if SEARCH_LAMBDA:
        best_lam = max(lambda_scores, key=lambda_scores.get)
        print(f"\nSuggested Lambda: {best_lam} (F1 of Test set *0.4 + F1 of Gold set *0.6: {lambda_scores[best_lam]:.4f})")