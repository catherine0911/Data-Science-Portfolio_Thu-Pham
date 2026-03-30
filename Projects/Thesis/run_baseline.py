import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import MODEL_NAME, DEVICE, OUTPUT_DIR, SEEDS, MODEL_DIR
from src.utils import set_seed
from src.data_loader import load_and_process_data, get_hf_datasets
from src.evaluation import compute_metrics, get_preds
from sklearn.metrics import accuracy_score, f1_score

def run_baseline_experiment():
    # 1. Load Data 
    df_train, df_val, df_test, _, df_golden = load_and_process_data()
    train_ds, val_ds, test_ds = get_hf_datasets(df_train, df_val, df_test)

    # CREATE SUBFOLDER: models/baseline_models
    baseline_dir = os.path.join(MODEL_DIR, "baseline_models")
    os.makedirs(baseline_dir, exist_ok=True)

    test_f1s, test_accs = [], []
    gold_f1s, gold_accs = [], []

    print(f"Baseline Experiment across seeds: {SEEDS}")

    for seed in SEEDS:
        set_seed(seed)
        
        # Define the specific path for this seed
        seed_model_path = os.path.join(baseline_dir, f"baseline_seed_{seed}")
        
        # CHECK IF MODEL EXISTS
        if os.path.exists(os.path.join(seed_model_path, "config.json")):
            print(f"Found existing model at {seed_model_path}. Skipping training...")
            model = AutoModelForSequenceClassification.from_pretrained(seed_model_path).to(DEVICE)
        else:
            print(f"Starting training for Seed {seed}...")
            # 2. Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=3,
                use_safetensors=True
            ).to(DEVICE)

            # 3. Trainer Setup
            training_args = TrainingArguments(
                output_dir=os.path.join(OUTPUT_DIR, f"temp_baseline_seed_{seed}"),
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True, 
                report_to="none",
                seed=seed,
                data_seed=seed
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
            )

            # 4. Train
            trainer.train()

            # 5. Save the model
            print(f"Saving baseline model to {seed_model_path}")
            trainer.save_model(seed_model_path) 

        # 6. Evaluate (Runs for both newly trained and loaded models)
        model.eval()
        
        # Performance on Standard Test Set
        with torch.no_grad():
            # Get predictions for test set
            test_preds = []
            test_labels = []
            for i in range(0, len(df_test), 16):
                batch_texts = df_test['text'].iloc[i:i+16].tolist()
                batch_labels = df_test['label'].iloc[i:i+16].tolist()
                preds = get_preds(model, batch_texts, is_mtl=False)
                test_preds.extend(preds)
                test_labels.extend(batch_labels)
        
        t_f1 = f1_score(test_labels, test_preds, average='macro')
        t_acc = accuracy_score(test_labels, test_preds)
        test_f1s.append(t_f1)
        test_accs.append(t_acc)

        # 7. Evaluate Golden Set
        gold_preds = get_preds(model, df_golden['text'], is_mtl=False)
        g_f1 = f1_score(df_golden['label'], gold_preds, average='macro')
        g_acc = accuracy_score(df_golden['label'], gold_preds)
        
        gold_f1s.append(g_f1)
        gold_accs.append(g_acc)
        
        print(f"Seed {seed} | Test F1: {t_f1:.4f} | Golden F1: {g_f1:.4f}")

        # Cleanup memory
        del model
        torch.cuda.empty_cache()

    # 8. Final Averaging & Reporting
    print("\n" + "="*40)
    print("Baseline Performance Summary:")
    print(f"Test Set: F1 {np.mean(test_f1s):.4f} (±{np.std(test_f1s):.4f}) | Acc {np.mean(test_accs):.4f}")
    print(f"Gold Set: F1 {np.mean(gold_f1s):.4f} (±{np.std(gold_f1s):.4f}) | Acc {np.mean(gold_accs):.4f}")

if __name__ == "__main__":
    run_baseline_experiment()