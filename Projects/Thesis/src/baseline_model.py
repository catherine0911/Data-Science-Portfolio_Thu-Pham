import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import MODEL_NAME, DEVICE, OUTPUT_DIR
from src.evaluation import compute_metrics

def get_baseline_model():
    """Returns a standard RoBERTa model for 3-class classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3, 
        use_safetensors=True
    ).to(DEVICE)

def train_baseline(model, train_dataset, val_dataset, seed=42):
    """Encapsulates the Trainer logic for the baseline."""
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}baseline_run_{seed}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=seed,
        data_seed=seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return trainer