import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
from src.config import DEVICE, MAX_LEN
from src.data_loader import tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro'),
    }

def get_preds(model, texts, is_mtl=False, batch_size=32):
    model.eval()
    all_preds = []

    texts = list(texts)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
            batch = [str(x) for x in texts[i:i+batch_size]]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN
            ).to(DEVICE)

            if is_mtl:
                outputs = model(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    task='sentiment'
                )
            else:
                outputs = model(**inputs)

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return all_preds