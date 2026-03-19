# NLP 2026: Prompt-Based vs. Head-Based Fine-Tuning

## 📌 Project Overview
This project investigates two common approaches to fine-tuning Large Language Models (LLMs) for classification tasks: **Head-Based Tuning** and **Prompt-Based Tuning**. Inspired by the research paper ["How Many Data Points is a Prompt Worth?"](https://aclanthology.org/2021.naacl-main.208/), we replicate experiments to compare the sample efficiency and performance of these methods, specifically in low-resource settings.

## 🧪 Experiments
We utilized the **CommitmentBank (CB)** task from the SuperGLUE benchmark. CB is a natural language inference problem where the model predicts the semantic relationship (entailment, contradiction, or neutral) between a premise and a hypothesis.

### 1. Head-Based Tuning
* **Approach:** The base RoBERTa model is coupled with a newly initialized linear classification head.
* **Input Format:** Standard concatenation of premise and hypothesis using special separator tokens.

### 2. Prompt-Based Tuning
* **Approach:** Reformulates classification as a cloze-style masked language modeling problem, aligning the fine-tuning objective with the model's original pre-training.
* **Pattern:** `hypothesis? <mask>, premise`.
* **Verbalizer:** Maps predictions to single tokens: `entailment -> "yes"`, `contradiction -> "no"`, `neutral -> "maybe"`.

## Technical Implementation
* **Model:** RoBERTa (utilizing `RobertaForSequenceClassification` and `RobertaForMaskedLM`).
* **Optimization:** AdamW optimizer with gradient clipping.
* **Evaluation Metric:** Macro-F1 Score.
* **Frameworks:** PyTorch, HuggingFace Transformers, Datasets, and Evaluate.

## Key Definitions
* **Pre-training:** Self-supervised learning over massive corpora to capture task-agnostic language properties.
* **Fine-tuning:** Supervised adaptation to specific downstream tasks via gradient-based updates.

## Results & Findings
The experiments yielded the following insights regarding model adaptation:
* **Task Performance:** The model was evaluated on the **CommitmentBank (CB)** dataset, which is notoriously difficult due to its small size and three-class distribution (Entailment, Contradiction, Neutral).
* **Adaptation Comparison:** Prompt-based tuning typically showed higher stability in low-data regimes compared to traditional head-based tuning, as it leverages the model's pre-existing knowledge of language patterns more effectively.
* **Optimization:** Through the use of a `RobertaForMaskedLM` architecture for the prompt-based approach, the model achieved competitive Macro-F1 scores by predicting task-specific labels ("yes", "no", "maybe") directly in the latent space.