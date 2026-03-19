# Intel Image Classification: A Comparative Deep Learning Study

## Project Overview
This project evaluates multiple Convolutional Neural Network (CNN) strategies for classifying landscape images into six categories: **Buildings, Forest, Glacier, Mountain, Sea, and Street.** Using the Intel Image Dataset, our team benchmarked a custom Baseline CNN, an Engineered/Improved CNN, and a Transfer Learning approach to identify the most robust architecture for low-resolution (42x42) image recognition.

## Key Architectures
### 1. Baseline CNN
* **Structure:** Simple 3-block convolutional architecture.
* **Findings:** Rapid convergence but significant overfitting after 5 epochs, highlighting the need for stronger regularization in small-scale image tasks.

### 2. Improved & Optimized CNN
* **Enhancements:** Increased filter depth (32 → 128), added **Batch Normalization**, and integrated **Dropout (0.4)**.
* **Optimization:** Used **Keras Tuner with Bayesian Optimization** to refine learning rates and layer units.
* **Result:** Achieved the project's highest test accuracy of **81.3%** by smoothing the loss landscape and stabilizing gradients.

### 3. Transfer Learning (DenseNet121)
* **Strategy:** Leveraged a pretrained DenseNet121 backbone (ImageNet weights) with a custom classification head.
* **Findings:** While powerful, the frozen weights and low input resolution (42x42) limited its ability to outperform the task-specific custom CNN.

## Performance Comparison
| Model | Accuracy | F1-Score (Macro) | Performance Note |
| :--- | :--- | :--- | :--- |
| Baseline | 77.0% | 0.77 | Significant overfitting after Epoch 5. |
| **Improved CNN** | **81.3%** | **0.81** | **Best balance of generalization & stability.** |
| Transfer Learning| 77.0% | 0.77 | Strongest on 'Forest' (0.93 F1), weakest on 'Glacier'. |

## Technical Analysis & Discussions
* **Class-Based Variation:** Across all models, **'Forest'** was the top-performing class (0.98 precision) due to distinct textures. **'Glacier'** and **'Mountain'** presented the greatest challenge (0.71 recall) due to overlapping semantic features (snow, jagged rocks).
* **Regularization vs. Depth:** We found that adding depth beyond 3 blocks caused overfitting. Instead, "widening" the model and using **Batch Normalization** provided better gains.
* **Optimization Choice:** Bayesian Optimization was prioritized over Random Search for its efficiency in navigating the hyperparameter space based on prior trial results.

## 🛠️ Tech Stack
* **Frameworks:** TensorFlow, Keras
* **Optimization:** Keras Tuner (Bayesian Optimization)
* **Visualization:** Matplotlib, Seaborn (Confusion Matrices, ROC/AUC Curves)
* **Dataset:** Intel Image Classification (Kaggle)
