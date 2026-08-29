# Heart Disease Classification using Neural Networks

## Overview

This project implements a multi-layer perceptron (MLP) neural network using scikit-learn to classify the presence and severity of heart disease from clinical data. The primary focus is investigating the effects of key hyperparameters on model performance, including network architecture, learning rate, and regularization strength. This work was contributed as part of the *Mathematics by Computer* neural network group project.

## Table of Contents
- [Overview](#overview)
- [Installation & Dependencies](#installation--dependencies)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Hyperparameter Analysis](#hyperparameter-analysis)
- [Conclusions](#conclusions)
- [References](#references)

## Installation & Dependencies

**Environment:** Python 3.x with Jupyter Notebook

**Required Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning and neural network implementation
- `matplotlib` - Data visualization

## Getting Started

### Prerequisites
Ensure you have Python 3.x installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/luca-wheeler/heart-disease-neural-network.git
   cd heart-disease-neural-network
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```
   
   Or, if using conda:
   ```bash
   conda install pandas numpy scikit-learn matplotlib jupyter
   ```

### Running the Project

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   Navigate to and open `MLP_SectionB_lucawheelerV2.ipynb` in your browser

3. **Ensure data is present:**
   Verify that `cleveland.csv` is in the same directory as the notebook

4. **Execute cells:**
   Run cells sequentially from top to bottom. The notebook is organized into logical sections:
   - Data loading and preprocessing
   - Model building and training functions
   - Optimized model evaluation
   - Hyperparameter analysis (hidden layers, learning rate, regularization)

### Project Structure
```
heart-disease-neural-network/
├── README.md                              # Project documentation
├── MLP_SectionB_lucawheelerV2.ipynb       # Main Jupyter notebook
└── cleveland.csv                          # UCI Heart Disease Dataset
```

## Dataset

### Source
The **UCI Cleveland Heart Disease Dataset** was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease). The processed Cleveland database was converted to CSV format for this analysis.

### Features
The dataset comprises 14 clinical variables:

| # | Feature | Description |
|---|---------|-------------|
| 1 | age | Patient age (years) |
| 2 | sex | Patient sex (binary) |
| 3 | cp | Chest pain type (categorical) |
| 4 | trestbps | Resting blood pressure (mmHg) |
| 5 | chol | Serum cholesterol (mg/dl) |
| 6 | fbs | Fasting blood sugar > 120 mg/dl (binary) |
| 7 | restecg | Resting electrocardiographic results (categorical) |
| 8 | thalach | Maximum heart rate achieved (bpm) |
| 9 | exang | Exercise-induced angina (binary) |
| 10 | oldpeak | ST depression induced by exercise (continuous) |
| 11 | slope | Slope of peak exercise ST segment (categorical) |
| 12 | ca | Number of major vessels colored by fluoroscopy |
| 13 | thal | Thalassemia (categorical) |
| 14 | num | **Target**: Diagnosis rating (0 = no disease, 1-4 = varying degrees of presence) |

### Data Preprocessing
1. **Missing Value Removal:** Records with missing values (represented as '?') were removed
2. **Type Conversion:** All features were converted to numeric values
3. **Feature Standardization:** Input features were standardized using `StandardScaler` with zero mean and unit variance to improve gradient descent convergence and balance feature importance

Train-test split: 75% training (220 samples) and 25% test (73 samples) with stratification to maintain class distribution.

## Methodology

### Cost Function
The model optimization uses cross-entropy loss (log loss), defined as:

$$L_{\log}(Y, P) = -\frac{1}{N} \sum_{i=0}^{N-1} \sum_{k=0}^{K-1} y_{i,k} \log p_{i,k}$$

where:
- $y_{i,k}$ is the one-hot encoded true label for sample $i$ and class $k$
- $p_{i,k}$ is the predicted probability for class $k$
- $N$ is the number of samples and $K$ is the number of classes (5)

### Training Procedure
The model was trained using stochastic gradient descent with batch processing:
- **Batch Size:** 5 samples
- **Batches per Epoch:** 44 (220 training samples ÷ 5)
- **Metrics:** Accuracy and cross-entropy loss computed for both training and test sets after each epoch

## Model Architecture

The baseline model uses a three-layer hidden network:

**Architecture:** $13 \to 8 \to 4 \to 2 \to 5$

**Hyperparameters:**
- **Hidden Layers:** (8, 4, 2) – 142 learnable parameters
- **Activation Function:** ReLU (Rectified Linear Unit): $f(x) = \max(x, 0)$
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum
- **Learning Rate:** 0.000056 (constant)
- **L2 Regularization (α):** 0.09
- **Momentum:** 0.9 (Nesterov)

**Total Parameters:** $13 \times 8 + 8 \times 4 + 4 \times 2 + 2 \times 5 = 142$

## Results

### Optimized Model Performance

**Final Results (at convergence, epoch 300):**
- Test Accuracy: **56%**
- Test Loss: Steady convergence with minimal overfitting

**Model Behavior Over Training:**

- **Epochs 0-80:** Underfitting phase with high loss and low accuracy on both sets
- **Epochs 80-200:** Rapid convergence with accuracy spike and loss stabilization
- **Epochs 200-300:** Continued gradual improvement with minimal divergence between train and test metrics
- **Epochs 300+:** Diminishing returns; training loss begins increasing beyond epoch 350, indicating overshooting of the cost minimum

The model achieves reasonable generalization with no significant overfitting through epoch 300, making this the optimal stopping point.

## Hyperparameter Analysis

Systematic investigation of three key hyperparameters was performed to understand their effects on model performance:

### 1. Network Architecture (Hidden Layers)

#### No Hidden Layers
- **Parameters:** 13 × 5 = 65
- **Observation:** Convergence reached by ~100 epochs due to reduced complexity. Underfitting evident from low accuracy; slight overfitting observed at later epochs with train accuracy exceeding test accuracy.

#### Large Hidden Layers (50, 25, 10)
- **Parameters:** 2,160
- **Observation:** Severe overfitting with training accuracy continuously increasing while test accuracy plateaus. Training loss decreases while test loss diverges, indicating the model memorizes training data rather than learning generalizable patterns.

#### Multiple Smaller Hidden Layers (10, 8, 6, 4, 2)
- **Parameters:** 228
- **Observation:** Complex architecture leads to rapid initial learning followed by plateau. Model exhibits good generalization (similar train/test accuracy) but gets stuck in a local minimum, unable to continue improving.

**Finding:** Moderate-sized networks with 2-3 hidden layers provide the best balance between model complexity and generalization capability.

### 2. Learning Rate

#### High Learning Rate (0.001)
- **Observation:** Rapid convergence with overfitting. Train-test accuracy divergence emerges after 200 epochs. Fluctuating loss curves indicate the optimizer overshoots the cost minimum.

#### Moderate Learning Rate (0.0001)
- **Observation:** Slower convergence (~350 epochs) with mild overfitting. Slight cost bump at epoch 325 suggests the optimizer momentarily overshoots the minimum.

#### Low Learning Rate (0.00001)
- **Observation:** Severe underfitting. After 2000 epochs, the model still hasn't converged due to excessively small gradient steps. Computationally inefficient.

**Finding:** A moderate learning rate (0.00005-0.0001) balances convergence speed with accuracy, while adaptive learning rate schedules could further improve performance.

### 3. Regularization Strength (L2 Regularization)

#### Weak Regularization (α = 0.0001)
- **Observation:** Overfitting evident from diverging test loss after 1000 epochs. Test cost increases while training cost decreases, indicating the model is not generalizing.

#### Low-Moderate Regularization (α = 0.01)
- **Observation:** Improved generalization compared to α = 0.0001, but overfitting still occurs. Test loss diverges less rapidly.

#### Moderate Regularization (α = 0.1)
- **Observation:** Strong generalization with train and test metrics remaining close through 400 epochs. Good balance between model complexity penalization and learning capability.

#### Strong Regularization (α = 1.0)
- **Observation:** Underfitting with rapid convergence to low accuracy (~0.4). High weights are heavily penalized, resulting in an overly simple model that cannot capture data complexity.

**Finding:** Regularization strength of 0.09-0.1 provides optimal generalization for this dataset and architecture.

## Conclusions

### Key Findings

1. **Network Architecture:** Moderate complexity (2-3 hidden layers with 8-16 nodes per layer) outperforms both shallow networks (underfitting) and deep networks (overfitting). The chosen (8, 4, 2) architecture represents an effective balance.

2. **Learning Rate:** Constant learning rates in the range 0.00005-0.0001 enable timely convergence without overshooting. Adaptive learning rate schedules may further improve optimization trajectories.

3. **Regularization:** L2 regularization strength between 0.09-0.1 effectively prevents overfitting while maintaining model expressiveness. Higher values induce underfitting, while lower values permit overfitting.

### Limitations and Future Improvements

The achieved test accuracy of 56% is moderate, primarily constrained by:
- Multi-class classification complexity (5 classes: 0-4)
- Limited dataset size (~297 samples after cleaning)
- scikit-learn's MLPClassifier implementation limitations

**Recommended Improvements:**
1. **Binary Classification:** Reformulate as binary classification (presence vs. absence of heart disease) to reduce complexity and improve performance
2. **Ensemble Methods:** Implement ensemble approaches combining multiple models
3. **Advanced Architectures:** Explore deeper networks with batch normalization and dropout
4. **Hyperparameter Optimization:** Employ automated hyperparameter tuning (grid search, Bayesian optimization)
5. **Data Augmentation:** Apply domain-appropriate data augmentation techniques to increase training set size

## References

- UCI Machine Learning Repository: [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Kaggle Reference: [Heart Disease Prediction using Neural Networks](https://www.kaggle.com/code/bulentsiyah/heart-disease-prediction-using-neural-networks/notebook)
- Scikit-learn Documentation: [Neural Network Module](https://scikit-learn.org/stable/modules/neural_network.html)
