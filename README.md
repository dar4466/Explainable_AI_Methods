# XAIMethodsAnalysis
- This notebook incXludes the implementation of most popular XAI Methods 

# Explainable AI (XAI) Analysis

This repository contains a comprehensive Jupyter/Colab Notebook demonstrating various state-of-the-art Explainable Artificial Intelligence (XAI) techniques. The goal of this project is to showcase how to interpret and explain the decision-making processes of different machine learning and deep learning models.

## 📌 Overview

As machine learning models become more complex (e.g., deep neural networks, ensemble methods), they often turn into "black boxes." This notebook explores multiple XAI frameworks to make these models more transparent, trustworthy, and understandable.

## 🛠️ Methods Covered

The notebook includes practical implementations of the following XAI techniques across tabular and image datasets:

1. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Explaining Random Forest predictions on tabular data.
   - Explaining ResNet50 predictions on image data.
2. **SHAP (SHapley Additive exPlanations)**
   - Using `DeepExplainer` for MNIST digit classification (TensorFlow/Keras).
   - Using `TreeExplainer` for Iris classification (Random Forest).
3. **Counterfactual Explanations**
   - Utilizing the `alibi` library on Decision Trees and Logistic Regression models.
4. **Permutation Feature Importance**
   - Evaluating feature significance for a Random Forest model on the California Housing dataset.
5. **t-SNE (t-distributed Stochastic Neighbor Embedding)**
   - Dimensionality reduction and visualization of the Wine and Iris datasets.
6. **ELI5**
   - Visualizing model weights for a Logistic Regression classifier.
7. **R-Squared Analysis**
   - Evaluating a Linear Regression model on the Diabetes dataset.
8. **Sensitivity Analysis (Sobol Indices)**
   - Using `SALib` to calculate first-order and total sensitivity indices on California Housing, Diabetes datasets, and the Ishigami test function.
9. **Layer-wise Relevance Propagation (LRP)**
   - Using `innvestigate` to generate relevance heatmaps for a custom neural network trained on MNIST.
10. **Occlusion Sensitivity**
    - Masking regions of an image to see how it affects a pre-trained VGG16 model's predictions.

## 💻 Prerequisites and Dependencies

To run the notebook locally, you will need Python 3.x and the following libraries. Most of these can be installed via `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
pip install lime shap alibi eli5 SALib innvestigate scikit-image
