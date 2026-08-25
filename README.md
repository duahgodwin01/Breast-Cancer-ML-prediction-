# Breast Cancer Prediction using Machine Learning

This project classifies breast tumors as benign or malignant using classical supervised machine learning. Recall was treated as the most important metric throughout, since in this context a missed diagnosis is far more costly than a false alarm.

**Team:** **Godwin Gyamfi Duah**, Feranmi Paul Adepoju, Khadka Ujjwal

**School:** Harrisburg University of Science and Technology 

**Course:** CISC 525 - Big Data Architectures

**Date:** April 2026

**Supervisor:** Sangwhan Cha, PhD 

## Overview

Breast cancer resulted in roughly 670,000 deaths annually as of 2022 and affects about 2.1 million women worldwide each year, according to the World Health Organization. The American Cancer Society's Breast Cancer Facts & Figures 2024 to 2025 report provides the broader epidemiological picture behind these numbers. Catching it early makes a real difference to survival, but traditional statistical methods often struggle to pick up the complex, non-linear patterns hiding in large clinical datasets.

Prior work backs up the case for this approach. Fatima et al. (2020) found that machine learning models, particularly SVMs and ensemble methods, consistently outperform traditional statistical classifiers on this type of task. Rabiei et al. (2022) reached a similar conclusion, showing that ensemble methods like Random Forest handle class imbalance and high-dimensional feature spaces especially well. Deep learning has also shown promise here. Ribli et al. (2018) demonstrated that CNNs using transfer learning could reach radiologist-level performance on mammogram images directly, which is part of why that direction shows up later in this project's future work.

This project trains and compares four classical machine learning algorithms on the Breast Cancer Wisconsin Diagnostic Dataset to classify tumors as benign or malignant. The model comparison itself wasn't just about picking the highest accuracy. It was about choosing the metric that actually matters clinically, and letting that guide which model earns the top spot. Stark et al. (2019) make the case for exactly this kind of thinking, arguing that clinical adoption depends as much on interpretability and calibration as it does on raw predictive power.

## Dataset

The data comes from the [Breast Cancer Wisconsin Diagnostic Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?select=breast-cancer), originally assembled by Dr. William H. Wolberg at the University of Wisconsin, Madison. It contains 569 patient records and 30 numeric features drawn from fine needle aspirate images of breast masses, split across two classes: benign and malignant.

The most clinically relevant features include radius mean, texture mean, concavity mean, area mean, smoothness mean, and compactness mean, each capturing a different aspect of tumor shape and structure.

## Methods

Four models were trained on a stratified 80/20 train-test split, with features standardized wherever the model's math depended on distance or margins.

Logistic Regression served as the baseline linear classifier. Support Vector Machine used an RBF kernel on the scaled features. Random Forest used 300 trees and was trained on the unscaled data, since tree-based models don't need standardized inputs, and it also powered the feature importance analysis. Linear Discriminant Analysis rounded out the comparison, also trained on scaled features.

Recall was the primary evaluation metric across the board. In this clinical setting, a false negative means a malignant tumor gets missed entirely, and that risk far outweighs the cost of a false positive. That priority shaped which model was considered the "best" one, not just how the numbers were reported afterward.

Every model was also checked with five-fold cross-validation, scored on recall, to confirm the results weren't just a lucky split. Learning curves were used to rule out overfitting, and a feature importance analysis on the Random Forest model helped identify which measurements actually drove its predictions.

## Results

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.929 | 0.975 | 0.951 |
| SVM | 0.974 | 0.929 | 1.000 | 0.963 |
| Random Forest | 0.974 | 0.929 | 1.000 | 0.963 |
| LDA | 0.965 | 0.905 | 1.000 | 0.950 |

SVM and Random Forest came out on top overall. Cross-validation backed this up, with SVM reaching a recall of 95.3% and Random Forest close behind at 93.5%, and there was no meaningful gap between test performance and cross-validated performance for any of the four models.

The ROC AUC scores tell a slightly different story worth noting. All four models scored between 0.994 and 0.997, with LDA actually coming out slightly ahead (0.997), followed by Logistic Regression (0.996), SVM (0.995), and Random Forest (0.994). That's a reminder that a model's ability to rank cases correctly across every possible threshold, which is what AUC measures, isn't the same as its performance at the single threshold actually used to make predictions, which is what the recall numbers above reflect. LDA ranks cases well overall but loses some ground once a hard cutoff is applied.

Looking at feature importance from the Random Forest model, the "worst" measurements, meaning the most extreme values recorded for each cell nucleus, dominated the top of the list. Perimeter worst and area worst ranked highest, followed by concave points worst, concave points mean, and radius worst. This suggests the most severe cell abnormalities in a sample carry more predictive weight than average measurements across the tumor, which lines up well with existing clinical research on what tumor shape tells us about malignancy.

## Why this isn't just an AutoML task

It's fair to ask whether an AutoML platform could have produced the same result with less effort. It can approximate the pipeline, but not the judgment calls behind it. Deciding to prioritize recall over accuracy, weighing the tradeoff between false positives and false negatives, and figuring out what "good enough" actually means in a clinical setting are domain-specific decisions. Generic platforms don't make those calls without a lot of custom configuration, and healthcare deployments also demand a level of interpretability and documentation that off-the-shelf AutoML tools don't provide by default.

## Conclusion

This project shows that classical machine learning models, without any deep learning or image analysis, can classify breast tumors with a high degree of accuracy and, more importantly, a high degree of recall. SVM and Random Forest were the strongest overall, but every model performed well, which reflects how much predictive signal these 30 features carry once properly prepared.

The more important takeaway isn't which algorithm won. It's that prioritizing recall over accuracy from the outset shaped every decision that followed, from which model was considered "best" to how the results were interpreted. That kind of domain-informed judgment is what separates a project built for a genuine clinical use case from one optimized purely for benchmark performance.

## Repository Structure

The repo includes `breast_cancer_prediction.py`, which holds the full training and evaluation pipeline, a `results` folder containing the confusion matrices, ROC curves, learning curves, and feature importance plots, and this README.

## Tech Stack

Python, Pandas, Scikit-learn, Matplotlib, Google Colab

## Future Work

There's room to extend this with convolutional neural networks for direct mammographic image analysis using transfer learning, and to explore a hybrid architecture that combines imaging data with structured electronic health record data. An interactive dashboard for clinicians, complete with confidence scores, would make this genuinely usable in practice, and adding SHAP or Grad-CAM explainability would help build the kind of trust clinicians need before relying on a model like this. Longer-term, a FHIR-compliant integration path would make it feasible to connect this to real hospital systems.

## References

Fatima, N., et al. (2020). Prediction of Breast Cancer. IEEE Access. https://doi.org/10.1109/ACCESS.2020.3016715

Rabiei, R., et al. (2022). Prediction of Breast Cancer using Machine Learning Approaches. Journal of Biomedical Physics & Engineering.

Ribli, D., et al. (2018). Detecting and classifying lesions in mammograms with Deep Learning. Scientific Reports.

Stark, G. F., et al. (2019). Predicting breast cancer risk. PLOS ONE. https://doi.org/10.1371/journal.pone.0226765

American Cancer Society. Breast Cancer Facts & Figures 2024 to 2025.
