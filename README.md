# CNN Image Classification — Fungi Detection

A computer vision project implementing a Convolutional Neural Network (CNN) 
using TensorFlow and Keras to classify fungi images — demonstrating deep 
learning architecture design, training dynamics analysis, and responsible 
deployment considerations for image classification systems.

---

## Business Context

Image classification models are increasingly deployed in high-stakes 
contexts — medical diagnosis, quality control, security screening, and 
environmental monitoring. Understanding how CNNs learn, where they fail, 
and what governance controls are needed before deployment is as important 
as achieving high accuracy on a test set.

This project builds a binary image classifier for fungi detection, 
documents training dynamics across epochs, and examines the governance 
implications of deploying computer vision models in real-world contexts.

---

## What It Does

- Loads and preprocesses fungi image data from external sources using 
  `requests` and `pickle`
- Builds a CNN architecture with Conv2D, MaxPooling2D, Flatten, and 
  Dense layers using TensorFlow/Keras
- Trains with Adam optimizer and sparse categorical crossentropy loss
- Monitors training vs. validation accuracy and loss across 10 epochs
- Documents epoch-by-epoch performance trends and convergence behavior
- Analyzes limitations and recommended improvements

---

## Model Architecture

Input: Preprocessed fungi images
↓
Conv2D (32 filters, 3×3 kernel, ReLU)
↓
MaxPooling2D (2×2)
↓
Flatten
↓
Dense (64 units, ReLU)
↓
Output: Dense (2 units, Sigmoid) → binary classification

**Optimizer:** Adam  
**Loss:** Sparse categorical crossentropy  
**Epochs:** 10  
**Peak validation accuracy:** ~76% (Epoch 4)

---

## Training Analysis

| Epoch | Behavior |
|-------|----------|
| 1 | Low accuracy (~44%), high loss — model underfitting, random initialization |
| 2–4 | Significant improvement — model learning spatial features, validation accuracy reaches 76% |
| 5–7 | Stabilization — accuracy 72–74%, loss decreasing slowly, model converging |
| 8–10 | Plateau — no significant improvement, model approaching capacity limit |

**Key observation:** The gap between training and validation performance 
remained small throughout, suggesting reasonable generalization without 
severe overfitting. However, plateauing at 76% indicates the architecture 
needs improvement before production use.

---

## Honest Assessment of Results

76% validation accuracy on a binary classification task is a proof of 
concept, not a production-ready result. For context:

- A random classifier would achieve ~50% on a balanced binary dataset
- A production medical imaging classifier typically requires 90%+ with 
  formal clinical validation
- The plateau at epoch 8–10 suggests the current architecture has 
  reached its learning capacity with the given data

This is documented honestly because deploying an underperforming model 
in a high-stakes context is a governance failure, not just a technical 
one.

---

## Recommended Improvements

**Data Augmentation**  
Rotation, zoom, horizontal flip, and brightness variation would increase 
effective training set size and improve generalization to real-world 
image variation.

**Deeper Architecture**  
Additional convolutional blocks (Conv2D → MaxPooling2D → Conv2D → 
MaxPooling2D) would enable the model to learn more complex spatial 
features at multiple scales.

**Regularization**  
Dropout layers between Dense layers would reduce overfitting risk as 
the model is deepened.

**Transfer Learning**  
Using a pretrained model (ResNet50, EfficientNet, MobileNetV2) as a 
feature extractor would dramatically improve accuracy with limited 
training data — a standard approach in production computer vision.

**Cross-Validation**  
K-fold cross-validation would provide a more robust accuracy estimate 
than a single train/validation split.

---

## Governance & Responsible Deployment Notes

**Accuracy thresholds before deployment**  
76% accuracy is insufficient for any consequential application. Before 
deployment, minimum accuracy thresholds should be defined based on the 
cost of false positives vs. false negatives in the specific use case. 
For fungi classification in a food safety context, a false negative 
(missing a toxic species) has far greater consequences than a false 
positive.

**Dataset bias**  
Image classifiers are vulnerable to dataset bias — if training images 
were collected under specific lighting conditions, angles, or backgrounds, 
the model may fail on images that look different. Distribution shift 
between training data and deployment conditions is a primary failure 
mode for computer vision systems.

**Explainability**  
CNNs are black boxes — it is not immediately obvious which image features 
drive a classification decision. Tools like Grad-CAM (Gradient-weighted 
Class Activation Mapping) can visualize which regions of an image the 
model is attending to, providing a form of explainability for human 
reviewers.

**Human oversight**  
For any consequential classification task (medical, legal, safety), 
model predictions should be reviewed by a qualified human before action 
is taken. The model provides a signal — the human makes the decision.

---

## Origin

This project was developed as part of the Ohio State University 
AI & ML Bootcamp (2024) and expanded with technical analysis and 
governance framing for portfolio purposes.

---

## Author

**Steven Hill**  
AI Ethics & Policy Professional | Purdue University MSAI  
[LinkedIn](https://linkedin.com/in/stevenrhill) | 
[GitHub](https://github.com/srhill12)

