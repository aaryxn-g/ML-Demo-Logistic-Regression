# Logistic Regression: Social Media Ads Classification

A comprehensive machine learning demo that predicts customer purchase behavior using logistic regression. This project demonstrates how demographic data (age and salary) can be used to classify potential customers and optimize marketing campaigns.

## Project Overview

This implementation uses a dataset of 400 social media users to predict whether they will purchase a product based on their age and estimated salary. The model achieves **89% accuracy** and provides valuable insights into customer behavior patterns.

### Key Results
- **Accuracy**: 89%
- **Features**: Age, Estimated Salary
- **Model**: Logistic Regression with Standard Scaling
- **Decision Boundary**: Linear separation in feature space

### Business Applications
- Targeted marketing campaigns
- Customer segmentation
- Lead scoring and prioritization
- Budget optimization for advertising

---

## Files

```
model-lr.py          # Main script with ML pipeline
social_ads.csv       # Dataset (400 users, 5 features)
requirements.txt     # Python dependencies
README.md            # This documentation
THEORY.md            # Comprehensive theoretical background
LICENSE              # MIT License
.gitignore           # Git ignore file
confusion_matrix.png # Model evaluation visualization
decision_boundary.png # Decision boundary plot
```

---

# Setup

**1. Create a virtual environment**

```bash
python3 -m venv venv
```

**2. Activate it**

On Mac/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

**3. Install the packages**

```bash
pip install -r requirements.txt
```

**4. Run the demo**

```bash
python model-lr.py
```

---

# Machine Learning Pipeline

The script implements a complete ML workflow:

```
Step 1 — Load and preview the dataset
Step 2 — Exploratory Data Analysis (statistics, class balance)
Step 3 — Preprocessing: Train-test split and feature scaling
Step 4 — Model training and coefficient analysis
Step 5 — Performance evaluation (accuracy, precision, recall)
Step 6 — Visualization of decision boundary and confusion matrix
Step 7 — Interactive prediction on new data
```

---

# Model Performance

## Evaluation Metrics
- **Overall Accuracy**: 89%
- **Precision (No-Buy)**: 89%
- **Recall (No-Buy)**: 96%
- **Precision (Buy)**: 89%
- **Recall (Buy)**: 75%

## Learned Coefficients
- **Age Weight**: 2.0767 (dominant factor)
- **Salary Weight**: 1.1101 (secondary factor)
- **Intercept**: -0.9522

### Key Insights
- Age has ~87% more influence than salary on purchase decisions
- Model performs better at identifying non-buyers (96% recall) than buyers (75% recall)
- Linear decision boundary effectively separates the two classes

# Visual Results

After running the script, two visualizations are generated:

1. **`confusion_matrix.png`**: Heatmap showing model performance with True Positives, False Positives, True Negatives, and False Negatives
2. **`decision_boundary.png`**: 2D scatter plot with red (No Buy) and green (Buy) regions separated by the decision boundary line

---

# Mathematical Background

## Logistic Regression Formula
The model uses the sigmoid function to convert linear combinations into probabilities:

```
z = w₁·Age + w₂·Salary + b
P(Purchase) = 1 / (1 + e^(-z))
```

## Decision Boundary
The classification threshold (P = 0.5) creates the decision boundary:

```
2.0767·Age + 1.1101·Salary - 0.9522 = 0
```

## Feature Scaling
Standard scaling ensures fair feature comparison:

```
z = (x - μ) / σ
```

Where μ is mean and σ is standard deviation.

# Key Concepts

## Standard Scaling
Brings features to comparable ranges (typically -3 to +3) to prevent features with larger scales from dominating the model.

## Sigmoid Function
"Squashes" any real number into a probability between 0 and 1, making it suitable for binary classification.

## Maximum Likelihood Estimation
The training process finds weights that maximize the likelihood of observing the actual training data. 

---

# Business Insights

## Customer Segmentation
- **Primary Target**: Ages 35-50 with income >$60k
- **Secondary Target**: Ages 30-35 or 50+ with income >$80k
- **Low Priority**: Ages 18-30 regardless of income

## Marketing Recommendations
- Focus 70% of budget on middle-aged, middle-to-high income users
- Age is the dominant purchase predictor, more important than salary
- Consider adjusting classification threshold to capture more potential buyers

## Model Limitations
- Linear decision boundary may miss complex patterns
- Only considers two demographic features
- May require additional behavioral data for improved accuracy

# Troubleshooting

**1. ModuleNotFoundError**
Ensure virtual environment is activated before installing packages:
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**2. Plot window not closing**
The script pauses to display visualizations. Close plot windows to continue to interactive prediction.

**3. Dependencies not installing**
Update pip and try again:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# Further Learning

For detailed mathematical explanations and extended theory, see [THEORY.md](THEORY.md).

## References
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning.

---

**Thanks for exploring this logistic regression demo!** 🚀
