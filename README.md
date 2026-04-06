# Logistic Regression (SUV Purchase Prediction)

This demo uses Logistic Regression to predict whether someone will buy an SUV based on their age and salary.

---

# Files

You should have downloaded these 4 files:

```
model-lr.py          ← the script
social_ads.csv       ← the dataset
requirements.txt     ← list of packages needed
README.md            ← this file
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

# What the Script Does

```
Step 1 — Load the CSV and preview the raw dataset
Step 2 — Exploratory Data Analysis (EDA) checking Class Balance and Statistics
Step 3 — Preprocessing: Splitting (75/25) and Scaling (Standardization)
Step 4 — Train the Logistic Regression model and display Weights (Coefficients)
Step 5 — Evaluate performance: Accuracy, Precision, Recall, and Heatmap
Step 6 — Generate the Decision Boundary Plot and Confusion Matrix
Step 7 — Prediction on unknown data
```

---

# Visual Results

After running the script, two image files will be generated in your folder:
1.  **`confusion_matrix.png`**: A heatmap showing exactly where the model predicted correctly and where it made mistakes.
2.  **`decision_boundary.png`**: A 2D plot showing the **Red** (No Buy) and **Green** (Buy) zones with a clear line separating them.

---

# Key Concepts

## Standard Scaling (The "Fairness" Step)
The formula used is **z = (x - mean) / std**. Salary is in thousands ($150k) and Age is in tens (60). Without scaling, the model would only care about Salary. Scaling brings both numbers into a similar range (usually -3 to +3).

## Logistic Regression (The "Squashing" Function)
We pass the **weighted sum (z)** into a **Sigmoid Function**. This "squashes" the result into a probability percentage between 0 and 1. 

---

# Troubleshooting

**1. "ModuleNotFoundError"**
This happens if you haven't activated your Virtual Environment. Make sure you run `source venv/bin/activate` before running the script.

**2. "Interactive demo not starting"**
The script pauses at Step 6 to show the plot window. You **must close the plot window** for Step 7 (the interactive part) to begin!

**Thanks for training and predicting. Goodbye!**
