import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


print(" SOCIAL NETWORK ADS — LOGISTIC REGRESSION ")

# ── STEP 1 : LOAD DATA ─────────────────────────────────────────
df = pd.read_csv("social_ads.csv")

print("\n [1] DataSet Preview:")
print(" ")
print(df.head())

# We only use Age (Column 2) and Estimated Salary (Column 3) to predict Purchase (Column 4)
X = df.iloc[:, [2, 3]].values
y = df.iloc[:, -1].values

print(" ")
print(f"    ✔ Loaded {df.shape[0]} users.")
print(f"    ✔ Features used: Age, Estimated Salary.")

# ── STEP 2: DATA INSIGHTS (EDA) ─────────────────────────────
print("\n[2] Exploratory Data Analysis (EDA)...")
print("")

# Check for missing values (Rigorous check like the other team)
missing = df.isnull().sum().sum()
print(f"✔ Missing Values: {missing}")

# Class Balance: How many Buy (1) vs Don't Buy (0)
balance = df.iloc[:, -1].value_counts()
print(f"✔ Class Balance: {balance[0]} No-Buy, {balance[1]} Buy")

# Basic Statistics summary
print("\nStatistical Summary of Features:")
print(" ")
print(df[['Age', 'EstimatedSalary']].describe().round(2))

# ── STEP 3 : PREPROCESSING ─────────────────────────────────────
print("\n[3] Preprocessing data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ── Scaling with example listed ────────────────────

sample_raw = X_test[0].copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sample_scaled = X_test[0]

print("\nScaling Being Showcased and Implemented:")
print(f"Raw Values   : Age = {sample_raw[0]:.0f}, Salary = {sample_raw[1]:.0f}")
print(f"Scaled Values: Age = {sample_scaled[0]:.4f}, Salary = {sample_scaled[1]:.4f}")

# ── Formula Used For Scaling ─────────────────
print("\nScaling Formula (How Z-score is made):")
print("    z = (X - Mean) / Std_Dev")
print(f"    - Age Mean: {scaler.mean_[0]:.2f}, Std Dev: {scaler.scale_[0]:.2f}")
print(f"    - Salary Mean: {scaler.mean_[1]:.2f}, Std Dev: {scaler.scale_[1]:.2f}")
print("\n    (Every number is converted using these values so '0' is the average)")

# ── STEP 4 : TRAIN MODEL ───────────────────────────────────────
print("\n[4] Training Logistic Regression model...")
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print("Model trained Successfully!")

# Interpretation (Good for students to see the math)
print(f"\n    Model Learned Coefficients:")
print(f"    - Weight for Age (w1)   : {classifier.coef_[0][0]:.4f}")
print(f"    - Weight for Salary (w2): {classifier.coef_[0][1]:.4f}")
print(f"    - Intercept (b)         : {classifier.intercept_[0]:.4f}")
print("    (Higher weights mean that feature has more 'voting power' in the prediction)")



# ── STEP 5 : EVALUATE ──────────────────────────────────────────
print("\n[5] Evaluating...")
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n    ➤ Final Accuracy : {acc * 100:.2f}%")

print("\n    Detailed Metrics (The 'Why' behind Accuracy):")
print(classification_report(y_test, y_pred, target_names=['Did not Buy', 'Bought']))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Purchase', 'Purchase'],
            yticklabels=['Not Purchase', 'Purchase'])
plt.title('Confusion Matrix', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
print("    ✔ Saved confusion matrix to 'confusion_matrix.png'")

# ── STEP 6 : VISUALISATION ─────────────────────────────────────
print("\n[6] Generating 2D Decision Boundary Plot...")
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare the meshgrid for the background colors (Red/Green regions)
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Predict exactly where the boundary line falls based on X1 (Age) and X2 (Salary)
ax.contourf(
    X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.3, cmap=ListedColormap(('#ff9999', '#99ff99'))
)

ax.set_xlim(X1.min(), X1.max())
ax.set_ylim(X2.min(), X2.max())

# Plot the actual data points
colors = ['red', 'green']
labels = ['0 (Did not buy)', '1 (Bought)']
for i, j in enumerate(np.unique(y_set)):
    ax.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        c=colors[i], label=labels[i], 
        edgecolor='black', s=60, alpha=0.9
    )

ax.set_title('SUV Purchase Prediction - Logistic Regression (Test set)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Age (Standardized)', fontsize=12, fontweight='bold')
ax.set_ylabel('Estimated Salary (Standardized)', fontsize=12, fontweight='bold')
ax.legend(title='Outcome', loc='best', fancybox=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
print("    ✔ Saved plot to 'decision_boundary.png'")
print("\nClose the plot window to check on unknown data")
plt.show()



# ── STEP 7 : Prediction on unknown data ────────────────────────────────
print("\n" + "=" * 60)
print(" Predict whether the new user will buy the car or not")
print("=" * 60)

while True:
    try:
        print("\n(Enter 'q' to quit)")
        age_input = input("Enter Age: ")
        if age_input.lower() == 'q': break
        
        salary_input = input("Enter Estimated Salary: ")
        if salary_input.lower() == 'q': break
        
        # Scale inputs before prediction
        new_data = scaler.transform([[float(age_input), float(salary_input)]])
        prediction = classifier.predict(new_data)
        probability = classifier.predict_proba(new_data)[0][1]
        
        result = "PURCHASE" if prediction[0] == 1 else "NO PURCHASE"
        
        print("-" * 30)
        print(f"Prediction: {result}")
        print(f"Probability of Buying: {probability * 100:.2f}%")
        print("-" * 30)
        
    except ValueError:
        print("!! Please enter valid numbers for Age and Salary.")
    except KeyboardInterrupt:
        break

print("\nThanks for training and predicting. Goodbye!\n")
