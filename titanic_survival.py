# ============================================================
#   TITANIC SURVIVAL PREDICTION
#   Model: Decision Tree
#   Dataset: Titanic Dataset (Kaggle)
# ============================================================

import pandas as pd 
import numpy as np 
from  sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,export_text
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# ── 1. LOAD DATASET ─────────────────────────────────────────
df = pd.read_csv("project_4/Titanic-Dataset.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 row: ")
print(df.head())
print("\nMissing value: ")
print(df.isnull().sum())

# ── 2. DATA CLEANING ────────────────────────────────────────
# Age mein missing values hain — median se fill karnge
df["Age"] = df["Age"].fillna(df["Age"].median())

# Cabin mein bahut missing values hain — drop karnge
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Embarked mein 2 missing values — mode se fill karo
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nAfter cleaning — missing values:")
print(df.isnull().sum())

# ── 3. ENCODE CATEGORICAL COLUMNS ───────────────────────────
# Sex: male → 0, female → 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
 
# Embarked: S → 0, C → 1, Q → 2
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# ── 4. SPLIT FEATURES AND LABEL ─────────────────────────────
X = df.drop("Survived", axis=1)
y = df["Survived"]
 
print(f"\nFeatures used: {X.columns.tolist()}")

# ── 5. TRAIN-TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")
 
# ── 6. TRAIN MODEL ──────────────────────────────────────────
# max_depth = kitni deep tree banegi
# zyada depth = overfitting, kam depth = underfitting
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
 
print("\nModel training complete!")

# ── 7. EVALUATE ─────────────────────────────────────────────
y_pred = model.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("  [TN  FP]")
print("  [FN  TP]")
 
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))
 
# ── 8. FEATURE IMPORTANCE ───────────────────────────────────
print("\nFeature Importance:")
for name, importance in sorted(zip(X.columns, model.feature_importances_),
                                key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {importance:.4f}")