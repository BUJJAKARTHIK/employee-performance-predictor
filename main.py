import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# STEP 1: CREATE SYNTHETIC DATA
# -----------------------------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'Age': np.random.randint(22, 60, n),
    'Experience': np.random.randint(1, 35, n),
    'Salary': np.random.randint(20000, 150000, n),
    'Training_Hours': np.random.randint(5, 100, n),
    'Department': np.random.choice(['HR', 'IT', 'Sales', 'Finance'], n)
})

# Performance Score
data['Performance'] = (
    data['Experience'] * 0.3 +
    data['Training_Hours'] * 0.2 +
    (data['Salary'] / 10000) * 0.1 +
    np.random.randn(n)
)

# Convert to labels
data['Performance_Label'] = pd.cut(
    data['Performance'],
    bins=[data['Performance'].min()-1, 5, 10, data['Performance'].max()+1],
    labels=['Low', 'Medium', 'High']
)

# Remove NaN
data = data.dropna()

print("Missing values:\n", data.isnull().sum())

# Save dataset
data.to_csv("data/employee_data.csv", index=False)

print("Dataset created and saved!")

# -----------------------------
# EDA (Exploratory Data Analysis)
# -----------------------------

# Age Distribution
plt.figure()
sns.histplot(data['Age'], kde=True)
plt.title("Age Distribution")
plt.savefig("outputs/age_distribution.png")

# Salary vs Performance
plt.figure()
sns.scatterplot(x=data['Salary'], y=data['Performance'])
plt.title("Salary vs Performance")
plt.savefig("outputs/salary_vs_performance.png")

# Department Distribution
plt.figure()
sns.countplot(x=data['Department'])
plt.title("Department Distribution")
plt.savefig("outputs/department_distribution.png")
# -----------------------------
# PREPROCESSING
# -----------------------------
data = pd.get_dummies(data, columns=['Department'], drop_first=True)

X = data.drop(['Performance', 'Performance_Label'], axis=1)
y = data['Performance_Label']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

print(classification_report(y_test, y_pred))