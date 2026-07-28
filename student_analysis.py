# Student Performance Analysis & Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Load Dataset
df = pd.read_csv("student_performance.csv")


# Display Data
print("First 5 Records")
print(df.head())


# Dataset Information
print("\nDataset Information")
print(df.info())


# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())


# Feature Engineering

df["Total_Marks"] = (
    df["Math"] +
    df["Science"] +
    df["English"]
)

df["Average_Marks"] = df["Total_Marks"] / 3


# Performance Category

def performance(avg):
    if avg >= 90:
        return "Excellent"
    elif avg >= 75:
        return "Good"
    elif avg >= 50:
        return "Average"
    else:
        return "Poor"


df["Performance"] = df["Average_Marks"].apply(performance)


print("\nStudent Performance")
print(df)


# Top 5 Students

print("\nTop 5 Students")

print(
    df.sort_values(
        by="Total_Marks",
        ascending=False
    ).head()
)


# Visualization 1: Average Subject Marks

plt.figure(figsize=(7,4))

sns.barplot(
    x=["Math","Science","English"],
    y=[
        df["Math"].mean(),
        df["Science"].mean(),
        df["English"].mean()
    ]
)

plt.title("Average Subject Marks")
plt.ylabel("Marks")
plt.show()


# Visualization 2: Performance Distribution

plt.figure(figsize=(6,4))

sns.countplot(
    x="Performance",
    data=df
)

plt.title("Student Performance Distribution")
plt.show()


# Visualization 3: Correlation Heatmap

plt.figure(figsize=(8,5))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)

plt.title("Correlation Analysis")
plt.show()


# Machine Learning Model

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Math",
        "Science",
        "English"
    ]
]

y = df["Total_Marks"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


prediction = model.predict(X_test)


print("\nModel Performance")

print("MAE:", mean_absolute_error(y_test, prediction))

print("R2 Score:", r2_score(y_test, prediction))


# Predict New Student

new_student = [[6,95,90,92,91]]

result = model.predict(new_student)

print(
    "\nPredicted Total Marks:",
    result[0]
)
