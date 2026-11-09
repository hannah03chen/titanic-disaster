from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def prepare(df, age_median, fare_median, embarked_mode):
    """Turn raw Titanic columns into numeric model features."""
    out = pd.DataFrame(index=df.index)
    out["Pclass"] = df["Pclass"]
    out["Sex"] = (df["Sex"] == "female").astype(int)
    out["Age"] = df["Age"].fillna(age_median)
    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["Fare"] = df["Fare"].fillna(fare_median)
    embarked = df["Embarked"].fillna(embarked_mode)
    for port in ["C", "Q", "S"]:
        out[f"Embarked_{port}"] = (embarked == port).astype(int)
    return out


print("Container is running!")

# Step 13: load the training data
train = pd.read_csv(DATA_DIR / "train.csv")
print(f"Loaded train.csv: {train.shape[0]} rows, {train.shape[1]} columns")

# Step 14: explore the data
print("\nFirst 5 rows:")
print(train.head())
print("\nMissing values per column:")
print(train.isna().sum()[train.isna().sum() > 0])
print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
print("\nSurvival rate by sex:")
print(train.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by passenger class:")
print(train.groupby("Pclass")["Survived"].mean())

# Step 14: adjust the data
age_median = train["Age"].median()
fare_median = train["Fare"].median()
embarked_mode = train["Embarked"].mode()[0]
print(f"\nFilling missing Age with median ({age_median}), "
      f"Fare with median ({fare_median:.2f}), Embarked with mode ({embarked_mode})")
print("Encoding Sex as 0/1, one-hot encoding Embarked, adding FamilySize = SibSp + Parch + 1")

X = prepare(train, age_median, fare_median, embarked_mode)
y = train["Survived"]
print(f"Features used: {list(X.columns)}")
print(f"Missing values after cleaning: {int(X.isna().sum().sum())}")

# Step 15: build the logistic regression model
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nSplit train.csv into {len(X_train)} training rows and {len(X_val)} validation rows")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Trained logistic regression model")
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name}: {coef:.3f}")

# Step 16: accuracy on the training set
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"\nTraining accuracy: {train_acc:.3f}")

# Step 17: load test.csv and predict
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"\nLoaded test.csv: {test.shape[0]} rows, {test.shape[1]} columns")
X_test = prepare(test, age_median, fare_median, embarked_mode)
test_pred = model.predict(X_test)
print(f"Predicted {int(test_pred.sum())} survivors out of {len(test_pred)} test passengers")

# Step 18: accuracy on data the model has not seen
# test.csv has no Survived column, so its true accuracy can't be computed.
# The held-out 20% of train.csv serves as the test-accuracy measure.
val_acc = accuracy_score(y_val, model.predict(X_val))
print(f"Held-out validation accuracy: {val_acc:.3f}")# Titanic survival model
