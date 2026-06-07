import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# LOAD DATA
df = pd.read_csv("data/churn.csv")

# CLEAN DATA
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# CONVERT TARGET
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# ENCODE CATEGORICAL COLUMNS
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# FEATURES & TARGET
X = df.drop("Churn", axis=1)
y = df["Churn"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = RandomForestClassifier()

model.fit(X_train, y_train)

# PREDICTION
predictions = model.predict(X_test)

# ACCURACY
score = accuracy_score(y_test, predictions)

print("Accuracy:", score)

# SAVE MODEL
joblib.dump(
    model,
    "models/churn_model.pkl"
)

print("Model Saved Successfully")
