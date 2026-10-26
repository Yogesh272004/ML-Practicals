import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("datasets/adult.csv")
X = data.drop(columns=["Income"])
y = data["Income"]

numeric_columns = ["Age", "HoursPerWeek", "CapitalGain"]
categorical_columns = ["Education", "Occupation"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

model = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual income:", y_test.to_list())
print("Predicted income:", predictions.tolist())
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

new_person = pd.DataFrame([{
    "Age": 40, "Education": "Masters", "HoursPerWeek": 50,
    "CapitalGain": 5000, "Occupation": "Professional"
}])
print("Predicted income:", model.predict(new_person)[0])
