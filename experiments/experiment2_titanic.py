import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("datasets/titanic.csv")
X = data.drop(columns=["Survived"])
y = data["Survived"]

numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_columns = ["Sex"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual survival:", y_test.to_list())
print("Predicted survival:", predictions.tolist())
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print(classification_report(y_test, predictions, zero_division=0))

new_passenger = pd.DataFrame([{
    "Pclass": 3, "Sex": "female", "Age": 25,
    "SibSp": 0, "Parch": 0, "Fare": 8.0
}])
print("New passenger prediction (0=no, 1=yes):", model.predict(new_passenger)[0])
