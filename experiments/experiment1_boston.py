import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("datasets/boston.csv")
print("First records:")
print(data.head())
print("Dataset shape:", data.shape)

X = data[["RM", "PTRATIO", "LSTAT"]]
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual values:", y_test.to_list())
print("Predicted values:", predictions.round(2))
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("MSE:", round(mean_squared_error(y_test, predictions), 2))
print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
print("R2 score:", round(r2_score(y_test, predictions), 4))

new_house = pd.DataFrame({
    "RM": [6.8], "PTRATIO": [16], "LSTAT": [11]
})
print("Predicted MEDV for new house:", round(model.predict(new_house)[0], 2))
