import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

try:
    df = pd.read_csv('Housing.csv')
except FileNotFoundError:
    print("Error: The file 'Housing.csv' was not found.")
    print("Please make sure the file is correctly uploaded.")
    exit()

print("--- Initial Dataset ---")
print(df.head())
print("\n" + "="*30 + "\n")

features = ['area', 'bedrooms', 'bathrooms']
target = 'price'

if not all(feature in df.columns for feature in features):
    print("Error: The required columns ('area', 'bedrooms', 'bathrooms') are not in the CSV file.")
    exit()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Data Splitting ---")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("\n" + "="*30 + "\n")

model = LinearRegression()
model.fit(X_train, y_train)

print("--- Model Training Complete ---")
for feature, coef in zip(features, model.coef_):
    print(f"Model coefficient for '{feature}': {coef:,.2f}")
print(f"Model intercept: {model.intercept_:,.2f}")
print("\n" + "="*30 + "\n")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (R2) score: {r2:.4f}")
print("\n" + "="*30 + "\n")

new_house_features = {
    'area': [6000],
    'bedrooms': [3],
    'bathrooms': [2]
}
new_house_df = pd.DataFrame(new_house_features)
predicted_price = model.predict(new_house_df)

print("--- New House Price Prediction ---")
print(f"Features of the new house: {new_house_features}")
print(f"Predicted price: ${predicted_price[0]:,.2f}")
