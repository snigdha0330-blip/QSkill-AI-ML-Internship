# Import libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target column
df['Price'] = housing.target

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Predict a sample house
sample_house = [[8.3, 41, 6.9, 1.0, 850, 2.5, 37.8, -122.4]]

predicted_price = model.predict(sample_house)

print("\nPredicted House Price:")
print(predicted_price[0])
