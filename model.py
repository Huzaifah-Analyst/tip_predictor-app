import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the cleaned dataset
data = pd.read_csv('final_cleaned_tips.csv')

# Define independent variables (X) and dependent variable (y)
X = data[['total_bill', 'size']]  # Independent variables: total_bill and size
y = data['tip']                   # Dependent variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the model coefficients
print("Intercept (a):", model.intercept_)
print("Coefficients (b):", model.coef_)
print("  - Coefficient for total_bill:", model.coef_[0])
print("  - Coefficient for size:", model.coef_[1])

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)

# Show a few predictions vs actual values
print("\nSample Predictions vs Actual Values:")
for i in range(5):
    print(f"Total Bill: {X_test['total_bill'].iloc[i]}, Size: {X_test['size'].iloc[i]}, Actual Tip: {y_test.iloc[i]}, Predicted Tip: {y_pred[i]:.2f}")

# Save the model to a file
joblib.dump(model, 'tip_prediction_model.pkl')
print("\nModel saved to 'tip_prediction_model.pkl'")