import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# Load Iris dataset from sklearn
iris = load_iris()

# Convert to DataFrame with correct column names
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Check column names (optional but safe)
print("Columns:", data.columns)

# Select feature (X) and target (y)
X = data[['sepal length (cm)']]   # Feature
y = data['petal length (cm)']     # Target

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot graph
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Linear Regression - Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
