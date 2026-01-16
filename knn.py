import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Iris dataset
iris = load_iris()

# Step 2: Convert to DataFrame with correct column names
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Step 3: Select correct features and target
X = data[['sepal length (cm)', 'sepal width (cm)', 
          'petal length (cm)', 'petal width (cm)']]
y = data['target']

# Step 4: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# Step 5: Apply KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 6: Predict
y_pred = knn.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))

print("Predicted:", y_pred)
print("Actual:", y_test.values)

