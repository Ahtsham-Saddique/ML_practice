import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Step 1: Load Iris dataset
iris = load_iris()

# Step 2: Convert to Pandas DataFrame with correct column names
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Step 3: Select correct features
X = data[['petal length (cm)', 'petal width (cm)']]

# Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Step 5: Get cluster labels
labels = kmeans.labels_

# Step 6: Plot result
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=labels)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clustering on Iris Dataset")
plt.show()
