import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

data = pd.read_csv("datasets/wholesale.csv")
print("Customer dataset:")
print(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# For scikit-learn versions that support metric="euclidean"
model = AgglomerativeClustering(
    n_clusters=3, linkage="ward", metric="euclidean"
)
clusters = model.fit_predict(scaled_data)
data["Cluster"] = clusters + 1

print("\nCustomer cluster assignments:")
print(data)

linkage_matrix = linkage(scaled_data, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer index")
plt.ylabel("Ward distance")
plt.tight_layout()
plt.show()
