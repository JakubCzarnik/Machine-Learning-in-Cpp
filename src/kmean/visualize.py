import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("E:/Vs Code Scripts/cpp/Machine-Learning-in-Cpp/datasets/Iris dataset/IRIS.csv")

df.drop(columns=["petal_length", "petal_width", "species"], inplace=True)


for col in df.columns:
   df[col] = (df[col] - df[col].mean()) / df[col].std()

centroids = [[-0.14944, -0.975363],
             [-0.946528,  0.97577],
             [1.0207, 0.0678782]]
centroids = np.array(centroids)


def assign_points_to_clusters(df, centroids):
   distances = np.sqrt(((df - centroids[:, np.newaxis])**2).sum(axis=2))
   cluster_labels = np.argmin(distances, axis=0)
   return cluster_labels


def compute_centroids(df, labels):
   return np.array([df[labels==k].mean(axis=0) for k in range(centroids.shape[0])])


labels = assign_points_to_clusters(df.values, np.array(centroids))
new_centroids = compute_centroids(df.values, labels)


plt.figure(figsize=(10, 6))
for i, color in enumerate(['blue', 'green', 'red']):
   plt.scatter(df.values[labels == i, 0], df.values[labels == i, 1], color=color, label=f'Cluster {i+1}')
plt.scatter(new_centroids[:, 0], new_centroids[:, 1], color='black', label='Centroids')
plt.title('Clusters')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend()
plt.show()
