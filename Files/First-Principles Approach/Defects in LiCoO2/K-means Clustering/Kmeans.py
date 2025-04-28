import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import numpy as np
import os
import string

# Load the data
df = pd.read_csv("J_FP_landscape.txt", sep="\t")

# Filter the bottom 10% of JFP values
threshold = df["JFP"].quantile(0.10)
df_low_jfp = df[df["JFP"] <= threshold].copy()

print(f"Selected {len(df_low_jfp)} rows with JFP <= {threshold:.4f}")

# Extract features for clustering
features = df_low_jfp[["U", "c1", "c2"]]

# Apply KMeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_low_jfp["Cluster"] = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_

# Label clusters as A–F
cluster_labels = list(string.ascii_uppercase[:n_clusters])  # ['A', 'B', 'C', 'D', 'E', 'F']
cluster_map = {i: cluster_labels[i] for i in range(n_clusters)}
df_low_jfp["ClusterLabel"] = df_low_jfp["Cluster"].map(cluster_map)

# Create output directory
output_dir = "K-means Clustering/cluster_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save U, c1, c2 values for each cluster to separate text files
for i in range(n_clusters):
    label = cluster_labels[i]
    cluster_data = df_low_jfp[df_low_jfp["ClusterLabel"] == label][["U", "c1", "c2"]]
    filename = os.path.join(output_dir, f"Cluster_{label}.txt")
    cluster_data.to_csv(filename, index=False, sep="\t")
    print(f"Saved Cluster {label} data to {filename}")

# Save cluster centroids to a text file
centroid_df = pd.DataFrame(centroids, columns=["U", "c1", "c2"])
centroid_df.index = cluster_labels
centroid_df.index.name = "Cluster"
centroid_path = os.path.join(output_dir, "centroids.txt")
centroid_df.to_csv(centroid_path, sep="\t")
print(f"Saved centroids to {centroid_path}")

# Plot only the centroids
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each centroid as a large marker with a label A–F
for i, (x, y, z) in enumerate(centroids):
    label = cluster_labels[i]
    ax.scatter(x, y, z, s=120, label=f"Cluster {label}", alpha=0.9,
               edgecolors='black', marker='o')

# Axis labels with line breaks and padding
ax.set_xlabel('Co 3$d$ Hubbard\nU Value (eV)', fontsize=16, labelpad=25)
ax.set_ylabel('Co 3$d$ Hubbard\nProjector $c_1$', fontsize=16, labelpad=25)
ax.set_zlabel('Co 3$d$ Hubbard\nProjector $c_2$', fontsize=16, labelpad=25)
ax.tick_params(labelsize=16)

# Legend centered above the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=1, fancybox=True, shadow=False, fontsize=16)

plt.tight_layout()
plt.show()
