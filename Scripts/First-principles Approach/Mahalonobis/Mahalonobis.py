#This script is from the ChemRxiv pre-print "Machine learning generalised DFT+U projectors in a numerical atom-centred orbital framework" at the DOI 10.26434/chemrxiv-2025-332z0 
#This script is used in Section 3.3.2, to compute the average Mahalonobis distance per material, i.e., the average similarity between the primary feature vectors for each material, to see whether this is related to the dataset size in Figure 13.

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

# Load data
data = pd.read_csv("Data.txt", sep="\t")  # Assuming tab-separated values

# Extract relevant features (excluding 'Material')
features = data.iloc[:, 1:]  # All columns except 'Material'

# Perform PCA to reduce to 2D
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Compute covariance matrix and its inverse
cov_matrix = np.cov(reduced_features, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Compute Mahalanobis distance for each row
mahalanobis_distances = [mahalanobis(point, np.mean(reduced_features, axis=0), inv_cov_matrix) for point in reduced_features]

# Add distances to DataFrame
data["Mahalanobis_Distance"] = mahalanobis_distances

# Compute average Mahalanobis distance per material
avg_distances = data.groupby("Material")["Mahalanobis_Distance"].mean().reset_index()

# Save to text file
avg_distances.to_csv("Mahalanobis_Averaged_Distances.txt", sep="\t", index=False)

print("Averaged Mahalanobis distances saved to Mahalanobis_Averaged_Distances.txt")
