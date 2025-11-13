import numpy as np

def compute_pca(X, p):
    """
    Computes PCA on the data matrix X from scratch using SVD.

    Args:
        X (np.ndarray): The data matrix of shape (n_samples, n_features).
        p (int): The number of principal components to retain.

    Returns:
        tuple: A tuple containing:
            - pca_vectors (np.ndarray): The PCA vectors (eigenvectors) of shape (n_features, p).
            - transformed_data (np.ndarray): The PCA transformed data of shape (n_samples, p).
    """
    # Center the data by subtracting the mean
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute SVD of the centered data matrix
    # U: left singular vectors (n_samples x n_samples)
    # S: singular values (min(n_samples, n_features))
    # Vt: right singular vectors transpose (n_features x n_features)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # The principal components are the right singular vectors (columns of V)
    # We need the first p components, end shape will be (n_features, p)
    pca_vectors = Vt.T[:, :p]

    # Transform the data using the first p principal components. end shape will be (n_samples, p)
    transformed_data = X_centered @ pca_vectors

    return pca_vectors, transformed_data
