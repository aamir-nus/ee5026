import traceback
import numpy as np

def compute_lda(X: np.ndarray, y: np.ndarray, p: int):
    """
    Computes LDA on the data matrix X from scratch using scatter matrices.

    Args:
        X: The data matrix of shape (n_samples, n_features)
        y: The class labels of shape (n_samples,)
        p: The number of LDA projections to retain

    Returns:
        Tuple of (lda_vectors, transformed_data):
            - lda_vectors: The LDA projection vectors of shape (n_features, p)
            - transformed_data: The LDA transformed data of shape (n_samples, p)
    """
    n_samples, n_features = X.shape
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    #limit p to min(n_classes - 1, n_features)
    p = min(p, n_classes - 1, n_features)

    #compute overall mean and initialize scatter matrices
    overall_mean = np.mean(X, axis=0)
    S_w = np.zeros((n_features, n_features))  #within-class scatter
    S_b = np.zeros((n_features, n_features))  #between-class scatter

    #compute scatter matrices
    for class_label in unique_classes:
        X_class = X[y == class_label]
        n_class = X_class.shape[0]
        class_mean = np.mean(X_class, axis=0)

        #update within-class scatter
        class_scatter = np.cov((X_class - class_mean).T, bias=True) * n_class
        S_w += class_scatter

        #update between-class scatter
        mean_diff = (class_mean - overall_mean).reshape(n_features, 1)
        S_b += n_class * (mean_diff @ mean_diff.T)

    #solve the generalized eigenvalue problem: S_b * w = λ * S_w * w

    #add small regularization for numerical stability
    #compute S_w^(-1) * S_b and solve eigenvalue problem
    #handle potential singularity of S_w
    try:
        reg_factor = 1e-6
        S_w_reg = S_w + reg_factor * np.eye(n_features)

        #compute S_w^(-1) * S_b and solve eigenvalue problem
        S_w_inv_S_b = np.linalg.inv(S_w_reg) @ S_b
        eigenvalues, eigenvectors = np.linalg.eig(S_w_inv_S_b)

        #take real parts and sort in descending order
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        idx = np.argsort(eigenvalues)[::-1]

        lda_vectors = eigenvectors[:, idx[:p]]
        transformed_data = X @ lda_vectors

    #fallback to PCA if S_w is singular
    except np.linalg.LinAlgError:
        print("Warning: S_w matrix is singular, falling back to PCA-like sol ")
        _, _, Vt = np.linalg.svd(S_b)
        lda_vectors = Vt.T[:, :p]
        transformed_data = X @ lda_vectors

    except Exception as e:
        print(f"Error computing LDA : {e}")
        print(f"Traceback : {traceback.format_exc(limit=3)}")
        lda_vectors = np.zeros((n_features, p))
        transformed_data = np.zeros((n_samples, p))

    return lda_vectors, transformed_data
