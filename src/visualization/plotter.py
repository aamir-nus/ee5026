import numpy as np
import matplotlib.pyplot as plt

def setup_plot_style():
    """
    Setup consistent plot styling for all visualizations
    """
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_pca_2d(transformed_data, labels, selfie_indices, save_path=None, show=True):
    """
    Plot 2D PCA visualization with different colors for each class and highlight selfies.

    Args:
        transformed_data: PCA transformed data of shape (n_samples, 2)
        labels: Class labels
        selfie_indices: Indices of selfie samples to highlight
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    #create color map with 26 different colors
    #use tab20 and repeat colors for additional classes
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 6))
    colors = np.vstack([base_colors, additional_colors])

    #plot non-selfie points first
    non_selfie_mask = np.ones(len(transformed_data), dtype=bool)
    non_selfie_mask[selfie_indices] = False

    for class_label in np.unique(labels):
        class_mask = (labels == class_label) & non_selfie_mask
        if np.any(class_mask):
            ax.scatter(transformed_data[class_mask, 0],
                      transformed_data[class_mask, 1],
                      c=[colors[class_label]],
                      label=f'Class {class_label}',
                      alpha=0.7, s=50)

    #plot selfie points with special highlighting
    if len(selfie_indices) > 0:
        selfie_mask = np.zeros(len(transformed_data), dtype=bool)
        selfie_mask[selfie_indices] = True

        #plot with larger markers and star shape
        ax.scatter(transformed_data[selfie_mask, 0],
                  transformed_data[selfie_mask, 1],
                  c='red', marker='*', s=200,
                  edgecolors='black', linewidths=2,
                  label='Selfie Photos', zorder=10)

    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('2D PCA Visualization of Face Images')
    ax.grid(True, alpha=0.3)

    #add legend
    handles, labels_legend = ax.get_legend_handles_labels()
    if len(handles) > 15:
        #show only every 3rd class plus selfies
        display_indices = list(range(0, min(12, len(handles)-1), 3)) + [-1]
        ax.legend([handles[i] for i in display_indices],
                 [labels_legend[i] for i in display_indices])
    else:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_pca_3d(transformed_data,
                labels,
                selfie_indices,
                save_path=None,
                show=True):
    """
    Plot 3D PCA visualization with different colors for each class and highlight selfies.

    Args:
        transformed_data: PCA transformed data of shape (n_samples, 3)
        labels: Class labels
        selfie_indices: Indices of selfie samples to highlight
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    #create color map with 26 different colors
    #use tab20 and repeat colors for additional classes
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 6))
    colors = np.vstack([base_colors, additional_colors])

    #plot non-selfie points first
    non_selfie_mask = np.ones(len(transformed_data), dtype=bool)
    non_selfie_mask[selfie_indices] = False

    for class_label in np.unique(labels):
        class_mask = (labels == class_label) & non_selfie_mask
        if np.any(class_mask):
            ax.scatter(transformed_data[class_mask, 0],
                      transformed_data[class_mask, 1],
                      transformed_data[class_mask, 2],
                      c=[colors[class_label]],
                      label=f'Class {class_label}',
                      alpha=0.7, s=50)

    #plot selfie points with special highlighting
    if len(selfie_indices) > 0:
        selfie_mask = np.zeros(len(transformed_data), dtype=bool)
        selfie_mask[selfie_indices] = True

        ax.scatter(transformed_data[selfie_mask, 0],
                  transformed_data[selfie_mask, 1],
                  transformed_data[selfie_mask, 2],
                  c='red', marker='*', s=200,
                  edgecolors='black', linewidths=2,
                  label='Selfie Photos', zorder=10)

    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    ax.set_title('3D PCA Visualization of Face Images')

    #adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    #add legend (limit to avoid overcrowding)
    handles, labels_legend = ax.get_legend_handles_labels()
    if len(handles) > 15:
        display_indices = list(range(0, min(12, len(handles)-1), 3)) + [-1]
        ax.legend([handles[i] for i in display_indices],
                 [labels_legend[i] for i in display_indices])
    else:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_lda_2d(transformed_data, labels, selfie_indices, save_path=None, show=True):
    """
    Plot 2D LDA visualization with different colors for each class and highlight selfies.

    Args:
        transformed_data: LDA transformed data of shape (n_samples, 2)
        labels: Class labels
        selfie_indices: Indices of selfie samples to highlight
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    #create color map with 26 different colors
    #use tab20 and repeat colors for additional classes
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 6))
    colors = np.vstack([base_colors, additional_colors])

    #plot non-selfie points first
    non_selfie_mask = np.ones(len(transformed_data), dtype=bool)
    non_selfie_mask[selfie_indices] = False

    for class_label in np.unique(labels):
        class_mask = (labels == class_label) & non_selfie_mask
        if np.any(class_mask):
            ax.scatter(transformed_data[class_mask, 0],
                      transformed_data[class_mask, 1],
                      c=[colors[class_label]],
                      label=f'Class {class_label}',
                      alpha=0.7, s=50)

    #plot selfie points with special highlighting
    if len(selfie_indices) > 0:
        selfie_mask = np.zeros(len(transformed_data), dtype=bool)
        selfie_mask[selfie_indices] = True

        ax.scatter(transformed_data[selfie_mask, 0],
                  transformed_data[selfie_mask, 1],
                  c='red', marker='*', s=200,
                  edgecolors='black', linewidths=2,
                  label='Selfie Photos', zorder=10)

    ax.set_xlabel('First LDA Component')
    ax.set_ylabel('Second LDA Component')
    ax.set_title('2D LDA Visualization of Face Images')
    ax.grid(True, alpha=0.3)

    #add legend
    handles, labels_legend = ax.get_legend_handles_labels()
    if len(handles) > 15:
        display_indices = list(range(0, min(12, len(handles)-1), 3)) + [-1]
        ax.legend([handles[i] for i in display_indices],
                 [labels_legend[i] for i in display_indices])
    else:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_lda_3d(transformed_data, labels, selfie_indices, save_path=None, show=True):
    """
    Plot 3D LDA visualization with different colors for each class and highlight selfies.

    Args:
        transformed_data: LDA transformed data of shape (n_samples, 3)
        labels: Class labels
        selfie_indices: Indices of selfie samples to highlight
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    #create color map with 26 different colors
    #use tab20 and repeat colors for additional classes
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 6))
    colors = np.vstack([base_colors, additional_colors])

    #plot non-selfie points first
    non_selfie_mask = np.ones(len(transformed_data), dtype=bool)
    non_selfie_mask[selfie_indices] = False

    for class_label in np.unique(labels):
        class_mask = (labels == class_label) & non_selfie_mask
        if np.any(class_mask):
            ax.scatter(transformed_data[class_mask, 0],
                      transformed_data[class_mask, 1],
                      transformed_data[class_mask, 2],
                      c=[colors[class_label]],
                      label=f'Class {class_label}',
                      alpha=0.7, s=50)

    #plot selfie points with special highlighting
    if len(selfie_indices) > 0:
        selfie_mask = np.zeros(len(transformed_data), dtype=bool)
        selfie_mask[selfie_indices] = True

        ax.scatter(transformed_data[selfie_mask, 0],
                  transformed_data[selfie_mask, 1],
                  transformed_data[selfie_mask, 2],
                  c='red', marker='*', s=200,
                  edgecolors='black', linewidths=2,
                  label='Selfie Photos', zorder=10)

    ax.set_xlabel('First LDA Component')
    ax.set_ylabel('Second LDA Component')
    ax.set_zlabel('Third LDA Component')
    ax.set_title('3D LDA Visualization of Face Images')

    #adjust viewing angle
    ax.view_init(elev=20, azim=45)

    #add legend
    handles, labels_legend = ax.get_legend_handles_labels()
    if len(handles) > 15:
        display_indices = list(range(0, min(12, len(handles)-1), 3)) + [-1]
        ax.legend([handles[i] for i in display_indices],
                 [labels_legend[i] for i in display_indices])
    else:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_eigenfaces(pca_vectors,
                    n_components=2,
                    save_path=None,
                    show=True):
    """
    Visualize eigenfaces (principal components of face images).

    Args:
        pca_vectors: PCA vectors (eigenvectors) of shape (1024, n_components)
        n_components: Number of eigenfaces to display
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    n_components = min(n_components, pca_vectors.shape[1])
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))

    if n_components == 1:
        axes = [axes]

    for i in range(n_components):
        eigenface = pca_vectors[:, i].reshape(32, 32)

        axes[i].imshow(eigenface, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Eigenface {i+1}')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_fisherfaces(lda_vectors,
                     n_components=2,
                     save_path=None,
                     show=True):
    """
    Visualize fisherfaces (LDA projection vectors of face images).

    Args:
        lda_vectors: LDA vectors of shape (1024, n_components)
        n_components: Number of fisherfaces to display
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    n_components = min(n_components, lda_vectors.shape[1])
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))

    if n_components == 1:
        axes = [axes]

    for i in range(n_components):
        fisherface = lda_vectors[:, i].reshape(32, 32)

        axes[i].imshow(fisherface, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Fisherface {i+1}')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_gmm_clustering(gmm,
                        X_2d,
                        labels,
                        selfie_indices,
                        save_path=None,
                        show=True):
    """
    Plot GMM clustering results in 2D.

    Args:
        gmm: Trained GMM model
        X_2d: 2D data for visualization
        labels: True labels for coloring points
        selfie_indices: Indices of selfie samples to highlight
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get GMM predictions and probabilities
    cluster_labels = gmm.predict(X_2d)
    probabilities = gmm.predict_proba(X_2d)

    # Create mesh grid for decision boundaries
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    #plot decision boundaries
    Z = gmm.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.argmax(axis=1).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, levels=3, cmap='viridis')

    #plot non-selfie points
    non_selfie_mask = np.ones(len(X_2d), dtype=bool)
    non_selfie_mask[selfie_indices] = False

    scatter = ax.scatter(X_2d[non_selfie_mask, 0], X_2d[non_selfie_mask, 1],
                        c=labels[non_selfie_mask], cmap='tab26', alpha=0.7, s=50)

    #plot selfie points with special highlighting
    if len(selfie_indices) > 0:
        ax.scatter(X_2d[selfie_indices, 0], X_2d[selfie_indices, 1],
                  c='red', marker='*', s=200,
                  edgecolors='black', linewidths=2,
                  label='Selfie Photos', zorder=10)

    #plot GMM centers
    centers = gmm.means_
    ax.scatter(centers[:, 0], centers[:, 1],
              c='black', marker='x', s=200, linewidths=3,
              label='GMM Centers', zorder=10)

    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_title('GMM Clustering Results (3 Components)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()