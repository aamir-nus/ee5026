import os
import sys
import numpy as np
import traceback

from pathlib import Path
from typing import Tuple, List

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.pca import compute_pca
from src.preprocessing.lda import compute_lda
from src.data.data_loader import prepare_dataset, select_random_subjects, load_config
from src.visualization.plotter import (
    plot_pca_2d, plot_pca_3d, plot_lda_2d, plot_lda_3d,
    plot_eigenfaces, plot_fisherfaces
)

def create_output_directories() -> None:
    os.makedirs('Outputs/pca', exist_ok=True)
    os.makedirs('Outputs/lda', exist_ok=True)
    os.makedirs('Outputs/features', exist_ok=True)

def prepare_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int]]:
    """
    Prepare the dataset for Part 1: Face Recognition with PCA+LDA for Feature Extraction

    Steps:
        - Dataset preparation with pie and selfie photos
        - PCA for feature extraction and visualization
        - LDA for feature extraction and visualization

    Returns:
        (x_train, x_test, y_train, y_test, valid_subjects, train_selfie_indices, test_selfie_indices)
    """

    config = load_config()
    print(f"using seed value: {config['seed']}")

    cmupie_dir = "PIE"
    selfie_dir = "Selfies and Images"

    selected_subjects = select_random_subjects()
    print(f"Selected {len(selected_subjects)} subjects: {selected_subjects}")

    try:
        x_train, x_test, y_train, y_test, valid_subjects, train_selfie_indices, test_selfie_indices = prepare_dataset(
            cmupie_dir, selfie_dir
        )

        print(f"Dataset preparation completed:\ntraining samples : {x_train.shape[0]} | test samples : {x_test.shape[0]}")
        print(f"Image feature dim : {x_train.shape[1]} | num classes : {len(np.unique(y_train))}")
        print(f"Selfie samples in training : {len(train_selfie_indices)} | Selfie samples in test : {len(test_selfie_indices)}")

        return x_train, x_test, y_train, y_test, valid_subjects, train_selfie_indices, test_selfie_indices

    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        print(traceback.print_exc(limit=3))

    return tuple([None] * 7)

def run_pca_analysis(x_train, y_train, train_selfie_indices):
    """
    Run PCA analysis with different numbers of components.

    Returns:
        Dictionary containing PCA results for different component settings.
    """

    create_output_directories()
    print("Part I\n")
    print("preparing PCA with 2 components for visualization...")
    try:
        pca_vectors_2, x_train_pca_2 = compute_pca(x_train, 2)
        print(f"PCA vectors shape: {pca_vectors_2.shape} | transformed data shape : {x_train_pca_2.shape}")

        #visualize
        plot_pca_2d(x_train_pca_2, y_train, train_selfie_indices,
                   save_path="Outputs/pca/pca_2d_visualization.png", show=False)
        print("PCA image saved to Outputs/pca/pca_2d_visualization.png")

        #eigenfaces
        plot_eigenfaces(pca_vectors_2, n_components=2,
                       save_path='Outputs/pca/eigenfaces_2.png', show=False)
        print("Eigenfaces (2 componenets) saved to Outputs/pca/eigenfaces_2.png")

    except Exception as e:
        print(f"Error in running pca analysis : {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    #3 component pca for 3d visualization
    print("preparing PCA with 3 components...")
    try:
        pca_vectors_3, x_train_pca_3 = compute_pca(x_train, 3)
        print(f"PCA vectors shape: {pca_vectors_3.shape} | transformed data shape : {x_train_pca_3.shape}")

        #visualize
        plot_pca_3d(x_train_pca_3, y_train, train_selfie_indices,
                   save_path='Outputs/pca/pca_3d_visualization.png', show=False)
        print("3D PCA plot saved to Outputs/pca/pca_3d_visualization.png")

        #eigenfaces - 3d
        plot_eigenfaces(pca_vectors_3, n_components=3,
                       save_path='Outputs/pca/eigenfaces_3.png', show=False)
        print("Eigenfaces (3 components) saved to Outputs/pca/eigenfaces_3.png")

    except Exception as e:
        print(f"Error in running pca analysis with 3 components : {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    print("Part II\n\nPrepaing PCA with 100 components...")
    try:
        pca_vectors_100, x_train_pca_100 = compute_pca(x_train, 100)
        print(f"PCA vectors shape: {pca_vectors_100.shape} | transformed data shape : {x_train_pca_100.shape}")

        #save np arrays
        np.save('Outputs/features/pca_100_features.npy', x_train_pca_100)
        np.save('Outputs/features/pca_100_vectors.npy', pca_vectors_100)
        print("Features saved to Outputs/features/pca_100_features.npy | Vectors saved to Outputs/features/pca_100_vectors.npy")

    except Exception as e:
        print(f"Error in running pca analysis with 100 components : {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    #300 components
    try:
        pca_vectors_300, x_train_pca_300 = compute_pca(x_train, 300)
        print(f"PCA vectors shape: {pca_vectors_300.shape} | transformed data shape : {x_train_pca_300.shape}")

        #save np arrays
        np.save('Outputs/features/pca_300_features.npy', x_train_pca_300)
        np.save('Outputs/features/pca_300_vectors.npy', pca_vectors_300)
        print("Features saved to Outputs/features/pca_300_features.npy | Vectors saved to Outputs/features/pca_300_vectors.npy")

    except Exception as e:
        print(f"Error in PCA with 300 components: {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    return {
        'pca_vectors_2': pca_vectors_2 if 'pca_vectors_2' in locals() else None,
        'pca_vectors_3': pca_vectors_3 if 'pca_vectors_3' in locals() else None,
        'pca_vectors_100': pca_vectors_100 if 'pca_vectors_100' in locals() else None,
        'pca_vectors_300': pca_vectors_300 if 'pca_vectors_300' in locals() else None,
        'x_train_pca_2': x_train_pca_2 if 'x_train_pca_2' in locals() else None,
        'x_train_pca_3': x_train_pca_3 if 'x_train_pca_3' in locals() else None,
        'x_train_pca_100': x_train_pca_100 if 'x_train_pca_100' in locals() else None,
        'x_train_pca_300': x_train_pca_300 if 'x_train_pca_300' in locals() else None,
    }

def run_lda_analysis(x_train, y_train, train_selfie_indices):
    """
    Run LDA analysis with different numbers of components.
    """
    print("Par I\n\nPreparing LDA...")

    #lda with 2 components for visualization
    print("preparing LDA with 2 components...")
    try:
        lda_vectors_2, x_train_lda_2 = compute_lda(x_train, y_train, 2)
        print(f"LDA vectors shape: {lda_vectors_2.shape} | transformed data shape : {x_train_lda_2.shape}")

        #visualize 2d lda
        plot_lda_2d(x_train_lda_2, y_train, train_selfie_indices,
                   save_path='Outputs/lda/lda_2d_visualization.png', show=False)
        print("2D LDA plot saved to Outputs/lda/lda_2d_visualization.png")

        #visualize fisherfaces
        plot_fisherfaces(lda_vectors_2, n_components=2,
                        save_path='Outputs/lda/fisherfaces_2.png', show=False)
        print("Fisherfaces (2 components) saved to Outputs/lda/fisherfaces_2.png")

    except Exception as e:
        print(f"Error in LDA with 2 components: {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    #lda with 3 components for visualization
    print("preparing data for LDA with 3 components...")
    try:
        lda_vectors_3, x_train_lda_3 = compute_lda(x_train, y_train, 3)
        print(f"LDA vectors shape: {lda_vectors_3.shape} | transformed data shape : {x_train_lda_3.shape}")

        #visualize 3d lda
        plot_lda_3d(x_train_lda_3, y_train, train_selfie_indices,
                   save_path='Outputs/lda/lda_3d_visualization.png', show=False)
        print("3D LDA plot saved to Outputs/lda/lda_3d_visualization.png")

        #visualize fisherfaces
        plot_fisherfaces(lda_vectors_3, n_components=3,
                        save_path='Outputs/lda/fisherfaces_3.png', show=False)
        print("Fisherfaces (3 components) saved to Outputs/lda/fisherfaces_3.png")

    except Exception as e:
        print(f"Error in LDA with 3 components: {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    #lda with 10 components
    print("preparing data for LDA with 10 components...")
    try:
        lda_vectors_10, x_train_lda_10 = compute_lda(x_train, y_train, 10)
        print(f"LDA vectors shape: {lda_vectors_10.shape} | transformed data shape : {x_train_lda_10.shape}")

        #save all features for Part 2
        np.save('Outputs/features/lda_10_features.npy', x_train_lda_10)
        np.save('Outputs/features/lda_10_vectors.npy', lda_vectors_10)
        print("Features saved to Outputs/features/lda_10_features.npy")

    except Exception as e:
        print(f"Error in LDA with 10 components: {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    #lda with 20 components
    print("preparing data for LDA with 20 components...")
    try:
        lda_vectors_20, x_train_lda_20 = compute_lda(x_train, y_train, 20)
        print(f"LDA vectors shape: {lda_vectors_20.shape} | transformed data shape : {x_train_lda_20.shape}")

        # Save features for Part 2
        np.save('Outputs/features/lda_20_features.npy', x_train_lda_20)
        np.save('Outputs/features/lda_20_vectors.npy', lda_vectors_20)
        print("Features saved to Outputs/features/lda_20_features.npy")

    except Exception as e:
        print(f"Error in LDA with 20 components: {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")

    return {
        'lda_vectors_2': lda_vectors_2 if 'lda_vectors_2' in locals() else None,
        'lda_vectors_3': lda_vectors_3 if 'lda_vectors_3' in locals() else None,
        'lda_vectors_10': lda_vectors_10 if 'lda_vectors_10' in locals() else None,
        'lda_vectors_20': lda_vectors_20 if 'lda_vectors_20' in locals() else None,
        'x_train_lda_2': x_train_lda_2 if 'x_train_lda_2' in locals() else None,
        'x_train_lda_3': x_train_lda_3 if 'x_train_lda_3' in locals() else None,
        'x_train_lda_10': x_train_lda_10 if 'x_train_lda_10' in locals() else None,
        'x_train_lda_20': x_train_lda_20 if 'x_train_lda_20' in locals() else None,
    }

def main() -> None:
    """
    Run face recognition with PCA and LDA.

    Steps:
        - Dataset Preparation
        - PCA Analysis
        - LDA Analysis
    
    Returns:
        None
    """

    try:
        #prepare data
        x_train, x_test, y_train, y_test, valid_subjects, train_selfie_indices, test_selfie_indices = prepare_dataset()

        #pca analysis
        pca_results = run_pca_analysis(x_train, y_train, train_selfie_indices)

        #lda analysis
        lda_results = run_lda_analysis(x_train, y_train, train_selfie_indices)

        print(f"Part I : completed successfully")

    except Exception as e:
        print(f"\nError in main execution: {e}")
        print(f"Full Traceback : {traceback.print_exc(limit=3)}")
        sys.exit(1)

if __name__ == "__main__":
    main()