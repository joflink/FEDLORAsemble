"""
THIS TEST IS OLD. USE test_real_lora_clustering.py 





Test script for HDBSCAN clustering functionality.

Run this to verify your clustering implementation works correctly.

Usage:
    python training/test_clustering.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.bmatrix_cluster import (
    cluster_client_vectors,
    compute_cluster_centroids,
    analyze_cluster_statistics,
    visualize_clusters,
    save_clustering_results
)


def create_test_data():
    """
    Create synthetic test data with known clusters.
    
    This simulates LoRA B-matrix fingerprints from different clients.
    The clustering should discover these clusters automatically based on
    vector similarity, NOT based on client names.
    
    Returns:
        Dictionary of client_id -> fingerprint vector
    """
    np.random.seed(42)  # For reproducibility
    
    print("Creating synthetic LoRA B-matrix fingerprints...")
    
    # Cluster 0: Similar learned competencies (tight cluster)
    cluster0_base = np.random.randn(1000)
    cluster0_vectors = {
        f'client_{i}': cluster0_base + np.random.randn(1000) * 0.1
        for i in range(5)
    }
    
    # Cluster 1: Different learned competencies (shifted base vector)
    cluster1_base = np.random.randn(1000) + 3.0  # Shifted by 3
    cluster1_vectors = {
        f'client_{i}': cluster1_base + np.random.randn(1000) * 0.1
        for i in range(5, 9)  # clients 5-8
    }
    
    # Cluster 2: Another different learned competency
    cluster2_base = np.random.randn(1000) - 2.0  # Shifted by -2
    cluster2_vectors = {
        f'client_{i}': cluster2_base + np.random.randn(1000) * 0.15
        for i in range(9, 12)  # clients 9-11
    }
    
    # Outliers: Random vectors (don't belong to any cluster)
    # These represent clients with unique learning patterns or potential attackers
    outlier_vectors = {
        f'client_{i}': np.random.randn(1000) * 5.0
        for i in range(12, 14)  # clients 12-13
    }
    
    # Combine all
    all_vectors = {
        **cluster0_vectors,
        **cluster1_vectors,
        **cluster2_vectors,
        **outlier_vectors
    }
    
    print(f"Created {len(all_vectors)} client LoRA fingerprints:")
    print(f"  - Expected cluster 0: 5 clients (similar vectors)")
    print(f"  - Expected cluster 1: 4 clients (shifted vectors)")
    print(f"  - Expected cluster 2: 3 clients (different shift)")
    print(f"  - Expected outliers: 2 clients (random vectors)")
    print(f"\nNote: Clustering should discover these based on vector similarity,")
    print(f"      NOT based on client names. Cluster IDs (0, 1, 2, etc.) are assigned")
    print(f"      automatically by HDBSCAN based on density.")
    
    return all_vectors


def test_clustering():
    """Test the clustering functionality."""
    
    print("\n" + "="*60)
    print("HDBSCAN Clustering Test")
    print("="*60)
    
    # Create test data
    client_vectors = create_test_data()
    
    # Perform clustering
    print("\nPerforming HDBSCAN clustering...")
    labels, cluster_members, membership_probs = cluster_client_vectors(
        client_vectors,
        min_cluster_size=2,  # Minimum 2 clients per cluster
        metric='cosine',
        cluster_selection_epsilon=0.0
    )
    
    # Print results
    print("\n" + "-"*60)
    print("CLUSTERING RESULTS")
    print("-"*60)
    
    client_ids = list(client_vectors.keys())
    print(f"\nLabels (cluster assignments):")
    for i, client_id in enumerate(client_ids):
        label = labels[i]
        prob = membership_probs.get(client_id, 0.0)
        status = "OUTLIER" if label == -1 else f"Cluster {label}"
        print(f"  {client_id:20s} -> {status:15s} (prob: {prob:.3f})")
    
    print(f"\nCluster Members (automatically assigned by HDBSCAN):")
    for cluster_id, members in sorted(cluster_members.items()):
        print(f"  Cluster {cluster_id}: {len(members)} clients")
        for member in members:
            print(f"    - {member}")
    print(f"\nNote: Cluster IDs (0, 1, 2, etc.) are assigned by HDBSCAN based on")
    print(f"      density and similarity. They are NOT based on client names.")
    
    # Compute centroids
    print("\nComputing cluster centroids...")
    centroids = compute_cluster_centroids(client_vectors, cluster_members, metric='cosine')
    print(f"Computed {len(centroids)} cluster centroids")
    
    # Analyze statistics
    print("\nAnalyzing cluster statistics...")
    stats = analyze_cluster_statistics(
        labels, cluster_members, membership_probs, client_vectors
    )
    
    print("\n" + "-"*60)
    print("STATISTICS")
    print("-"*60)
    print(f"Total clients: {stats['num_clients']}")
    print(f"Number of clusters: {stats['num_clusters']}")
    print(f"Number of outliers: {stats['num_outliers']} ({stats['outlier_percentage']:.1f}%)")
    print(f"Average cluster size: {stats['avg_cluster_size']:.1f}")
    print(f"Average membership probability: {stats['avg_membership_probability']:.3f}")
    
    print("\nCluster sizes:")
    for cluster_id, size in stats['cluster_sizes'].items():
        avg_dist = stats['intra_cluster_distances'][cluster_id]['mean']
        print(f"  Cluster {cluster_id}: {size} clients (avg distance: {avg_dist:.3f})")
    
    # Save results
    print("\nSaving results...")
    save_clustering_results(
        labels, cluster_members, membership_probs, stats,
        output_path="test_clustering_results.json"
    )
    
    # Visualize
    print("\nCreating visualization...")
    try:
        visualize_clusters(
            client_vectors, labels, cluster_members,
            output_path="test_clustering_visualization.png",
            method='tsne'
        )
        print("✓ Visualization saved to: test_clustering_visualization.png")
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
        print("  (This is okay if matplotlib/sklearn not installed)")
    
    # Validation
    print("\n" + "-"*60)
    print("VALIDATION")
    print("-"*60)
    
    expected_clusters = 3  # We created 3 clusters
    found_clusters = stats['num_clusters']
    
    if found_clusters >= 2:  # At least 2 clusters (might merge some)
        print("SUCCESS: Clustering found multiple clusters")
    else:
        print("WARNING: Expected more clusters. Try adjusting min_cluster_size.")
    
    if stats['num_outliers'] >= 1:
        print("SUCCESS: Outliers detected")
    else:
        print("WARNING: No outliers detected (might be okay)")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check test_clustering_results.json for detailed results")
    print("2. View test_clustering_visualization.png for visual representation")
    print("3. Integrate clustering into strategy.py (see CLUSTERING_GUIDE.md)")


if __name__ == "__main__":
    try:
        test_clustering()
    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}")
        print("\nInstall required packages:")
        print("  pip install hdbscan scikit-learn matplotlib")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

