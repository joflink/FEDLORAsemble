"""
Test script for clustering real LoRA adapters using HSDBSCAN.

This script:
1. Loads real LoRA adapters from disk
2. Extracts B-matrices from each adapter
3. Vectorizes B-matrices into fingerprint vectors
4. Clusters the vectors using HDBSCAN
5. Validates that adapters from the same domain cluster together

Usage:
    python training/test_real_lora_clustering.py \
        --base-model models/qwens/Qwen2.5-3B-Instruct \
        --lora-dirs lora/0.5B/code lora/0.5B/math lora/0.5B/general lora/0.5B/reasoning
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoModelForCausalLM
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.bmatrix_cluster import (
    cluster_client_vectors,
    compute_cluster_centroids,
    analyze_cluster_statistics,
    visualize_clusters,
    save_clustering_results,
    vectorize_b_matrices,
    get_bmatrix_keys_and_shapes,
)


def load_base_model(base_model_path: str, device: str = "cpu") -> AutoModelForCausalLM:
    """Load the base model ONCE (this is the biggest VS Code crash-prevention step)."""
    print(f"Loading base model from: {base_model_path}")
    # NOTE: we load on CPU by default because we only need weights for extraction.
    # This avoids GPU RAM spikes and makes the run more stable in IDEs.
    return AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # stable numeric extraction
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,  # avoid hub lookups when using local model dirs
    )


def load_lora_adapter(base_model: AutoModelForCausalLM, lora_adapter_path: str) -> PeftModel:
    """
    Load a LoRA adapter from disk.
    
    Args:
        base_model: A loaded base model instance
        lora_adapter_path: Path to the LoRA adapter directory
        
    Returns:
        PeftModel with LoRA adapter loaded
    """
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model, 
        lora_adapter_path,
        local_files_only=True,  # Force local loading, don't try HuggingFace Hub
    )
    
    return model


def extract_b_matrices_from_lora_adapter(
    model: PeftModel,
    bmatrix_keys: List[str]
) -> Dict[str, np.ndarray]:
    """
    Extract B-matrices from a loaded LoRA adapter.
    
    Args:
        model: PeftModel with LoRA adapter loaded
        bmatrix_keys: List of B-matrix key names (from get_bmatrix_keys_and_shapes)
        
    Returns:
        Dictionary mapping B-matrix key -> numpy array
    """
    # Get the complete PEFT state dictionary
    state_dict = get_peft_model_state_dict(model)
    
    # Extract only B-matrices
    b_matrices = {}
    for key in bmatrix_keys:
        if key in state_dict:
            tensor = state_dict[key]
            # Convert to numpy array
            if isinstance(tensor, torch.Tensor):
                b_matrices[key] = tensor.detach().cpu().numpy()
            else:
                b_matrices[key] = np.array(tensor)
        else:
            print(f"Warning: B-matrix key '{key}' not found in adapter")
    
    return b_matrices


def extract_adapter_name(lora_path: str) -> str:
    """
    Extract a readable name from the LoRA adapter path.
    
    e.g:
        lora/0.5B/code -> code
        ../FEDLORAsemble-TGI-Linux/lora/math -> math
    """
    # Get the last directory name
    name = Path(lora_path).name
    
    # If it's empty try parent directory
    if not name or name in ['0.5B', '3B', 'lora']:
        name = Path(lora_path).parent.name
    
    return name


def test_real_lora_clustering(
    base_model_path: str,
    lora_adapter_paths: List[str],
    output_dir: str = "test_results",
    min_cluster_size: int = 2,
    device: str = "cpu",
):
    """
    Test clustering on real LoRA adapters.
    
    Args:
        base_model_path: Path to base model
        lora_adapter_paths: List of paths to LoRA adapter directories
        output_dir: Directory to save results
        min_cluster_size: Minimum cluster size for HDBSCAN
        device: Device to use ('cpu' or 'cuda')
    """
    print("="*70)
    print("Testing HSDBSCAN Clustering with Real LoRA B-Weights")
    print("="*70)
    print(f"\nBase model: {base_model_path}")
    print(f"Number of LoRA adapters: {len(lora_adapter_paths)}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base model once 
    base_model = load_base_model(base_model_path, device=device)
    
    # Load first adapter to get B-matrix structure
    print("\n" + "-"*70)
    print("1. Loading first adapter to determine B-matrix structure")
    print("-"*70)
    
    first_adapter_path = lora_adapter_paths[0]
    first_model = load_lora_adapter(base_model, first_adapter_path)
    
    # Get B-matrix keys and shapes (we need this for consistent vectorization)
    bmatrix_keys, bmatrix_shapes = get_bmatrix_keys_and_shapes(first_model)
    print(f"Found {len(bmatrix_keys)} B-matrices")
    print(f"Example keys: {bmatrix_keys[:3]}")
    
    # Get all PEFT keys for reference
    all_peft_keys = list(get_peft_model_state_dict(first_model).keys())
    print(f"Total PEFT parameters: {len(all_peft_keys)}")
    
    # Extract B-matrices from all adapters
    print("\n" + "-"*70)
    print("2. Extracting B-matrices from all adapters")
    print("-"*70)
    
    adapter_vectors = {}
    adapter_names = {}
    
    for i, lora_path in enumerate(lora_adapter_paths):
        adapter_name = extract_adapter_name(lora_path)
        adapter_names[f"adapter_{i}"] = adapter_name
        
        print(f"\n[{i+1}/{len(lora_adapter_paths)}] Processing: {adapter_name}")
        print(f"  Path: {lora_path}")
        
        try:
            # Load adapter
            model = load_lora_adapter(base_model, lora_path)
            
            # Extract B-matrices
            b_matrices = extract_b_matrices_from_lora_adapter(model, bmatrix_keys)
            
            if len(b_matrices) != len(bmatrix_keys):
                print(f"  Warning: Expected {len(bmatrix_keys)} B-matrices, got {len(b_matrices)}")
            
            # Vectorize B-matrices
            fingerprint_vector = vectorize_b_matrices(b_matrices, bmatrix_keys)
            
            adapter_vectors[f"adapter_{i}"] = fingerprint_vector
            print(f"  ✓ Extracted {len(b_matrices)} B-matrices")
            print(f"  ✓ Vector length: {len(fingerprint_vector)}")
            
            # Free memory
            del model
            # To load on GPU this is supposed to help
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            print(f"Error loading adapter: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(adapter_vectors) < 2:
        print("\nERROR: Need at least 2 adapters to cluster!")
        return
    
    print(f"\nSuccessfully processed {len(adapter_vectors)} adapters")
    
    # Check diagnostics
    print("\n" + "-"*70)
    print("3. Computing pairwise distances")
    print("-"*70)
    
    adapter_ids = list(adapter_vectors.keys())
    vectors_matrix = np.array([adapter_vectors[cid] for cid in adapter_ids])
    
    # Normalize for cosine distance
    norms = np.linalg.norm(vectors_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vectors = vectors_matrix / norms
    
    # Compute pairwise cosine distances
    from sklearn.metrics.pairwise import cosine_distances
    distance_matrix = cosine_distances(normalized_vectors)
    
    print("\nPairwise Cosine Distances Between Adapters:")
    for i, adapter_id_i in enumerate(adapter_ids):
        adapter_name_i = adapter_names.get(adapter_id_i, adapter_id_i)
        for j, adapter_id_j in enumerate(adapter_ids):
            if i < j:  # Only show upper triangle
                adapter_name_j = adapter_names.get(adapter_id_j, adapter_id_j)
                distance = distance_matrix[i, j]
                print(f"  {adapter_name_i:15s} <-> {adapter_name_j:15s}: {distance:.4f}")
    
    avg_distance = np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
    print(f"\nAverage pairwise distance: {avg_distance:.4f}")
    print(f"  (Lower = more similar, Higher = more different)")
    print(f"  (If all distances > 0.5, adapters are very different)")
    
    # Perform clustering
    print("\n" + "-"*70)
    print("4. Performing HDBSCAN clustering")
    print("-"*70)
    
    labels, cluster_members, membership_probs = cluster_client_vectors(
        adapter_vectors,
        min_cluster_size=min_cluster_size,
        metric='cosine',
        cluster_selection_epsilon=0.0
    )
    
    # Analyze results
    print("\n" + "-"*70)
    print("5. Analyzing clustering results")
    print("-"*70)
    
    stats = analyze_cluster_statistics(
        labels, cluster_members, membership_probs, adapter_vectors
    )
    
    # Print results
    print(f"\nClustering Results:")
    print(f"  Total adapters: {stats['num_clients']}")
    print(f"  Number of clusters: {stats['num_clusters']}")
    print(f"  Number of outliers: {stats['num_outliers']} ({stats['outlier_percentage']:.1f}%)")
    print(f"  Average membership probability: {stats['avg_membership_probability']:.3f}")
    
    print(f"\nCluster Details:")
    adapter_ids = list(adapter_vectors.keys())
    for cluster_id, members in sorted(cluster_members.items()):
        print(f"  Cluster {cluster_id}: {len(members)} adapters")
        for member_id in members:
            adapter_name = adapter_names.get(member_id, member_id)
            prob = membership_probs.get(member_id, 0.0)
            print(f"    - {adapter_name} (prob: {prob:.3f})")
        avg_dist = stats['intra_cluster_distances'][cluster_id]['mean']
        print(f"    Average intra-cluster distance: {avg_dist:.4f}")
    
    # Print outliers
    outlier_indices = [i for i, label in enumerate(labels) if label == -1]
    if outlier_indices:
        print(f"\nOutliers:")
        for idx in outlier_indices:
            adapter_id = adapter_ids[idx]
            adapter_name = adapter_names.get(adapter_id, adapter_id)
            print(f"  - {adapter_name}")
    
    # Validate clustering
    print("\n" + "-"*70)
    print("Step 6. Validating clustering quality")
    print("-"*70)
    
    # Check if adapters with same domain name are in same cluster
    domain_to_cluster = {}
    for adapter_id, adapter_name in adapter_names.items():
        if adapter_id in adapter_vectors:
            idx = adapter_ids.index(adapter_id)
            cluster_id = labels[idx]
            if adapter_name not in domain_to_cluster:
                domain_to_cluster[adapter_name] = []
            domain_to_cluster[adapter_name].append(cluster_id)
    
    print("\nDomain-to-Cluster Mapping:")
    validation_passed = True
    for domain, cluster_ids in domain_to_cluster.items():
        unique_clusters = set(cluster_ids)
        if len(unique_clusters) == 1:
            status = "passed"
        else:
            status = "failed"
            validation_passed = False
        print(f"  {status} {domain}: clusters {sorted(unique_clusters)}")
    
    if validation_passed:
        print("\n Validation passed: Adapters from same domain are in same cluster")
    else:
        print("\n Validation failed: Some adapters from same domain are in different clusters")
    
    # Save results and visualization
    print("\n" + "-"*70)
    print("7. Saving results")
    print("-"*70)
    
    # Save JSON results
    json_path = os.path.join(output_dir, "real_lora_clustering_results.json")
    save_clustering_results(
        labels, cluster_members, membership_probs, stats,
        output_path=json_path
    )
    print(f"Saved results to: {json_path}")
    
    # Create visualization
    try:
        viz_path = os.path.join(output_dir, "real_lora_clustering_visualization.png")
        # PCA is far more stable than t-SNE in many VM/conda setups (and avoids common segfaults)
        visualize_clusters(
            adapter_vectors,
            labels,
            cluster_members,
            output_path=viz_path,
            method="pca",
        )
        print(f"Saved visualization to: {viz_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Processed {len(adapter_vectors)} LoRA adapters")
    print(f"Found {stats['num_clusters']} clusters")
    print(f"Average membership probability: {stats['avg_membership_probability']:.3f}")
    print(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
    print(f"\nResults saved to: {output_dir}")
    print("="*70)
    
    return {
        'adapter_vectors': adapter_vectors,
        'labels': labels,
        'cluster_members': cluster_members,
        'stats': stats,
        'validation_passed': validation_passed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test HSDBSCAN clustering with real LoRA adapters"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model (e.g., models/qwens/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--lora-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to LoRA adapter directories (e.g., lora/code lora/math)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Directory to save results (default: test_results)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size for HDBSCAN (default: 2; HDBSCAN requires > 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)"
    )

    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.base_model):
        print(f"ERROR: Base model not found: {args.base_model}")
        return
    
    for lora_path in args.lora_dirs:
        if not os.path.exists(lora_path):
            print(f"ERROR: LoRA adapter not found: {lora_path}")
            return
    
    # Run test
    try:
        test_real_lora_clustering(
            base_model_path=args.base_model,
            lora_adapter_paths=args.lora_dirs,
            output_dir=args.output_dir,
            min_cluster_size=args.min_cluster_size,
            device=args.device,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

