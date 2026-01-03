# extract_matrices_from_parameters(prameters: Parameters, key_shapes: Dict[str, Tuple]) -> Dict[str, np.ndarray]


# vectorize_b_matrices(b_matrices: Dict[str, Dict[str, np.ndarray]]) -> Dict[str.npdarray]


# run_dbscan_clustering(vectors: Dict[str, np.ndarray], eps: float, min_samples: int) -> Tuple[np.ndarray, Dict[int, List[str]]]
"""
B-Matrix Clustering Module for FedLoRA-Orchestrator

This module handles extraction, vectorization, and clustering of LoRA B-matrices
from federated learning clients. The B-matrices represent the "fingerprint" of
what each client has learned.
"""
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.common import Parameters
from flwr.common.parameter import parameters_to_ndarrays
from peft import get_peft_model_state_dict
import torch
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")

# Visualization imports (optional, for testing)
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: sklearn/matplotlib not installed. Visualization disabled.")
# from sklearn.cluster import HDBSCAN

# B-matrices are like a "fingerprint" of the information that the model learns which can be extracted
def get_bmatrix_keys_and_shapes(model) -> Tuple[List[str], Dict[str, Tuple[int, ...]]]:
    """
    Extract B-matrix keys and their shapes from PEFT model.
    
    This function identifies which parameters in the model are B-matrices
    (the "up" projection in LoRA) and records their shapes for later reconstruction from vectorizing. 
    This function only runs once.
    
    Args:
        model: A PEFT model (from get_peft_model() in AImodels.py)
        
    Returns:
        Tuple of:
        - bmatrix_keys: List of parameter key names ending with '.lora_B.weight'
        - bmatrix_shapes: Dictionary mapping key name -> shape tuple
        
    Example:
        bmatrix_keys = ['base_model.model.layers.0.self_attn.q_proj.lora_B.weight', ...]
        bmatrix_shapes = {'base_model.model.layers.0.self_attn.q_proj.lora_B.weight': (32, 128), ...}
    """
    # Get the complete PEFT state dictionary so we can check for B-matrices
    # This returns an OrderedDict with all LoRA parameters (both A and B matrices)
    state_dict = get_peft_model_state_dict(model)
    
    # Initialize lists/dicts to store results
    bmatrix_keys = []
    bmatrix_shapes = {}
    
    # Iterate through all parameter keys in the state dict
    for key in state_dict.keys():
        # Check if this key represents a B-matrix, which we can check like this:
        # B-matrices in PEFT always end with '.lora_B.weight'
        if key.endswith('.lora_B.weight'):
            # Add the key to our list (IMPORTANT: maintains order)
            bmatrix_keys.append(key)
            
            # Get the tensor shape and store it (maps to each key)
            # We need this shape later to reconstruct the matrix from a flattened vector
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                shape = tuple(tensor.shape)
            else:
                # Even if it's already a numpy array, it still needs to be converted to a tuple
                shape = tuple(tensor.shape)
            
            # Updated list with the shape for each key
            bmatrix_shapes[key] = shape
    
    return bmatrix_keys, bmatrix_shapes


def extract_b_matrices_from_parameters(
    parameters: Parameters,
    bmatrix_keys: List[str],
    all_peft_keys: List[str]
) -> Dict[str, np.ndarray]:
    """
    Extract B-matrix arrays from Flower 'Parameters' object.
    
    Flower sends parameters as a Parameters object containing serialized tensors (format we can save/send).
    This function converts them to numpy arrays and extracts only the B-matrices.
    (Parameters object is a wrapper around a list of byte strings so you need to convert)
    
    Args:
        parameters: Flower Parameters object from FitRes.parameters (Fit response from a client)
        bmatrix_keys: List of B-matrix key names (from get_bmatrix_keys_and_shapes)
        all_peft_keys: Complete list of all PEFT parameter keys in correct order
        
    Returns:
        Dictionary mapping B-matrix key name -> numpy array
        
    Example:
        {
            'base_model.model.layers.0.self_attn.q_proj.lora_B.weight': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'base_model.model.layers.0.self_attn.v_proj.lora_B.weight': np.array([[0.5, 0.6], [0.7, 0.8]]),
            ...
        }
    """
    # 1: Convert Flower Parameters to list of numpy arrays
    # parameters_to_ndarrays() deserializes the tensors and returns them as numpy arrays
    # The arrays are in the SAME ORDER as all_peft_keys
    ndarrays = parameters_to_ndarrays(parameters)
    
    # 2: Create a mapping from key name to numpy array
    # This assumes the order of ndarrays matches the order of all_peft_keys
    # zip() pairs them up: first key with first array, second key with second array, etc.
    key_to_array = dict(zip(all_peft_keys, ndarrays))
    
    # 3: Extract only the B-matrices we care about
    b_matrices = {}
    for key in bmatrix_keys:
        # Check if this key exists in the parameters
        # (It should always exist, but good to check for safety)
        if key in key_to_array:
            b_matrices[key] = key_to_array[key]
        else:
            # If key is missing, this might indicate a problem
            # Could maybe raise an error or skip it, for now we skip with a warning
            print(f"Warning: B-matrix key '{key}' not found in parameters")
    
    return b_matrices


def vectorize_b_matrices(
    b_matrices: Dict[str, np.ndarray],
    bmatrix_keys: List[str]
) -> np.ndarray:
    """
    Flatten and concatenate B-matrices into a single 1D vector (Row-Major).
    
    This creates a vector representing the client's learned competency (call it a "fingerprint").
    The order of concatenation is determined by bmatrix_keys to make sure it's consistent.
    
    Args:
        b_matrices: Dictionary of B-matrix arrays (from extract_b_matrices_from_parameters)
        bmatrix_keys: List of B-matrix keys in the desired concatenation order
        
    Returns:
        1D numpy array (the fingerprint vector)
        
    Example:
        If we have two B-matrices:
        - 'key1': shape (32, 128) -> 4096 elements when flattened
        - 'key2': shape (32, 128) -> 4096 elements when flattened
        Result: array of length 8192 (4096 + 4096)
    """
    # Initialize list to store flattened parts
    flattened_parts = []
    
    # Iterate through keys in the specified order
    # This ensures consistent vectorization across all clients
    for key in bmatrix_keys:
        # Get the B-matrix for this key
        if key not in b_matrices: # shouldn't happen but good to check
            raise ValueError(f"B-matrix key '{key}' not found in b_matrices dict")
        
        matrix = b_matrices[key]
        
        # Flatten the matrix to 1D
        # .flatten() creates a 1D copy of the array
        # .ravel() creates a view (faster but can have weird side effects so I'll skip for now)
        flattened = matrix.flatten()
        
        # Append to our list
        flattened_parts.append(flattened)
    
    # Concatenate all flattened parts into a single 1D array
    # This is the "fingerprint vector" for this client
    fingerprint_vector = np.concatenate(flattened_parts)
    
    return fingerprint_vector


def extract_and_vectorize_all_clients(
    results: List[Tuple],
    bmatrix_keys: List[str],
    all_peft_keys: List[str]
) -> Dict[str, np.ndarray]:
    """
    Extract and vectorize B-matrices for all clients in a federated round.
    
    This is the main function we call from strategy.py. It processes all
    client results and creates fingerprint vectors for clustering.
    
    Args:
        results: List of (ClientProxy, FitRes) tuples from aggregate_fit()
        bmatrix_keys: List of B-matrix key names
        all_peft_keys: Complete list of all PEFT parameter keys
        
    Returns:
        Dictionary mapping client_id -> fingerprint vector (1D numpy array)
        
    Example:
        {
            'client_1': np.array([0.1, 0.2, 0.3, ...]),  # length depends on model
            'client_2': np.array([0.2, 0.3, 0.4, ...]),
            ...
        }
    """
    # Initialize dictionary to store client vectors
    client_vectors = {}
    
    # Iterate through all client results
    for client_proxy, fit_res in results:
        # Get client identifier
        # ClientProxy has a 'cid' attribute that uniquely identifies the client
        client_id = client_proxy.cid
        
        # Get parameters from this client's training result
        parameters = fit_res.parameters
        
        # Extract B-matrices for this client
        # This gives us a dict: {key: numpy_array}
        b_matrices = extract_b_matrices_from_parameters(
            parameters,
            bmatrix_keys,
            all_peft_keys
        )
        
        # Vectorize the B-matrices into a single fingerprint vector
        fingerprint_vector = vectorize_b_matrices(b_matrices, bmatrix_keys)
        
        # Store the vector with client_id as key
        client_vectors[client_id] = fingerprint_vector
    
    return client_vectors


# Avoid DBSCAN because we want to avoid having to care abot Epislon and Epislon decay, too tedious
# The magnitude of model updates changes over time which makes it innefficient at making good clusters
# We don't want to hardcode Epsilon (distance between points)
# For clustering directional updates I think Cosine seems better than Euclidean since it measures angle/direction compared to magnitude

# CLUSTERING FUNCTIONS
def cluster_client_vectors(
    client_vectors: Dict[str, np.ndarray],
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    metric: str = 'cosine',
    cluster_selection_epsilon: float = 0.0
) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[str, float]]:
    """
    Cluster client fingerprint vectors using HDBSCAN.
    
    HDBSCAN (Hierarchical Density-Based Spatial Clustering) is better than DBSCAN because:
    - Handles varying density clusters better
    - More robust to parameter tuning
    - Returns cluster membership probabilities
    - Better at identifying outliers
    
    Args:
        client_vectors: Dictionary mapping client_id -> fingerprint vector
        min_cluster_size: Minimum number of clients in a cluster (default: 3)
        min_samples: Minimum samples in neighborhood (default: same as min_cluster_size)
        metric: Distance metric ('cosine', 'euclidean', etc.)
        cluster_selection_epsilon: Controls cluster selection (0.0 = strict, higher = more clusters)
        
    Returns:
        Tuple of:
        - labels: Array of cluster labels (-1 = outlier/noise, 0+ = cluster ID)
        - cluster_members: Dictionary mapping cluster_id -> list of client_ids
        - membership_probs: Dictionary mapping client_id -> membership probability
        
    Example:
        labels = [-1, 0, 0, 1, 1, -1]  # client_0=outlier, client_1&2=cluster0, client_3&4=cluster1, client_5=outlier
        cluster_members = {
            0: ['client_1', 'client_2'],
            1: ['client_3', 'client_4']
        }
        membership_probs = {
            'client_1': 0.95,
            'client_2': 0.87,
            ...
        }
    """
    # This is for others
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is required. Install with: pip install hdbscan")
    
    # HDBSCAN cannot read a Python dictionary 
    # It needs a raw Numpy Matrix (Rows = Clients, Cols = Features)
    client_ids = list(client_vectors.keys())
    vectors_matrix = np.array([client_vectors[cid] for cid in client_ids])
    
    if metric == 'cosine':
            # Normalize each vector to unit length
            norms = np.linalg.norm(vectors_matrix, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            vectors_matrix = vectors_matrix / norms # To get the topic (all are now on a hypersphere)
            # Use euclidean on normalized vectors (equivalent to cosine since lenght are all 1.0)
            actual_metric = 'euclidean' 
    else:
            actual_metric = metric

    # Set default min_samples if not provided
    # min_cluster_size: The smallest group allowed to be an "Expert"
    # min_samples: How conservative the algorithm is (so higher = more points become noise)
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Initialize HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=actual_metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True  # Enable prediction for new points
    )
    
    # Perform clustering
    # It returns an array of integers: labels = [0, 0, 1, -1, 1] 
    # (Client 0 is in Cluster 0, Client 3 is Noise/-1)
    labels = clusterer.fit_predict(vectors_matrix)
    
    # Extract cluster membership probabilities
    membership_probs = {}
    if hasattr(clusterer, 'probabilities_'):
        for i, client_id in enumerate(client_ids):
            membership_probs[client_id] = float(clusterer.probabilities_[i])
    else:
        # If probabilities not available, set to 1.0 for in-cluster, 0.0 for outliers
        for i, client_id in enumerate(client_ids):
            membership_probs[client_id] = 1.0 if labels[i] != -1 else 0.0
    
    # Organize clients by cluster
    cluster_members = {}
    for i, client_id in enumerate(client_ids):
        cluster_id = int(labels[i])
        if cluster_id == -1:
            # Outliers - we might want to track these separately
            continue
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(client_id)
    
    return labels, cluster_members, membership_probs


def compute_cluster_centroids(
    client_vectors: Dict[str, np.ndarray],
    cluster_members: Dict[int, List[str]],
    metric: str = 'cosine'
) -> Dict[int, np.ndarray]:
    """
    Compute centroid (mean) vector for each cluster.
    
    The centroid represents the "average" learned competency of clients in that cluster.
    This will be used later to initialize new experts.
    
    Args:
        client_vectors: Dictionary mapping client_id -> fingerprint vector
        cluster_members: Dictionary mapping cluster_id -> list of client_ids
        metric: Distance metric used ('cosine' or 'euclidean')
        
    Returns:
        Dictionary mapping cluster_id -> centroid vector
        
    Note:
        For cosine metric, we compute the mean and then normalize.
        For euclidean, we just compute the mean.
    """
    centroids = {}
    
    for cluster_id, member_ids in cluster_members.items():
        # Get vectors for all members of this cluster
        member_vectors = np.array([client_vectors[cid] for cid in member_ids])
        
        # Compute mean/centroid
        if metric == 'cosine':
            # For cosine, compute mean then normalize
            centroid = np.mean(member_vectors, axis=0)

            # Normalize to unit length
            # If you average two points on the surface of a sphere, the midpoint is inside the sphere (under the surface)
            # We must push it back out to the surface so it acts like a valid B-matrix
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
        else:
            # For euclidean, just compute mean
            centroid = np.mean(member_vectors, axis=0)
        
        centroids[cluster_id] = centroid
    
    return centroids


def analyze_cluster_statistics(
    labels: np.ndarray,
    cluster_members: Dict[int, List[str]],
    membership_probs: Dict[str, float],
    client_vectors: Dict[str, np.ndarray]
) -> Dict[str, any]:
    """
    Compute statistics about the clustering results.
    
    Useful for understanding cluster quality and presenting results
    
    Args:
        labels: Array of cluster labels
        cluster_members: Dictionary mapping cluster_id -> list of client_ids
        membership_probs: Dictionary mapping client_id -> membership probability
        client_vectors: Dictionary mapping client_id -> fingerprint vector
        
    Returns:
        Dictionary with clustering statistics
    """
    num_clients = len(labels)
    num_outliers = np.sum(labels == -1)
    num_clusters = len(cluster_members)
    
    # Cluster sizes
    cluster_sizes = {cid: len(members) for cid, members in cluster_members.items()}
    
    # Average membership probabilities
    in_cluster_probs = [prob for cid, prob in membership_probs.items() if labels[list(client_vectors.keys()).index(cid)] != -1]
    avg_membership_prob = np.mean(in_cluster_probs) if in_cluster_probs else 0.0
    
    # Compute intra-cluster distances (cohesion)
    intra_cluster_distances = {}
    for cluster_id, member_ids in cluster_members.items():
        member_vectors = np.array([client_vectors[cid] for cid in member_ids])
        centroid = np.mean(member_vectors, axis=0)
        
        # Compute cosine distances to centroid
        distances = []
        for vec in member_vectors:
            # Dot product measures similarity
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(vec, centroid)
            norm_vec = np.linalg.norm(vec)
            norm_centroid = np.linalg.norm(centroid)
            if norm_vec > 0 and norm_centroid > 0:
                cosine_sim = dot_product / (norm_vec * norm_centroid)
                cosine_dist = 1 - cosine_sim
                distances.append(cosine_dist)
        
        # Distance is the opposite of Similarity
        # Distance 0 = Perfect Match. Distance 1 = Orthogonal (90 degrees)
        intra_cluster_distances[cluster_id] = {
            'mean': np.mean(distances) if distances else 0.0,
            'std': np.std(distances) if distances else 0.0
        }
    
    stats = {
        'num_clients': num_clients,
        'num_clusters': num_clusters,
        'num_outliers': num_outliers,
        'outlier_percentage': (num_outliers / num_clients * 100) if num_clients > 0 else 0.0,
        'cluster_sizes': cluster_sizes,
        'avg_cluster_size': np.mean(list(cluster_sizes.values())) if cluster_sizes else 0.0,
        'avg_membership_probability': avg_membership_prob,
        'intra_cluster_distances': intra_cluster_distances
    }
    
    return stats


def visualize_clusters(
    client_vectors: Dict[str, np.ndarray],
    labels: np.ndarray,
    cluster_members: Dict[int, List[str]],
    output_path: Optional[str] = None,
    method: str = 'tsne',
    perplexity: float = 30.0
) -> None:
    """
    Visualize client clusters in 2D using dimensionality reduction (t-SNE or PCA).

    - Takes high-dimensional fingerprint vectors and projects them to 2D.
    - Colors each cluster differently, marks outliers with gray 'x'.
    - Can both save the figure (if output_path is given) and try to display it.

    Args:
        client_vectors: Mapping client_id -> fingerprint vector (same length).
        labels: 1D array of cluster labels (-1 = outlier).
        cluster_members: Mapping cluster_id -> list of member client_ids.
        output_path: If set, save the plot to this path.
        method: 'tsne' for t-SNE, 'pca' for PCA.
        perplexity: t-SNE perplexity (automatically clamped to < n_samples).
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install sklearn and matplotlib.")
        return
    
    # Safety: need at least 2 samples to make a scatter plot
    client_ids = list(client_vectors.keys())
    n_samples = len(client_ids)
    
    if n_samples < 2:
        print("Not enough clients to generate a scatter plot (< 2).")
        return

    vectors_matrix = np.array([client_vectors[cid] for cid in client_ids])
    
    # Dimensionality reduction
    if method == 'tsne':
        # t-SNE requires perplexity < n_samples and >= 1
        safe_perplexity = min(perplexity, n_samples - 1)
        if safe_perplexity < 1:
            safe_perplexity = 1

        reducer = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            random_state=42,
            n_iter=1000,
        )
        reduced = reducer.fit_transform(vectors_matrix)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(vectors_matrix)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot outliers first
    outlier_mask = labels == -1
    if np.any(outlier_mask):
        plt.scatter(
            reduced[outlier_mask, 0],
            reduced[outlier_mask, 1],
            c='gray',
            marker='x',
            s=50,
            alpha=0.5,
            label=f'Outliers ({np.sum(outlier_mask)})'
        )
    
    # Plot clusters
    unique_labels = sorted([l for l in set(labels) if l != -1])
    # Use a distinct colormap
    cmap = plt.get_cmap('tab10') 
    
    for i, cluster_id in enumerate(unique_labels):
        cluster_mask = labels == cluster_id
        # Safety: handle cases with more clusters than colors
        color = cmap(i % 10) 
        
        plt.scatter(
            reduced[cluster_mask, 0],
            reduced[cluster_mask, 1],
            c=[color],
            label=f'Cluster {cluster_id} ({np.sum(cluster_mask)})',
            s=100,
            alpha=0.7
        )
    
    plt.title(f'Client Clustering ({method.upper()}) - {len(unique_labels)} clusters')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Fixed Logic for Saving vs Showing
    # We want to SAVE if a path is given. 
    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Clustering visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    # We generally always try to show, but wrap in try/except 
    # because this often crashes on headless servers (like Colab background)
    try:
        plt.show()
    except Exception:
        pass # Ignore show() errors on servers
    
    plt.close()


def save_clustering_results(
    labels: np.ndarray,
    cluster_members: Dict[int, List[str]],
    membership_probs: Dict[str, float],
    stats: Dict[str, any],
    output_path: str = "clustering_results.json"
) -> None:
    """
    Serialize clustering results (labels, cluster memberships, statistics) to JSON.

    - Handles nested numpy types by recursively converting them to native Python types.
    - Safe to use even when 'stats' contains nested dicts with numpy scalars/arrays.

    Args:
        labels: 1D array of cluster labels (-1 = outlier).
        cluster_members: Mapping cluster_id -> list of client_ids in that cluster.
        membership_probs: Mapping client_id -> membership probability (0.0–1.0).
        stats: Dictionary of clustering statistics (can contain nested dicts).
        output_path: Path to write the JSON file to.
    """
    
    # Recursive helper to convert numpy types to JSON-serializable types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    # Construct the final dictionary
    results = {
        'labels': [int(l) for l in labels],
        'cluster_members': {str(k): v for k, v in cluster_members.items()},
        'membership_probs': {k: float(v) for k, v in membership_probs.items()},
        'statistics': convert_numpy_types(stats) # Uses the helper here
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Clustering results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save clustering results: {e}")


# Helper functions for clustering
def reconstruct_b_matrices_from_vector(
    vector: np.ndarray,
    bmatrix_keys: List[str],
    bmatrix_shapes: Dict[str, Tuple[int, ...]]
) -> Dict[str, np.ndarray]:
    """
    Reconstruct B-matrix dictionaries from a flattened vector.
    
    This is the inverse of vectorize_b_matrices(). We need this when
    creating new experts from cluster centroids.
    
    Args:
        vector: 1D numpy array (fingerprint vector)
        bmatrix_keys: List of B-matrix keys in the same order as vectorization
        bmatrix_shapes: Dictionary mapping key -> shape tuple
        
    Returns:
        Dictionary mapping key -> reconstructed B-matrix array
        
    Note: This function will be useful in Phase 3 when creating new experts
    """
    reconstructed = {}
    current_index = 0
    
    for key in bmatrix_keys:
        # Get the shape for this B-matrix
        shape = bmatrix_shapes[key]
        
        # Calculate how many elements this matrix needs
        num_elements = np.prod(shape)  # product of all dimensions
        
        # Extract the slice from the vector
        matrix_flat = vector[current_index:current_index + num_elements]
        
        # Reshape to original shape
        matrix = matrix_flat.reshape(shape)
        
        # Store in dictionary
        reconstructed[key] = matrix
        
        # Move index forward
        current_index += num_elements
    
    return reconstructed


