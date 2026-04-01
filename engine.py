import os
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import zipfile
import plotly.express as px

# Initialize Model (Global Cache)
# Using CLIP ViT-B/32 for balanced speed and accuracy
MODEL = SentenceTransformer('clip-ViT-B-32')

# Candidate Labels for Zero-Shot Classification (Smart Naming)
CANDIDATE_LABELS = [
    "a photo of a person", "a photo of a car", "a photo of a building",
    "a photo of nature", "a photo of an animal", "a photo of food",
    "a photo of text or document", "a photo of technology", "a photo of sports",
    "a photo of indoor scene", "a photo of outdoor scene"
]

def extract_features(image_paths):
    """
    ETL Step: Extract features from images using CLIP model.
    Returns: List of paths and numpy array of embeddings.
    """
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not images:
        return [], np.array([])
    
    # Generate Embeddings (The "Feature Extraction")
    embeddings = MODEL.encode(images, batch_size=16, show_progress_bar=False)
    return valid_paths, embeddings

def save_to_warehouse(paths, embeddings, warehouse_dir="data_warehouse"):
    """
    DW Step: Store vectors and metadata in a structured format (CSV).
    """
    os.makedirs(warehouse_dir, exist_ok=True)
    
    # Create a DataFrame (Structured Table)
    df = pd.DataFrame({
        'file_path': paths,
        'file_name': [os.path.basename(p) for p in paths],
        # Store embeddings as string lists for CSV compatibility
        'vector_embedding': [str(e.tolist()) for e in embeddings] 
    })
    
    # Save to CSV (Simulating a Warehouse Table)
    df.to_csv(os.path.join(warehouse_dir, "feature_table.csv"), index=False)
    return df

def cluster_and_organize(paths, embeddings, n_clusters=5, output_dir="output_sorted"):
    """
    Mining Step: Cluster vectors and move files into folders.
    """
    if len(paths) < n_clusters:
        n_clusters = max(1, len(paths))
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Calculate Cluster Centers for Labeling
    cluster_centers = kmeans.cluster_centers_
    
    # Generate Smart Labels for each cluster
    cluster_names = []
    cluster_detections = []
    
    # Encode candidate labels once
    label_embeddings = MODEL.encode(CANDIDATE_LABELS, convert_to_numpy=True)
    
    for i in range(n_clusters):
        # Find which text label is closest to this cluster's center
        center = cluster_centers[i].reshape(1, -1)
        similarities = np.dot(center, label_embeddings.T)[0]
        best_match_idx = np.argmax(similarities)
        
        # Clean up the label name
        raw_label = CANDIDATE_LABELS[best_match_idx]
        clean_label = raw_label.replace("a photo of a ", "").replace("a photo of ", "").capitalize()
        
        # Get top 3 detections for display
        top_3_idx = np.argsort(similarities)[::-1][:3]
        top_3_labels = [CANDIDATE_LABELS[idx].replace("a photo of a ", "").replace("a photo of ", "") for idx in top_3_idx]
        
        cluster_names.append(f"Cluster_{i}_{clean_label}")
        cluster_detections.append(top_3_labels)
        
        # Create folder with smart name
        cluster_folder = os.path.join(output_dir, f"{i}_{clean_label}")
        os.makedirs(cluster_folder, exist_ok=True)
        
        # Move files
        for j, path in enumerate(paths):
            if labels[j] == i:
                shutil.copy(path, cluster_folder)
                
    return labels, cluster_names, cluster_detections

def create_zip_file(output_dir, zip_path="output_sorted.zip"):
    """
    Utility: Zip the sorted output for download.
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    return zip_path

def visualize_clusters(embeddings, labels, cluster_names):
    """
    Mining Visualization: Reduce dimensions to 2D using t-SNE.
    """
    # Reduce dimensions to 2D for plotting
    # perplexity depends on sample size, keep it low for small demos
    perplexity = min(5, len(embeddings) - 1) if len(embeddings) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create DataFrame for Plotly
    df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df['cluster'] = labels.astype(str)
    # Extract clean name for hover
    df['name'] = [cluster_names[l].split('_')[2] if len(cluster_names[l].split('_')) > 2 else f"Cluster {l}" for l in labels]
    
    # Create Interactive Plot
    fig = px.scatter(df, x='x', y='y', color='cluster', hover_name='name', 
                     title="t-SNE Visualization of Image Embeddings",
                     labels={'cluster': 'Cluster ID'},
                     color_discrete_sequence=px.colors.qualitative.Set1)
    return fig

def calculate_duplicates(embeddings):
    """
    Data Quality: Detect near-duplicates using Cosine Similarity.
    """
    if len(embeddings) < 2:
        return 0
    sim_matrix = cosine_similarity(embeddings)
    # Count pairs with similarity > 0.95 (excluding self-comparison which is 1.0)
    duplicates = np.sum((sim_matrix > 0.95) & (sim_matrix < 1.0)) / 2
    return int(duplicates)