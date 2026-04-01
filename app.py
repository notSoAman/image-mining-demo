import streamlit as st
import os
import io
import uuid
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
from engine import (
    extract_features, 
    save_to_warehouse, 
    cluster_and_organize, 
    create_zip_file, 
    visualize_clusters,
    calculate_duplicates
)

try:
    from streamlit_paste_button import paste_image_button
except Exception:
    paste_image_button = None

# Page Config
st.set_page_config(page_title="Image Data Warehouse", layout="wide")

# Title & Info
st.title("Intelligent Image Data Mining")
st.markdown("""
**Pipeline:** Ingestion → Feature Extraction → Vector Warehousing → Semantic Clustering
""")

# Sidebar
st.sidebar.header("1. Data Ingestion")
st.sidebar.caption("Use any input method: upload, drag and drop, or paste from clipboard.")

if "pasted_images" not in st.session_state:
    st.session_state.pasted_images = []
if "pasted_image_hashes" not in st.session_state:
    st.session_state.pasted_image_hashes = set()

uploaded_files = st.sidebar.file_uploader(
    "Upload or drag and drop images",
    accept_multiple_files=True,
    type=['jpg', 'png', 'jpeg', 'webp']
)

if paste_image_button:
    with st.sidebar:
        st.markdown("Paste image from clipboard")
        paste_result = paste_image_button("Paste image", key="paste_image")
    if paste_result is not None and getattr(paste_result, "image_data", None) is not None:
        pasted = paste_result.image_data
        if isinstance(pasted, Image.Image):
            pil_image = pasted.convert("RGB")
        else:
            pil_image = Image.fromarray(np.asarray(pasted)).convert("RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_hash = hashlib.md5(image_bytes).hexdigest()

        if image_hash not in st.session_state.pasted_image_hashes:
            st.session_state.pasted_images.append(image_bytes)
            st.session_state.pasted_image_hashes.add(image_hash)
            st.sidebar.success(f"Clipboard image added. Total pasted: {len(st.session_state.pasted_images)}")

    if st.session_state.pasted_images:
        st.sidebar.caption(f"Clipboard images queued: {len(st.session_state.pasted_images)}")
        if st.sidebar.button("Clear pasted images", key="clear_pasted_images"):
            st.session_state.pasted_images = []
            st.session_state.pasted_image_hashes = set()
            st.rerun()
else:
    st.sidebar.info("Install streamlit-paste-button to enable clipboard paste support.")

has_inputs = bool(uploaded_files) or len(st.session_state.pasted_images) > 0

if has_inputs:
    # Save uploads temporarily
    save_dir = "temp_uploads"
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    # Save uploaded/drag-dropped images
    for file in uploaded_files or []:
        original_name = os.path.basename(file.name)
        name_root, ext = os.path.splitext(original_name)
        unique_name = f"{name_root}_{uuid.uuid4().hex[:8]}{ext}"
        path = os.path.join(save_dir, unique_name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(path)

    # Save pasted clipboard images
    for idx, image_bytes in enumerate(st.session_state.pasted_images, start=1):
        unique_name = f"clipboard_{idx}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(save_dir, unique_name)
        with open(path, "wb") as f:
            f.write(image_bytes)
        paths.append(path)
    
    st.success(f"{len(paths)} images ingested.")
    
    # Main Controls
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Mining Parameter: K (Clusters)", 2, 10, 5)
    with col2:
        run_button = st.button("Execute Mining Pipeline")
    
    if run_button:
        with st.spinner("Mining patterns... (This may take a moment on first run)"):
            # 1. Extract
            valid_paths, embeddings = extract_features(paths)
            
            if len(valid_paths) > 0:
                # 2. Warehouse
                df = save_to_warehouse(valid_paths, embeddings)
                
                # 3. Mine & Label
                labels, cluster_names, cluster_detections = cluster_and_organize(valid_paths, embeddings, n_clusters)
                
                st.success("Mining Complete!")
                
                # --- EDUCATIONAL SECTION 1: FEATURE EXTRACTION ---
                with st.expander("How Feature Extraction Works (The 'Black Box')"):
                    st.markdown("""
                    **What is the model actually doing?**
                    1. **Convolutional Layers:** The model scans the image with filters to detect edges, textures, and shapes.
                    2. **Pooling:** It compresses this information, keeping only the most important features.
                    3. **Embedding Layer:** Finally, it outputs a **512-dimensional vector**. 
                       - Think of each number in the vector as a "concept score" (e.g., Index 5 = 'Furriness', Index 20 = 'Roundness').
                       - Images with similar concepts have similar numbers.
                    """)
                    st.code("""
                    Input: [Pixels] -> [CNN Layers] -> [Feature Vector]
                    Example Vector: [0.8 (Dog-like), 0.1 (Car-like), 0.9 (Furry), ...]
                    """, language="python")

                # Show Warehouse Data
                with st.expander("View Data Warehouse Table (Vectors)"):
                    st.dataframe(df.head())
                
                # --- VISUALIZATION SECTION ---
                st.subheader("Vector Space Visualization (t-SNE)")
                
                # --- EDUCATIONAL SECTION 2: T-SNE ---
                st.info("""
                **Understanding this Plot:**
                - **t-SNE** squashes 512 dimensions into 2 dimensions for visualization.
                - **Distance Matters:** Points close together are semantically similar.
                - **Axes are Arbitrary:** The X and Y axes don't have specific meanings. Only the *relative distance* between points matters.
                - **Clusters:** Distinct groups indicate the model found strong semantic patterns.
                """)
                
                fig = visualize_clusters(embeddings, labels, cluster_names)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- DATA QUALITY METRICS ---
                st.subheader("Data Warehouse Metrics")
                col1, col2, col3 = st.columns(3)
                
                # Storage Compression
                original_size = sum(os.path.getsize(p) for p in paths) / (1024 * 1024)
                vector_size = embeddings.nbytes / (1024 * 1024)
                compression_ratio = original_size / vector_size if vector_size > 0 else 0
                col1.metric("Storage Compression", f"{compression_ratio:.1f}x", 
                            delta="Vectors are smaller than images")
                
                # Duplicate Detection
                duplicates = calculate_duplicates(embeddings)
                col2.metric("Near-Duplicates Found", duplicates, 
                            delta="Potential Data Cleaning Needed")
                
                # Cluster Entropy
                if len(labels) > 0:
                    unique, counts = np.unique(labels, return_counts=True)
                    probs = counts / len(labels)
                    entropy = -np.sum(probs * np.log2(probs))
                else:
                    entropy = 0
                col3.metric("Cluster Diversity (Entropy)", f"{entropy:.2f}", 
                            delta="Higher is more balanced")
                
                # --- SORTED OUTPUT ---
                st.subheader("Sorted Output & Detections")
                
                # Download Button
                zip_path = create_zip_file("output_sorted")
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download Sorted Clusters (ZIP)",
                        data=f,
                        file_name="sorted_images.zip",
                        mime="application/zip"
                    )
                
                # Display Cluster Info
                for i in range(n_clusters):
                    with st.container():
                        st.markdown(f"### {cluster_names[i]}")
                        st.info(f"**Detected Concepts:** {', '.join(cluster_detections[i])}")
                        cluster_image_paths = [
                            valid_paths[idx] for idx, cluster_id in enumerate(labels) if cluster_id == i
                        ]
                        st.write(f"Contains **{len(cluster_image_paths)}** images.")

                        if cluster_image_paths:
                            st.caption("Thumbnail preview")
                            columns_per_row = 5
                            cols = st.columns(columns_per_row)
                            for idx, image_path in enumerate(cluster_image_paths):
                                col = cols[idx % columns_per_row]
                                with col:
                                    st.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
                        st.divider()
                
            else:
                st.error("No valid images processed.")
else:
    st.info("Please upload, drag and drop, or paste images from the sidebar to begin the pipeline.")