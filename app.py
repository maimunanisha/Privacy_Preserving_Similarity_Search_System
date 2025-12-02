import streamlit as st
import numpy as np
import os
import json
import logging
import itertools
import base64
import io
from datetime import datetime

from phe import paillier
from scipy.spatial import distance
from sklearn.decomposition import PCA
import plotly.express as px

# =====================
# LOGGING AND MONITORING
# =====================
class PrivacyLogger:
    def __init__(self, log_file='privacy_search.log'):
        """
        Comprehensive logging for privacy-preserving operations
        """
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_search_operation(self, query_vector, results, user_id=None):
        """
        Log each similarity search operation
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_vector': query_vector.tolist(),
            'results_count': len(results),
            'user_id': user_id or 'anonymous'
        }
        self.logger.info(f"Similarity Search: {json.dumps(log_entry)}")

# =====================
# DATASET VALIDATION
# =====================
def validate_dataset(dataset):
    """
    Advanced dataset validation function
    """
    try:
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError("Dataset must contain only numeric values")

        if np.isnan(dataset).any():
            st.warning("Dataset contains missing values. Filling missing values with 0.")
            dataset = np.nan_to_num(dataset)

        z_scores = np.abs((dataset - dataset.mean()) / dataset.std())
        outliers = np.where(z_scores > 3)
        if len(outliers[0]) > 0:
            st.warning(f"Detected {len(outliers[0])} potential outliers in the dataset")

        return True, dataset
    except Exception as e:
        st.error(f"Dataset Validation Error: {e}")
        return False, None

# =====================
# ENCRYPTION FUNCTIONS
# =====================
def generate_keys():
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

class MultiLayerEncryption:
    def __init__(self, public_key):
        self.public_key = public_key
        self.salt = os.urandom(16)
    
    def encrypt(self, data):
        return [self.public_key.encrypt(float(x)) for x in data]

    def xor_encrypt(self, data, salt):
        return bytes(a ^ b for a, b in zip(str(data).encode(), itertools.cycle(salt)))

# =====================
# DOWNLOAD UTILITY
# =====================
def create_download_link(data, filename):
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, data, delimiter=",", fmt='%s')
    b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Encrypted Dataset</a>'
    return href

# =====================
# VP-TREE IMPLEMENTATION
# =====================
class OptimizedVPTree:
    def __init__(self, encrypted_data, private_key, distance_metric='euclidean'):
        self.private_key = private_key
        self.data = encrypted_data
        self.distance_metric = distance_metric
        self.cache = {}
        self.distance_func = {
            'euclidean': distance.euclidean,
            'manhattan': distance.cityblock,
            'cosine': distance.cosine
        }.get(distance_metric, distance.euclidean)
        self.tree = self.build_tree(self.data)

    def decrypt_row(self, row):
        return [self.private_key.decrypt(cell) for cell in row]

    def cached_distance(self, point1, point2):
        cache_key = (tuple(point1), tuple(point2))
        if cache_key not in self.cache:
            self.cache[cache_key] = self.distance_func(point1, point2)
        return self.cache[cache_key]

    def build_tree(self, data):
        if len(data) == 0:
            return None
        vp = self.decrypt_row(data[0])
        st.write("Vantage Point (Decrypted):", vp)

        if len(data) == 1:
            return {'vp': vp, 'mu': None, 'left': None, 'right': None}

        distances = [self.cached_distance(vp, self.decrypt_row(point)) for point in data[1:]]
        mu = np.median(distances)

        left = [data[i + 1] for i, d in enumerate(distances) if d <= mu]
        right = [data[i + 1] for i, d in enumerate(distances) if d > mu]

        return {
            'vp': vp,
            'mu': mu,
            'left': self.build_tree(left),
            'right': self.build_tree(right)
        }

    def search(self, query, k=3):
        decrypted_query = query
        distances = [
            self.cached_distance(decrypted_query, self.decrypt_row(row))
            for row in self.data
        ]
        nearest_indices = np.argsort(distances)[:k]
        return [self.decrypt_row(self.data[i]) for i in nearest_indices]

# =====================
# MAIN STREAMLIT APPLICATION
# =====================
def main():
    st.title("Privacy-Preserving Similarity Search")

    # 🌙 Dark Mode Toggle
    dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode")
    if dark_mode:
        st.markdown("""
            <style>
            .stApp { background-color: #121212; color: white; }
            section[data-testid="stSidebar"] { background-color: #1e1e1e; color: white; }
            </style>
        """, unsafe_allow_html=True)

    # 🎨 Gradient for Header & Sidebar
    st.markdown("""
        <style>
        .st-emotion-cache-10trblm {
            background: linear-gradient(90deg, #6a11cb, #2575fc, #ff6f61);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
            background-size: 200% 200%;
            animation: sidebarGradient 6s ease infinite;
        }
        @keyframes sidebarGradient {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        </style>
    """, unsafe_allow_html=True)

    st.write("Upload your dataset, encrypt it, and perform similarity searches using advanced encryption and VP-tree.")

    privacy_logger = PrivacyLogger()
    st.sidebar.header("Dataset Management")

    public_key, private_key = generate_keys()
    multi_layer_encryption = MultiLayerEncryption(public_key)

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            dataset = np.genfromtxt(uploaded_file, delimiter=",")
            is_valid, cleaned_dataset = validate_dataset(dataset)
            if not is_valid:
                st.stop()

            st.write("Raw Dataset:", cleaned_dataset)

            # 🎬 Step Animation
            st.markdown("""
                <div style="display:flex;align-items:center;gap:15px;">
                    <div style="padding:8px 12px;border-radius:8px;background:#6a11cb;color:white;">🔒 Encrypt</div>
                    <div style="animation:blink 1s infinite;">➡️</div>
                    <div style="padding:8px 12px;border-radius:8px;background:#2575fc;color:white;">⚙️ Compute</div>
                    <div style="animation:blink 1s infinite;">➡️</div>
                    <div style="padding:8px 12px;border-radius:8px;background:#ff6f61;color:white;">🔓 Decrypt</div>
                </div>

                <style>
                @keyframes blink { 50% { opacity: 0; } }
                </style>
            """, unsafe_allow_html=True)

            st.info("⚙️ Performing encryption and computation...")

            encrypted_dataset = [multi_layer_encryption.encrypt(row) for row in cleaned_dataset]
            encrypted_flat = [[str(num.ciphertext()) for num in row] for row in encrypted_dataset]

            st.write("Encrypted Dataset (Ciphertexts):", encrypted_flat)

            vp_tree = OptimizedVPTree(encrypted_dataset, private_key)
            st.write("VP-Tree built successfully.")

            # 🔍 Enhanced Visualization
            if st.sidebar.checkbox("Visualize Dataset Embedding"):
                plot_type = st.sidebar.selectbox("Choose Visualization Type:", ["Scatter Plot", "Histogram"])
                try:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(cleaned_dataset)

                    if plot_type == "Scatter Plot":
                        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], title="Dataset Scatter Plot (PCA Reduced)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Scatter Plot Insights
                        - Each point represents a dataset record after dimension reduction.
                        - Points closer together imply higher similarity.
                        - Helps identify possible clusters and groupings.
                        """)

                    elif plot_type == "Histogram":
                        fig = px.histogram(x=reduced_data[:, 0], title="Dataset Histogram (First PCA Component)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Histogram Insights
                        - Shows distribution of dataset values along the primary PCA component.
                        - Identifies most common value ranges and spread.
                        - Useful to detect variance and skewness.
                        """)

                except Exception as e:
                    st.error(f"Visualization Error: {e}")

            if st.sidebar.button("Download Encrypted Dataset"):
                download_filename = "encrypted_dataset.csv"
                download_link = create_download_link(encrypted_flat, download_filename)
                st.sidebar.markdown(download_link, unsafe_allow_html=True)

            st.sidebar.header("Query")
            query_input = st.sidebar.text_input("Enter a query (comma-separated values):")
            k = st.sidebar.slider("Select the number of similar results (k):", 1, len(cleaned_dataset), 3)

            if query_input:
                try:
                    query_vector = np.array([float(x) for x in query_input.split(",")])
                    if len(query_vector) != cleaned_dataset.shape[1]:
                        st.error(f"Query must have {cleaned_dataset.shape[1]} values.")
                    else:
                        decrypted_results = vp_tree.search(query_vector, k)
                        st.write(f"Top {k} Similar Records:")
                        st.dataframe(decrypted_results)
                        privacy_logger.log_search_operation(query_vector, decrypted_results)
                except ValueError:
                    st.error("Query input contains invalid or non-numeric values.")
        except Exception as e:
            st.error(f"Dataset Processing Error: {e}")
    else:
        st.info("Please upload a dataset to begin.")

# Run the app
if __name__ == "__main__":
    main()



