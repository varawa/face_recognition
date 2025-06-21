import cv2
import os
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import numpy as np
import shutil

# --- SETTINGS ---
ALBUM_DIR = 'album'
FACE_DIR = 'myfaces'
CLUSTERED_DIR = 'clustered_faces'
MODEL_PATH = 'res10_300x300_ssd_iter_140000.caffemodel'
CONFIG_PATH = 'deploy.prototxt'
CONFIDENCE_THRESHOLD = 0.5
EMBEDDING_MODEL = 'Facenet512'  # Better for distinguishing blurred faces
MIN_FACE_SIZE = 50

# --- Simple face enhancement ---
def enhance_face(face):
    """Simple face enhancement for blurred images"""
    if face.size == 0:
        return face
    
    # Resize small faces
    h, w = face.shape[:2]
    if min(h, w) < 80:
        scale = 80 / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Simple denoising
    face = cv2.bilateralFilter(face, 9, 75, 75)
    
    # Mild sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    face = cv2.filter2D(face, -1, kernel)
    
    return face

# --- Step 1: Clean face extraction ---
def extract_faces():
    # Clear existing faces first
    if os.path.exists(FACE_DIR):
        shutil.rmtree(FACE_DIR)
    os.makedirs(FACE_DIR, exist_ok=True)
    
    net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
    count = 0

    print(f"[INFO] Processing images from {ALBUM_DIR}")
    image_files = [f for f in os.listdir(ALBUM_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for filename in tqdm(image_files, desc="Extracting faces"):
        image_path = os.path.join(ALBUM_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        
        # Single scale detection - simple and reliable
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        faces_in_image = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                
                # Basic bounds checking
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face = image[y1:y2, x1:x2]
                if face.size == 0 or min(face.shape[:2]) < MIN_FACE_SIZE:
                    continue
                
                # Simple enhancement
                enhanced_face = enhance_face(face)
                
                # Save with clear naming
                base_name = os.path.splitext(filename)[0]
                face_filename = f"{base_name}_face_{faces_in_image:02d}.jpg"
                face_path = os.path.join(FACE_DIR, face_filename)
                cv2.imwrite(face_path, enhanced_face)
                
                count += 1
                faces_in_image += 1

    print(f"[INFO] Extracted {count} faces from {len(image_files)} images")

# --- Step 2: Simple embedding extraction ---
def get_embeddings():
    embeddings = []
    face_files = sorted([f for f in os.listdir(FACE_DIR) if f.endswith('.jpg')])
    
    print(f"[INFO] Processing {len(face_files)} face images")
    
    for fname in tqdm(face_files, desc="Getting embeddings"):
        face_path = os.path.join(FACE_DIR, fname)
        try:
            # Use enforce_detection=False for blurred faces
            result = DeepFace.represent(
                img_path=face_path, 
                model_name=EMBEDDING_MODEL, 
                enforce_detection=False
            )
            embedding = np.array(result[0]["embedding"])
            embeddings.append((face_path, embedding))
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {str(e)}")
    
    print(f"[INFO] Successfully processed {len(embeddings)} faces")
    return embeddings

# --- Simple clustering with quality filtering ---
def cluster_faces(embeddings):
    if len(embeddings) < 2:
        print("[ERROR] Need at least 2 faces to cluster")
        return
    
    # Clear existing clusters
    if os.path.exists(CLUSTERED_DIR):
        shutil.rmtree(CLUSTERED_DIR)
    os.makedirs(CLUSTERED_DIR, exist_ok=True)
    
    # Extract embeddings and normalize them
    from sklearn.preprocessing import normalize
    vectors = np.array([emb for _, emb in embeddings])
    vectors = normalize(vectors, axis=1)  # Normalize for better comparison
    
    # Calculate quality scores for each face
    quality_scores = []
    for face_path, _ in embeddings:
        face = cv2.imread(face_path)
        if face is not None:
            # Calculate sharpness (higher = sharper)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_scores.append(sharpness)
        else:
            quality_scores.append(0)
    
    # Try multiple eps values and choose the best one
    best_eps = 0.4
    best_n_clusters = 0
    
    for eps in [0.3, 0.4, 0.5, 0.6]:
        clustering = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit(vectors)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        print(f"[INFO] eps={eps}: {n_clusters} clusters")
        
        # Prefer more clusters for better separation
        if n_clusters > best_n_clusters and n_clusters < len(embeddings) * 0.8:
            best_n_clusters = n_clusters
            best_eps = eps
    
    print(f"[INFO] Using eps={best_eps} for final clustering")
    
    # Final clustering with best parameters
    clustering = DBSCAN(eps=best_eps, min_samples=1, metric='cosine').fit(vectors)
    labels = clustering.labels_
    
    # Group faces by cluster and select best representative
    clusters = {}
    for idx, (face_path, _) in enumerate(embeddings):
        label = labels[idx]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((face_path, quality_scores[idx]))
    
    # Save clustered faces and show best representative
    for cluster_id, face_data in clusters.items():
        if cluster_id == -1:
            cluster_name = "unclustered"
        else:
            cluster_name = f"person_{cluster_id:02d}"
        
        cluster_dir = os.path.join(CLUSTERED_DIR, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Sort by quality (best first)
        face_data.sort(key=lambda x: x[1], reverse=True)
        
        for i, (face_path, quality) in enumerate(face_data):
            filename = os.path.basename(face_path)
            # Mark the best quality face
            if i == 0:
                new_filename = f"BEST_{filename}"
            else:
                new_filename = filename
            shutil.copy(face_path, os.path.join(cluster_dir, new_filename))
    
    # Print detailed results
    valid_clusters = [k for k in clusters.keys() if k != -1]
    print(f"[INFO] Clustering Results:")
    print(f"  - Found {len(valid_clusters)} person clusters")
    print(f"  - Unclustered faces: {len(clusters.get(-1, []))}")
    
    for cluster_id in valid_clusters:
        cluster_faces = clusters[cluster_id]
        avg_quality = np.mean([q for _, q in cluster_faces])
        print(f"  - Person {cluster_id:02d}: {len(cluster_faces)} faces (avg quality: {avg_quality:.1f})")
        
        # Show file names for manual verification
        if len(cluster_faces) > 5:  # Only show for large clusters
            print(f"    Files: {[os.path.basename(f) for f, _ in cluster_faces[:3]]}...")
        else:
            print(f"    Files: {[os.path.basename(f) for f, _ in cluster_faces]}")
    
    # Additional check for overclustering
    large_clusters = [cid for cid in valid_clusters if len(clusters[cid]) > 10]
    if large_clusters:
        print(f"[WARNING] Large clusters detected: {large_clusters}")
        print("This might indicate overclustering. Consider:")
        print("1. Lowering eps value (more strict)")
        print("2. Improving image quality")
        print("3. Manual verification of results")

# --- Main execution ---
if __name__ == "__main__":
    print("=== FACE CLUSTERING PIPELINE ===")
    
    # Check if required directories exist
    if not os.path.exists(ALBUM_DIR):
        print(f"[ERROR] Album directory '{ALBUM_DIR}' not found!")
        exit(1)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        print("[ERROR] DNN model files not found!")
        print("Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")
        exit(1)
    
    print("[1/3] Extracting faces...")
    extract_faces()
    
    print("[2/3] Getting face embeddings...")
    embeddings = get_embeddings()
    
    if len(embeddings) == 0:
        print("[ERROR] No face embeddings generated!")
        exit(1)
    
    print("[3/3] Clustering faces...")
    cluster_faces(embeddings)
    
    print("=== DONE ===")
    print(f"Check results in '{CLUSTERED_DIR}' directory")