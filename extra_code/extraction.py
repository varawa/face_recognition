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
EMBEDDING_MODEL = 'ArcFace'  # You can try 'Facenet512' if needed

# --- Step 1: Detect and crop all faces from all images ---
def extract_faces():
    net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
    os.makedirs(FACE_DIR, exist_ok=True)
    count = 0

    for filename in os.listdir(ALBUM_DIR):
        image_path = os.path.join(ALBUM_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_path = os.path.join(FACE_DIR, f'face_{count}.jpg')
                cv2.imwrite(face_path, face)
                count += 1

    print(f"[INFO] Total faces extracted: {count}")

# --- Step 2: Get embeddings for each face ---
def get_embeddings():
    embeddings = []
    face_files = sorted(os.listdir(FACE_DIR))
    for fname in tqdm(face_files, desc="Extracting embeddings"):
        path = os.path.join(FACE_DIR, fname)
        try:
            embedding = DeepFace.represent(img_path=path, model_name=EMBEDDING_MODEL, enforce_detection=False)[0]["embedding"]
            embeddings.append((path, np.array(embedding)))
        except Exception as e:
            print(f"[WARN] Failed to embed {fname}: {e}")
    return embeddings

# --- Step 3: Cluster faces using DBSCAN ---
def cluster_faces(embeddings, eps=0.5, min_samples=1):
    if not embeddings:
        print("[ERROR] No embeddings found.")
        return
    vectors = np.array([emb for _, emb in embeddings])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(vectors)

    os.makedirs(CLUSTERED_DIR, exist_ok=True)
    labels = clustering.labels_
    for idx, (path, _) in enumerate(embeddings):
        label = labels[idx]
        folder = os.path.join(CLUSTERED_DIR, f"person_{label}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(path, os.path.join(folder, os.path.basename(path)))

    print(f"[INFO] Clustering complete: {len(set(labels))} groups")

# --- Main Pipeline ---
if __name__ == "__main__":
    extract_faces()
    embeddings = get_embeddings()
    cluster_faces(embeddings)