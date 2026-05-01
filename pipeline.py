from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── 1. Load models ──────────────────────────────────────────────
sbert = SentenceTransformer('all-MiniLM-L6-v2')
yolo = YOLO('yolov5n.pt')

# ── 2. COCO class names ─────────────────────────────────────────
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ── 3. Load or compute class embeddings ─────────────────────────
try:
    class_embeddings = np.load('class_embeddings.npy')
    print("Loaded cached class embeddings")
except FileNotFoundError:
    class_embeddings = sbert.encode(coco_classes)
    np.save('class_embeddings.npy', class_embeddings)
    print("Computed and saved class embeddings")

# ── 4. Define your task ─────────────────────────────────────────
task = "pick up a drink"

# ── 5. Embed the task and compute similarity to all 80 classes ──
task_embedding = sbert.encode(task)
sims = cosine_similarity([task_embedding], class_embeddings)[0]

# ── 6. Filter to relevant classes above threshold ───────────────
THRESHOLD = 0.25
relevant_indices = np.where(sims > THRESHOLD)[0]
relevant_classes = [coco_classes[i] for i in relevant_indices]

print(f"\nTask: '{task}'")
print(f"Relevant classes (similarity > {THRESHOLD}): {relevant_classes}")

# ── 7. Run YOLO with only relevant classes ──────────────────────
image_path = 'image.jpg'
results = yolo(image_path, classes=relevant_indices.tolist())

# ── 8. Compute combined score for each detection ────────────────
print("\nDetections and scores:")
best_score = -1
best_object = None

for box in results[0].boxes:
    class_id = int(box.cls)
    label = coco_classes[class_id]
    confidence = float(box.conf)
    cos_sim = sims[class_id]
    combined_score = confidence * cos_sim

    print(f"  {label}: conf={confidence:.2f}, sim={cos_sim:.3f}, combined={combined_score:.3f}")

    if combined_score > best_score:
        best_score = combined_score
        best_object = {
            "label": label,
            "confidence": confidence,
            "similarity": cos_sim,
            "combined_score": combined_score,
            "bbox": box.xyxy.tolist()
        }

# ── 9. Print the winner ─────────────────────────────────────────
print(f"\n✅ Best object for task '{task}': {best_object['label']}")
print(f"   Confidence:    {best_object['confidence']:.2f}")
print(f"   Similarity:    {best_object['similarity']:.3f}")
print(f"   Combined Score:{best_object['combined_score']:.3f}")
print(f"   Bounding Box:  {best_object['bbox']}")

# ── 10. Show result ─────────────────────────────────────────────
results[0].show()