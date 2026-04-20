from sentence_transformers import SentenceTransformer
import numpy as np

# Load the SBERT model
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# All 80 COCO class names — exactly as YOLOv5 knows them
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

# Embed all 80 class names
class_embeddings = sbert.encode(coco_classes)

print(f"Class embeddings shape: {class_embeddings.shape}")  # should be (80, 384)

# Save them so we don't recompute every time
np.save('class_embeddings.npy', class_embeddings)
print("Saved class_embeddings.npy")

# Quick sanity check — embed a task and print similarity to first 5 classes
from sklearn.metrics.pairwise import cosine_similarity

task = "pick up a drink"
task_embedding = sbert.encode(task)

sims = cosine_similarity([task_embedding], class_embeddings)[0]

# Print top 5 most similar classes
top5 = np.argsort(sims)[::-1][:5]
print(f"\nTask: '{task}'")
print("Top 5 most relevant classes:")
for i in top5:
    print(f"  {coco_classes[i]}: {sims[i]:.3f}")