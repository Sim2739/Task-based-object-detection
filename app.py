import gradio as gr
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

# ── 1. Load models once at startup ──────────────────────────────
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
except FileNotFoundError:
    class_embeddings = sbert.encode(coco_classes)
    np.save('class_embeddings.npy', class_embeddings)

THRESHOLD = 0.20

# ── 4. Core detection function ───────────────────────────────────
def detect(image, task):

    # Embed task and compute similarity
    task_embedding = sbert.encode(task)
    sims = cosine_similarity([task_embedding], class_embeddings)[0]

    # Filter relevant classes
    relevant_indices = np.where(sims > THRESHOLD)[0]
    if len(relevant_indices) == 0:
        relevant_indices = np.argsort(sims)[::-1][:3]

    relevant_classes = [coco_classes[i] for i in relevant_indices]

    # Run YOLO
    results = yolo(image, classes=relevant_indices.tolist(), verbose=False)

    # Score detections
    best_score = -1
    best_object = None

    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        cos_sim = sims[class_id]
        combined_score = confidence * cos_sim

        if combined_score > best_score:
            best_score = combined_score
            best_object = {
                "label": coco_classes[class_id],
                "confidence": confidence,
                "similarity": cos_sim,
                "combined_score": combined_score,
                "bbox": box.xyxy[0].tolist()
            }

    # Draw only the best object's bbox on the image
    output_image = np.array(image)
    summary = ""

    if best_object:
        x1, y1, x2, y2 = [int(c) for c in best_object["bbox"]]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label_text = f"{best_object['label']} ({best_object['combined_score']:.2f})"
        cv2.putText(output_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        summary = f"""
### ✅ Best Object: {best_object['label']}

| Metric | Value |
|---|---|
| Detection Confidence | {best_object['confidence']:.2f} |
| Semantic Similarity  | {best_object['similarity']:.3f} |
| Combined Score       | {best_object['combined_score']:.3f} |

**Relevant classes considered:** {', '.join(relevant_classes)}
        """
    else:
        summary = "### ❌ No relevant object detected in this image for the given task."

    return output_image, summary

# ── 5. Build the Gradio UI ───────────────────────────────────────
tasks_list = ["get lemon out of tea",
    "dig hole", "open bottle of beer", "open parcel", "serve wine",
    "pour sugar", "smear butter", "extinguish fire", "pound carpet"
]

with gr.Blocks(title="Task-Aware Object Detection") as app:
    gr.Markdown("# 🎯 Task-Aware Object Detection")

    gr.Markdown("Upload an image, select a task, and the system will find the most relevant object.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Image")
            task_input = gr.Dropdown(choices=tasks_list, label="Select Task", value=tasks_list[0])
            run_btn = gr.Button("Detect", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="Result")
            summary_output = gr.Markdown()

    run_btn.click(fn=detect, inputs=[image_input, task_input], outputs=[image_output, summary_output])

app.launch()