# app.py
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image

app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet("object_counting_chatbot\yolov3.weights", "object_counting_chatbot\yolov3.cfg")
with open("object_counting_chatbot\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/count', methods=['POST'])
def count_objects():
    # Check if image and object name are provided
    if 'image' not in request.files or 'object' not in request.form:
        return render_template('index.html', error="Please provide an image and an object name.")

    object_name = request.form['object'].lower()
    image_file = request.files['image']

    try:
        # Load image with PIL and convert to numpy array
        image = Image.open(image_file)
        image_np = np.array(image.convert('RGB'))
        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
          return render_template('index.html', error=f"Error processing the image: {e}")
    # Detect objects in the image
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    try:
        detections = net.forward(output_layers)
    except cv2.error as e:
         return render_template('index.html', error=f"Error during object detection: {e}")

    # Count specified objects
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust the confidence threshold if needed
                # Get bounding box coordinates
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to filter out redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Count specified objects after applying NMS
    count = 0
    for i in indices.flatten():  # Flatten to handle list format returned by NMSBoxes
        if classes[class_ids[i]] == object_name:
            count += 1

    return render_template('result.html', object_name=object_name, count=count)
if __name__ == "__main__":
    app.run(debug=True)
   
 