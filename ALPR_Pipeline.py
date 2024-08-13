from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time
import numpy as np
import re

ncnn_model = YOLO('/Saved Models/yolov8n_ncnn_model', task='detect')
video_path = 'IMG_0914.MOV'
device = 'cpu'
model = 'microsoft/trocr-base-printed'
processor = TrOCRProcessor.from_pretrained(model)
model = VisionEncoderDecoderModel.from_pretrained(model).to(device)
ncnn_model.to(device)

import cv2
import time
import numpy as np
from PIL import Image
import re

# Initialize video capture
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Error: Could not access video")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_resized.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_height, original_width = frame.shape[:2]

    if frame_count % 3 == 0:
        resized_img = cv2.resize(frame, (640, 640))

        results = ncnn_model(resized_img)
        class_names = results[0].names
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls

        class_colors = {
            '0': (255, 0, 0),
            '1': (0, 0, 255)
        }

        # Scaling factors
        x_scale = original_width / 640.0
        y_scale = original_height / 640.0

        # Iterate through the boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]  # Extract coordinates
            conf = results[0].boxes.conf[i].item()  # Extract confidence score
            label = int(results[0].boxes.cls[i].item())  # Extract label (if available)

            # Scale coordinates back to the original image size
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            if label == 1:
                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue  # Skip empty crops
                image = Image.fromarray(np.uint8(cropped_img)).convert('RGB')
                pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Remove all special characters
                txt = re.sub(r'\W+', '', generated_text).upper()

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[str(label)], 2)
                msg = f"{class_names[int(label)]}: {txt}"
                cv2.putText(frame, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors[str(label)], thickness=2)
            else:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[str(label)], 2)
                msg = f"{class_names[int(label)]}"
                cv2.putText(frame, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors[str(label)], thickness=2)

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f'FPS:{fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=2)

    # Uncomment the line below if you want to display the frames
    if frame_count % 20 == 0:
      cv2_imshow(frame)

    out.write(frame)

    # If you want to view the frames, add a break condition for quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
