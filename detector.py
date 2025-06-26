import cv2
import numpy as np
import requests

vehicle_model = cv2.dnn.readNet('./models/yolov3.weights', './models/yolov3.cfg')

layer_names = vehicle_model.getLayerNames()
output_layers = [layer_names[i - 1] for i in vehicle_model.getUnconnectedOutLayers()]

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture('sample.mp4')

def calculate_red_signal_duration(pedestrian_count, vehicle_count,
                                  base_duration=10, adjustment_pedestrian=0.5, adjustment_vehicle=0.3,
                                  min_duration=5, max_duration=60,
                                  pedestrian_threshold=5, vehicle_threshold=5,
                                  pedestrian_weight=1.2, vehicle_weight=1.1):
    if pedestrian_count == 0:
        return 0  
    
    if pedestrian_count > pedestrian_threshold:
        T_P = pedestrian_count * adjustment_pedestrian * pedestrian_weight
    else:
        T_P = pedestrian_count * adjustment_pedestrian
    
    if vehicle_count > vehicle_threshold:
        T_V = vehicle_count * adjustment_vehicle * vehicle_weight
    else:
        T_V = vehicle_count * adjustment_vehicle
    
    T_red = base_duration + T_P + T_V
    T_red = max(min_duration, min(T_red, max_duration))
    
    return T_red

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 2 == 0:
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        vehicle_model.setInput(blob)
        detections = vehicle_model.forward(output_layers)
        
        pedestrian_count = 0
        vehicle_count = 0
        detected_pedestrians = []
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                class_id = class_ids[i]
                label = str(classes[class_id])
                
                if label == 'person':
                    center_x = x + w // 2
                    center_y = y + h // 2
                    if not any((abs(center_x - px) < w / 2 and abs(center_y - py) < h / 2) for px, py in detected_pedestrians):
                        pedestrian_count += 1
                        detected_pedestrians.append((center_x, center_y))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                elif label in ['car', 'truck', 'bus', 'motorbike']:
                    vehicle_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        total_red_time = calculate_red_signal_duration(pedestrian_count, vehicle_count)
        
        data = {
            "pedestrian_count": pedestrian_count,
            "vehicle_count": vehicle_count,
            "red_light_duration": total_red_time
        }
        cv2.putText(frame, f'Pedestrians: {pedestrian_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Red Light Time: {total_red_time:.1f} sec', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if total_red_time == 0:
            cv2.circle(frame, (width - 50, 50), 20, (0, 255, 0), -1)  
        else:
            cv2.circle(frame, (width - 50, 50), 20, (0, 0, 255), -1) 
        
        cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()