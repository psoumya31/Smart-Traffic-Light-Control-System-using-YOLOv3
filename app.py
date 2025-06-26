from flask import Flask, render_template, jsonify, request, Response
import threading
import cv2

app = Flask(_name_)

# Shared data for pedestrian count, vehicle count, and red light duration
shared_data = {
    "pedestrian_count": 0,
    "vehicle_count": 0,
    "red_light_duration": 10
}

def generate_video():
    # Open the video file
    cap = cv2.VideoCapture('videoo.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        # Yield the frame in the format needed for video streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def data():
    return jsonify(shared_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/increase_duration', methods=['POST'])
def increase_duration():
    shared_data["red_light_duration"] += 10
    return jsonify(success=True)

def run_flask():
    app.run(debug=True, use_reloader=False)

# Start Flask server in a separate thread
if _name_ == "_main_":
    threading.Thread(target=run_flask).start()