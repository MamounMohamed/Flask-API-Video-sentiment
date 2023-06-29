from flask import Flask, request, render_template ,jsonify
from keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
models = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotions_cnt = np.zeros(7)

@app.route('/')

def home():
       return render_template("video_uploader.html")              

@app.route('/upload_model', methods=["POST"])
def upload_model():
    try:
        model = request.files['model']
        model_path = os.path.join('models', model.filename)
        model.save(model_path)
        models.append(model.filename)
        return "model is uploded" 
    except:
        return "model couldn't be updated"
    return "model couldn't be updated" 

    
@app.route('/upload_video', methods=['POST'])

def upload_video():
    # Get the uploaded video file
    model_name = "Emotion.h5"
    model_path = os.path.join('models',model_name)
    model = load_model(model_path)
    video = request.files['video']
    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)
    
    
    # Load the video and run emotion detection on each frame
    
    
    cap = cv2.VideoCapture(video_path)
    results =[]
    while cap.isOpened():
          
        # resize image
        ret, frame = cap.read()
        if not ret:
            continue
               
        width = 912
        height = 512
        dim = (width, height)
        
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 1:
            continue 
          
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = face_roi.reshape(1, 48, 48, 1)
            predictions = model.predict(face_roi)
            emotion_index = predictions.argmax()
            emotions_cnt[emotion_index]+= predictions[0][emotion_index]

    
    cap.release()
    sum = np.sum(emotions_cnt)
    results = dict([(emotions[i] , emotions_cnt[i]/sum) for i in range(0,7)])
   
    
    return jsonify(results)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0')
