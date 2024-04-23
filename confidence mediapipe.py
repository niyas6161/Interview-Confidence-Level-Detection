import cv2
import tensorflow as tf
import mediapipe as mp

pose = 0

emotion_model = tf.keras.models.load_model('/home/niyas/jan_dl/intenship/emotion detection/EMOTION.h5')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
pose_detection = mp_pose.Pose()

cap = cv2.VideoCapture(0)

def detect_eye_contact(face_roi):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')



    while True:

        ret, frame = cap.read()


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
     
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

   
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
         
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            if len(eyes) >= 2:
               
                return 1

def is_head_centered(landmarks, image_width):
    nose_tip = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_eye = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

 
    nose_horizontal_position = (left_eye.x + right_eye.x) / 2

   
    centered_threshold = 0.1 

    
    is_centered = abs(nose_horizontal_position - 0.5) < centered_threshold

 
    return int(is_centered)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
   
        face_roi = gray[y:y + h, x:x + w]

   
        results = face_detection.process(frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

            
            
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

                
                face_landmarks = pose_detection.process(frame).pose_landmarks
                if face_landmarks:
                   
                    pose = is_head_centered(face_landmarks, iw)

      
        face_input = cv2.resize(face_roi, (48, 48))
        emotion_prediction = emotion_model.predict(face_input.reshape(1, 48, 48, 1))
        emotion_class_index = tf.argmax(emotion_prediction, axis=-1)

        eye_contact_result = detect_eye_contact(face_roi)

        if emotion_class_index == 0:
            emotion_label = 'angry'
        elif emotion_class_index == 1:
            emotion_label = 'disgusted'
        elif emotion_class_index == 2:
            emotion_label = 'fearful'
        elif emotion_class_index == 3:
            emotion_label = 'happy'
        elif emotion_class_index == 4:
            emotion_label = 'neutral'
        elif emotion_class_index == 5:
            emotion_label = 'sad'
        elif emotion_class_index == 6:
            emotion_label = 'surprised'

     
        if emotion_label == 'happy' and eye_contact_result == 1 and pose == 1:
            cv2.putText(frame, 'Confidence : High', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif emotion_label == 'neutral' and eye_contact_result == 1 and pose == 1:
            cv2.putText(frame, 'Confidence : Moderate', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'Confidence : Low', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()