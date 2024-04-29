import cv2  
import math
import argparse
import random

def preprocess_image(image):
    # Resize image to a fixed size
    resized_image = cv2.resize(image, (227, 227))
    
    # Convert image to float32 and normalize pixel values
    normalized_image = resized_image.astype('float32') / 255.0
    
    # Mean subtraction (optional)
    # normalized_image -= MEAN_PIXEL_VALUES
    
    return normalized_image

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def predict_emotion(face):
    # Implement your logic to predict emotion here
    # For demonstration, let's return a random emotion
    emotions = ["Happy", "Sad", "Angry", "Neutral", "Surprised"]
    return random.choice(emotions)

def overlayEmoji(frame, faceBox, emotion):
    emoji_dict = {
        "Happy": "😊",
        "Sad": "😢",
        "Angry": "😠",
        "Neutral": "😐",
        "Surprised": "😮",
        # Add more emotions and corresponding emojis as needed
    }
    x1, y1, x2, y2 = faceBox
    emoji = emoji_dict.get(emotion, "❓")  # Default emoji for unknown emotion
    text = f'Emotion: {emotion}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y2 + text_size[1] + 5
    cv2.putText(frame, text, (x1, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.putText(frame, emoji, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Load pre-trained models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Constants for age and gender prediction
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize video capture
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    # Highlight faces in the frame
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    # Process each detected face
    for faceBox in faceBoxes:
        # Extract face region
        face = frame[max(0, faceBox[1]-padding): min(faceBox[3]+padding, frame.shape[0]-1),
                     max(0, faceBox[0]-padding): min(faceBox[2]+padding, frame.shape[1]-1)]

        # Predict gender and age
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Predict emotion
        emotion = predict_emotion(face)
        
        # Overlay emotion on the frame
        overlayEmoji(resultImg, faceBox, emotion)

        # Display gender, age, and emotion
        cv2.putText(resultImg, f'Gender: {gender}, Age: {age}, Emotion: {emotion}',
                    (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Show the frame with detections
        cv2.imshow("Detecting age, gender, and emotion", resultImg)
