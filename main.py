import os
import csv
import cv2
import cv2.face
import datetime
import time
import pandas as pd
import numpy as np
from PIL import Image
from threading import Thread
import tensorflow as tf
from tensorflow.keras.models import load_model




# Load the pre-trained emotion and gender models
emotion_model = load_model('C:\Users\illiMercenary\Desktop\Project22/model_v6_23.hdf5')
gender_model = load_model('C:\Users\illiMercenary\Desktop\Project22/model.h5')

# Define the list of gender labels
#genders = ['Male', 'Female']
genders = ['Male', 'Female']
# Define the list of emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to predict emotion and gender
def predict_emotion(face_roi):
    face_roi_resized = cv2.resize(face_roi, (48, 48))
    
    face_roi_normalized = face_roi_resized / 255.0
    face_roi_final = np.expand_dims(face_roi_normalized, axis=0)

    emotion_prediction = emotion_model.predict(face_roi_final)
    emotion_label = np.argmax(emotion_prediction)

    return emotions[emotion_label]


# Function to recognize gender based on face proportions
def recognize_gender(face_roi):
    # Calculate the aspect ratio of the face (width / height)
    aspect_ratio = face_roi.shape[1] / face_roi.shape[0]

    # Define a threshold for the aspect ratio to classify gender
    gender_threshold = 1.0  # Adjust as needed

    if aspect_ratio >= gender_threshold:
        return 'Male'
    else:
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        face_roi_normalized = face_roi_resized / 255.0
        face_roi_final = np.expand_dims(face_roi_normalized, axis=0)

        gender_prediction = gender_model.predict(face_roi_final)
        gender_label = np.argmax(gender_prediction)

        return genders[gender_label]
    
# -------------- Capture_Image code --------------

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# Function to take images and save with person information
def takeImages():
    code = input("Enter Your Code: ")
    name = input("Enter Your Name: ")

    if is_number(code) and name.isalpha():
        person_dir = os.path.join("TrainingImage", f"{name}_{code}")

        if not os.path.exists(person_dir):
            os.makedirs(person_dir, exist_ok=True)
        else:
            print("Directory already exists. Skipping creation.")

        cam = cv2.VideoCapture(0)
        harcascadePath = 'C:\Users\illiMercenary\Desktop\Project22/haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
                sampleNum = sampleNum + 1

                img_name = f"{name}_{code}_{sampleNum}.jpg"
                img_path = os.path.join(person_dir, img_name)
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = f"Images Saved for code: {code}, Name: {name}"
        row = [code, name]
        with open("StudentDetails"+os.sep+"StudentDetails.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
    else:
        if is_number(code):
            print("Enter Alphabetical Name")
        if name.isalpha():
            print("Enter Numeric code")

# -------------- check_camera code --------------

def camer():
    window_width = 640
    window_height = 360

    face_cascade = cv2.CascadeClassifier('C:\Users\illiMercenary\Desktop\Project22/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)

        cv2.imshow('Webcam Check', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------- Recognize code --------------

def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8)
    recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
    harcascadePath = 'C:\Users\illiMercenary\Desktop\Project22/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['code', 'Name', 'Date', 'Time']

    window_width = 480
    window_height = 360

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, window_width)
    cam.set(4, window_height)
    minW = 0.1 * window_width
    minH = 0.1 * window_height
    
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            code, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:
                aa = df.loc[df['code'] == code]['Name'].values
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(code) + "-" + aa
            else:
                code = '  Unknown  '
                tt = str(code)
                confstr = "  {0}%".format(round(100 - conf))

            if (100-conf) > 67:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = str(aa)[2:-2]

            tt = str(tt)[2:-2]
            if (100-conf) > 67:
                tt = tt + " [Identified]"
                cv2.putText(im, str(tt), (x+5, y-5), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100-conf) > 62:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            elif (100-conf) > 50:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

            # Call the prediction functions
     emotion_label = predict_emotion(gray[y:y+h, x:x+w])
    gender_label = recognize_gender(gray[y:y+h, x:x+w])

    emotion_text = f'Emotion: {emotion_label}'
    cv2.putText(im, emotion_text, (x, y+h+60), font, 0.8, (0, 255, 0), 2)

    gender_text = f'Gender: {gender_label}'
    cv2.putText(im, gender_text, (x, y+h+120), font, 0.8, (0, 255, 0), 2)
        cv2.imshow('code', im)

        if cv2.waitKey(1) == ord('q'):
            break

    print("Done!")
    cam.release()
    cv2.destroyAllWindows()

# -------------- Train_Image code --------------

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    codes = []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
        except (PermissionError, FileNotFoundError) as e:
            print(f"Error opening image at path {imagePath}: {e}")
            continue

        imageNp = np.array(pilImage, 'uint8')
        code = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        codes.append(code)
    return faces, codes

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = 'C:\Users\illiMercenary\Desktop\Project22/haarcascade_frontalface_default.xml'

    detector = cv2.CascadeClassifier(harcascadePath)
    faces, codes = getImagesAndLabels("TrainingImage")

    if faces and codes:
        recognizer.train(faces, np.array(codes))
        Thread(target=counter_img("TrainingImage")).start()
        recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
        print("All Images Trained")
    else:
        print("Not enough training data.")

def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

# -------------- main code --------------

def title_bar():
    os.system('cls')
    print("Face Recognition System")

def mainMenu():
    title_bar()
    print()
    print("1 Face Detection")
    print("2 Capture Faces")
    print("3 Train Images")
    print("4 Recognize")
    print("5 Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                checkCamera()
                break
            elif choice == 2:
                CaptureFaces()
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                RecognizeFaces()
                break
            elif choice == 5:
                print("End!!")
                break
                mainMenu()
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
    exit

def checkCamera():
    camer()
    key = input("Enter any key to return to the main menu")
    mainMenu()

def CaptureFaces():
    takeImages()
    key = input("Enter any key to return to the main menu")
    mainMenu()

def Trainimages():
    TrainImages()
    key = input("Enter any key to return to the main menu")
    mainMenu()

def RecognizeFaces():
    recognize_attendance()
    key = input("Enter any key to return to the main menu")
    mainMenu()

mainMenu()
