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
emotion_model = load_model('C:\Users\illiMercenary\Desktop\Project22\\model_v6_23.hdf5')
#gender_model = load_model('C:\Users\illiMercenary\Desktop\Project22\\model.h5')

# Define the list of gender labels
#genders = ['Male', 'Female']
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
        return 'Female'
    


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

# Take image function
def takeImages():
    code = input("Enter Your Code: ")
    name = input("Enter Your Name: ")

    if is_number(code) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = 'C:\Users\illiMercenary\Desktop\Project22\\haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage" + os.sep + name + "." + code + '.' +
                            str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)

            # wait for 100 milliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            

            # break if the sample number is more than 100
            if sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for code : " + code + " Name : " + name
        row = [code, name]
        with open("StudentDetails" + os.sep + "StudentDetails.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
    else:
        if is_number(code):
            print("Enter Alphabetical Name")
        if name.isalpha():
            print("Enter Numeric code")
    mainMenu()


# -------------- check_camera code --------------

def camer():
    import cv2

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('C:\Users\illiMercenary\Desktop\Project22\\haarcascade_frontalface_default.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10,159,255), 2)


        # Display
        cv2.imshow('Webcam Check', img)

        # Stop if escape key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()



# -------------- Recognize code --------------

def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel" + os.sep + "Trainner.yml")
    harcascadePath = 'C:\Users\illiMercenary\Desktop\Project22\\haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails" + os.sep + "StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['code', 'Name', 'Date', 'Time']

    window_width = 640
    window_height = 480

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, window_width)
    cam.set(4, window_height)
    minW = 0.1 * window_width
    minH = 0.1 * window_height

    # Load face, age, and gender detection models
    faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)
            code, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 100:
                aa = df.loc[df['code'] == code]['Name'].values[0]  # Extract the actual name from the values array
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(code) + "-" + str(aa)
            else:
                code = '  Unknown  '
                tt = str(code)
                confstr = "  {0}%".format(round(100 - conf))

            if (100 - conf) > 67:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = str(aa)
                tt = str(tt)
                tt = tt + " [Identified]"
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100 - conf) > 62:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            elif (100 - conf) > 50:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

            # Extract the face region
            face = im[y:y + h, x:x + w]

            # Preprocess the face for age and gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            # Predict age
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            # Display gender and age labels
            gender_text = f'Gender: {gender}'
            age_text = f'Age: {age}'
            cv2.putText(im, gender_text, (x, y + h + 60), font, 0.8, (0, 255, 0), 2)
            cv2.putText(im, age_text, (x, y + h + 120), font, 0.8, (0, 255, 0), 2)

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
    harcascadePath = 'C:\Users\illiMercenary\Desktop\Project22\\haarcascade_frontalface_default.xml'

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
                mainMenu()
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                recognize_attendance()
                break
            elif choice == 5:
                print("End!!")
                break
                mainMenu()
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4 Try Again")
    exit

def checkCamera():
    camer()
    key = input("Enter any key to return to the main menu")
    mainMenu()

def CaptureFaces():
    takeImages()
    camer()  # Add this line to open the camera after capturing images

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
