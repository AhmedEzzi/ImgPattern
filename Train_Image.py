import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread



# -------------- image labesl ------------------------

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    codes = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        code = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        codes.append(code)
    return faces, codes


# ----------- train images function ---------------
def TrainImages():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    harcascadePath = 'C:\\Users\\User\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, code = getImagesAndLabels("TrainingImage")
    Thread(target = recognizer.train(faces, np.array(code))).start()
    # Below line is optional for a visual counter effect
    Thread(target = counter_img("TrainingImage")).start()
    recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
    print("All Images")

# Optional, adds a counter for images trained (You can remove it)
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

