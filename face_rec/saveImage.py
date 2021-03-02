import face_recognition
import os #to iterate over directories
import cv2
from datetime import datetime

def saveImage(image, name):
    filepath = "saved/" #where we want to save the images
    currentTime = datetime.now()
    fn_time = currentTime.strftime("%Y-%m-%d %H.%M.%S") #includes time in filename
    fn_name = name + '_' + fn_time + '.jpg' #includes name and time in filename
    filename = filepath + fn_name #full filename of the image

    isSaved = cv2.imwrite(filename, image) #save the frame where desired visitor has been located
    if isSaved:
        print("Visitor has been saved.")
