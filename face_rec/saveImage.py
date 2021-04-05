import face_recognition
import os #to iterate over directories
import cv2
from datetime import datetime
from write_json import write_json


def saveImage(image, name, check_action):
    filepath = "saved/" #where we want to save the images
    currentTime = datetime.now()
    date = currentTime.strftime("%Y-%m-%d")
    time = currentTime.strftime("%H:%M")
    fn_time = currentTime.strftime("%Y-%m-%d %H.%M") #excludes seconds to avoid duplication
    fn_name = name + '_' + fn_time + '.jpg' #includes name and time in filename
    filename = filepath + fn_name #full filename of the image

    data = {
        'name': name,
        'date': date,
        'time': time,
        'image_filename': fn_name,
        'action': check_action
        }
    #save the frame where desired visitor has been located
    #saves image to the 'saved' folder
    isSaved = cv2.imwrite(filename, image)

    #after successfully saving the image
    #then save the information to a json file
    if isSaved:
        print("Visitor has been saved.")
        write_json(data)
    else:
        print("Error with saving visitors data.")
