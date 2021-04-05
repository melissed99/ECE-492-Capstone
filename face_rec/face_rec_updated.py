import cv2
import math
from sklearn import neighbors # Used for KNN algorithm
import os
import pickle
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
from saveImage import saveImage
#from signal import signal
from pygame import mixer
from time import sleep, time
from picamera import PiCamera

KNOWN_FACES_DIR = "known_faces"
video = cv2.VideoCapture(0)
# Training the KNN classifer
def train(KNOWN_FACES_DIR, model_save_path=None, n_neighbors=None, knn_algo='ball_tree'):

    known_faces = []
    known_names = []

    # Iterate over known faces and store info
    # Each subfolder of known_faces become the label (name) for found faces
    for name in os.listdir(KNOWN_FACES_DIR):
        if not os.path.isdir(os.path.join(KNOWN_FACES_DIR, name)):
            continue

        # Loop through each training image for the current person in the subfolder
        for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
            if not filename.startswith('._'):
                #print(filename)
                 image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name,filename))
                 face_bounding_boxes = face_recognition.face_locations(image)

            # Add face encoding and name for current image to the training set
                 known_faces.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                 known_names.append(name)

    # Determine how many neighbors (how many K neighbors) to use for weighting in the KNN classifier
    # Can formalize "nearest" as euclidean distance
    # Ideal K is sqrt(N) where N is number of samples, in this case number of known faces
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(known_faces))))

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(known_faces, known_names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

# Comparing faces from video feed to known faces in training set
# c_frame is the face obtained from camera feed
def predict(X_frame, knn_clf=None, model_path=None, distance_threshold = 0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    found_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(found_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=found_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    are_matches = [min(closest_distances[0][i][0], closest_distances[0][i][1]) <= distance_threshold for i in range(len(found_face_locations))]
    return [faces_encodings, found_face_locations, are_matches, knn_clf]


# function to check if the visitor is desired or not desired
def evaluate_visitor(c_frame, faces_encodings, found_face_locations, are_matches, knn_clf):
    # Predict classes and remove classifications that aren't within the threshold
    predictions = []
    for pred, loc, rec in zip(knn_clf.predict(faces_encodings), found_face_locations, are_matches):
        if rec: # if recognized, then save image, no action
            print("Face detected")
            predictions.append((pred,loc))
            saveImage(c_frame, pred, "No action.")
        else: # if not recognizes, play warning, then trigger if needed
            print("Undesired visitor detected")
            predictions.append(("unknown",loc))
            ##### plays audible warning ######
            mixer.init()
            warning = mixer.Sound("sound.ogg")
            warning.play(loops=100)  # sound is 3 secs long so looping for more duration/2 to give headroom
            sleep(10)
            mixer.quit()
            #### check logic function# #####
            seconds = 5    # within 5 seconds, it will check if visitor stayed
            count = 0  # used to check if within those 10 secs, a similar visitor is found
            while seconds >0:
                sleep(1 - time() % 1)
                seconds -=1
                check = unknown_stayed(knn_clf) # check if 0 or 1 => 0 means person left, 1 means person stayed
                print(check)
                if check > 0:
                    count += 1
            if count == 0:
                saveImage(c_frame, "unknown", "Action: Person left. Nerf gun is not triggered.")
                print("Undesired visitor left, no action.")
                
            else: #within those 10 secs, the visitor stayed
                saveImage(c_frame, "unknown", "Action: Person did not leave. Nerf gun is triggered.")
                print("Undesired visitor did not leave. Nerf gun is triggered.")
    return predictions

# function to check if unknown is still there after the audible warning
# returns an integer: 1-> undesired visitor stayed after warning
def unknown_stayed(knn_clf):
    check = 0 # if 0, person left; if 1, person stayed
    ret, frame = video.read()
    process_this_frame = 29
    if ret:
        img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        process_this_frame = process_this_frame + 1
        if process_this_frame % 30 == 0:
            results = predict(img, model_path="trained_knn_model.clf")
            if len(results) == 0: # no face detected after warning
                  check = 0
            else:
                face_encodings = results[0]
                found_face_locations = results[1]
                are_matches = results[2]
                knn_clf = results[3]
                for pred, loc, rec in zip(knn_clf.predict(face_encodings), found_face_locations, are_matches):
                    if rec:
                        check = 0;
                    else:
                        check += 1; # person stayed
    return check
                

#opencv not working so had to comment out show_prediciton_labels function
def show_prediction_labels_on_image(frame, predictions):

     pil_image = Image.fromarray(frame)
     draw = ImageDraw.Draw(pil_image)

     for name, (top, right, bottom, left) in predictions:
         # Rezise image back
         top *= 2
         right *= 2
         bottom *= 2
         left *= 2
         # Draw a box around the face using the Pillow module
         draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
         name = name.encode("UTF-8")

         # Draw a label with a name below the face
         text_width, text_height = draw.textsize(name)
         draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
         draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

     # Remove the drawing library from memory as per the Pillow docs.
     del draw
     # Save image in open-cv format to be able to show it.

     opencvimage = np.array(pil_image)
     return opencvimage

def main():
    print("Training KNN classifier...")
    train('known_faces', model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    # process one frame in every 30 frames for faster processing
    process_this_frame = 29
    print('Setting cameras up...')
    while 1 > 0:
        ret, frame = video.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                results = predict(img, model_path="trained_knn_model.clf")
                if len(results) == 0:
                    print("No face detected.")
                else:
                    face_encodings = results[0]
                    found_face_locations = results[1]
                    # locations gives this result [(num1, num2, num3, num4)]
                    # num1 and num4 corresponds to top left corner of face
                    # num2 and num3 corresponds to bottom right corner of face
                    print(found_face_locations)  
                    are_matches = results[2]
                    knn_clf = results[3]
                    predictions = evaluate_visitor(frame, face_encodings, found_face_locations, are_matches, knn_clf)

            #frame = show_prediction_labels_on_image(frame, predictions)
#             cv2.imshow('camera', frame)
#             if ord('q') == cv2.waitKey(10):
#                  # cap1.release()
#                   cv2.destroyAllWindows()
#                   exit(0)

if __name__ == "__main__":
    main()
    
