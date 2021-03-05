# import face_recognition
# import os #to iterate over directories
# import cv2 #to do image tests, draw rectangles label image and stuff
# import numpy as np

# KNOWN_FACES_DIR = "known_faces"
# #UNKNOWN_FACES_DIR = "unknown_faces"
# TOLERANCE = 0.4 #LOWER THE TOLERANCE, more strict (more chances of known faces to be missed), HIGHER TOLERANCE, more chance for false positive (more faces incorrectly recognized)
# FRAME_THICKNESS = 3 #RECTANGLE FRAME FOR FACE DETECTION, IN PIXELS
# FONT_THICKNESS = 2
# MODEL = "hog" #convolutional neural network, can also use hog (older way for image detection)
# video = cv2.VideoCapture(0) #video feed

# #load in known faces, have something for the unknowns to compare to
# print("loading known faces")

# known_faces = []
# known_names = []

# #iterate over known faces and store info
# #each subfolder of known_faces become the label for found faces
# for name in os.listdir(KNOWN_FACES_DIR):
#     #load every file of faces of known person
#     for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):

#         #load an image
#         image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name, filename))

#         #get 128-dimension face encoding
#         #always returns a list of found faces, 0th index bc only want first face found since can have image with two people
#         encoding = face_recognition.face_encodings(image)[0]

#         #append encodings and name
#         known_faces.append(encoding)
#         known_names.append(name)



# #iterate over unknown faces and try to identify
# print("processing unknown faces...")

# face_names = []
# process_frame = True

# #loop over video feed from camera
# while(video.isOpened()):
#     #capture frame by frame
#     ret, image = video.read()

#     #resize video frame to 1/4 for faster face rec processing
#     resized_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

#     #convert to rgb since face_recogniton uses it
#     rgb_resized_frame = resized_frame[:, :, ::-1]

#     #process every other frame
#     if process_frame:

#         #locate the faces in the unknown images
#         locations = face_recognition.face_locations(rgb_resized_frame, model=MODEL)
#         #encode the images
#         encodings = face_recognition.face_encodings(rgb_resized_frame, locations)

#         #assume there might be more than one face in unknown image, find faces of different people
#         print(f', found {len(encodings)} face(s)')
#         for face_encoding in encodings:
#             #return array of true/false values in order of passed known_faces
#             results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
#             #print(results)

#             #if found face, label of first matching known face within tolerance
#             match = None
#             #take index value of any Trues, and look for name from known_names variable
#             # if True in results:
#             #     match = known_names[results.index(True)]
#             #     #print(f'Match found: {match}')

#             face_distances = face_recognition.face_distance(known_faces, face_encoding)
#             best_match = np.argmin(face_distances)
#             if results[best_match]:
#                 match = known_names[best_match]

#     process_frame = not process_frame

#     for (top, right, bottom, left), name in zip(locations, known_names):
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         #make rectangle on face
#         cv2.rectangle(image, (left, top), (right, bottom),(0,0,255), FRAME_THICKNESS)

#         #write name
#         cv2.putText(image, name, (left + 6, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)


#     cv2.imshow(filename, image)
#     # break if q is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#     #cv2.waitKey(10000)
#     #cv2.destroyWindow(filename)

# video.release()
# cv2.destroyAllWindows()




# import face_recognition
# import os #to iterate over directories
# import cv2 #to do image tests, draw rectangles label image and stuff

# KNOWN_FACES_DIR = "known_faces"
# #UNKNOWN_FACES_DIR = "unknown_faces"
# TOLERANCE = 0.4 #LOWER THE TOLERANCE, more strict (more chances of known faces to be missed), HIGHER TOLERANCE, more chance for false positive (more faces incorrectly recognized)
# FRAME_THICKNESS = 3 #RECTANGLE FRAME FOR FACE DETECTION, IN PIXELS
# FONT_THICKNESS = 2
# MODEL = "hog" #convolutional neural network, can also use hog (older way for image detection)
# video = cv2.VideoCapture(0) #video feed

# #load in known faces, have something for the unknowns to compare to
# print("loading known faces")

# known_faces = []
# known_names = []

# #iterate over known faces and store info
# #each subfolder of known_faces become the label for found faces
# for name in os.listdir(KNOWN_FACES_DIR):
#     #load every file of faces of known person
#     for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):

#         #load an image
#         image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name, filename))

#         #get 128-dimension face encoding
#         #always returns a list of found faces, 0th inex bc only want first face found since can have image with two people
#         encoding = face_recognition.face_encodings(image)[0]

#         #append encodings and name
#         known_faces.append(encoding)
#         known_names.append(name)



# #iterate over unknown faces and try to identify
# print("processing unknown faces...")

# #loop over unknown faces folder that want to label
# #for filename in os.listdir(UNKNOWN_FACES_DIR):
# while True:
#     #video feed
#     ret, image = video.read()

#     # resized_frame = cv2.resize(image, (0,0), fx=0.25, fy=0.25)

#     #locate the faces in the unknown images
#     locations = face_recognition.face_locations(image, model=MODEL)
#     #encode the images
#     encodings = face_recognition.face_encodings(image, locations)

#     #if find a match with a known face, draw rectangle around face, using opencv

#     #assume there might be more thaan one face in unknown image, find faces of different people
#     print(f', found {len(encodings)} face(s)')
#     for face_encoding, face_location in zip(encodings, locations):
#         #return array of true/false values in order of pased known_faces
#         results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
#         print(results)

#         #if found face, label of first matching known face within tolerance
#         match = None
#         #take index value of any Trues, and look for name from known_names variable
#         if True in results:
#             match = known_names[results.index(True)]
#             print(f'Match found: {match}')

#             #corners of face found to draw rectangle
#             top_left = (face_location[3], face_location[0])
#             bottom_right = (face_location[1], face_location[2])

#             #get colr by name
#             #color = name_to_color(match)
#             color = (0,0,255)

#             #make rectangle on face
#             cv2.rectangle(image, top_left, bottom_right,color, FRAME_THICKNESS)

#             #need smaller rectangle for name of face
#             #use bottom in both corners, start from bottom and move 50 pixels down
#             top_left = (face_location[3], face_location[2])
#             bottom_right = (face_location[1], face_location[2] + 22)

#             #create rectangle for label
#             cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

#             #write name
#             cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)


#     cv2.imshow(filename, image)
#     # break if q is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#     #cv2.waitKey(10000)
#     #cv2.destroyWindow(filename)

# video.release()
# cv2.destroyAllWindows()


from cv2 import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

KNOWN_FACES_DIR = "known_faces"

def train(KNOWN_FACES_DIR, model_save_path=None, n_neighbors=None, knn_algo='ball_tree'):
    
    known_faces = []
    known_names = []

    # Loop through each person in the training set
    for name in os.listdir(KNOWN_FACES_DIR):
        if not os.path.isdir(os.path.join(KNOWN_FACES_DIR, name)):
            continue

        # Loop through each training image for the current person
        for filename in image_files_in_folder(os.path.join(KNOWN_FACES_DIR, name)):
            image = face_recognition.load_image_file(filename)
            face_bounding_boxes = face_recognition.face_locations(image)

            # Add face encoding for current image to the training set
            known_faces.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            known_names.append(name)

    # Determine how many neighbors to use for weighting in the KNN classifier
    # Unknown face will be tested based on the weight of its two nearest neighbours using euclidean distance
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


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold = 0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    are_matches = [min(closest_distances[0][i][0], closest_distances[0][i][1]) <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    predictions = []
    for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches):
        #print(knn_clf.predict(faces_encodings))
        if rec:
            predictions.append((pred,loc))
        else:
            predictions.append(("unknown",loc))
    
    print(predictions)
    return predictions

def show_prediction_labels_on_image(frame, predictions):

    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # rezise image back
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
    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Setting cameras up...')
    video = cv2.VideoCapture(0)
    while 1 > 0:
        ret, frame = video.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap1.release()
                cv2.destroyAllWindows()
                exit(0)

if __name__ == "__main__":
    main()