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




import face_recognition
import os #to iterate over directories
import cv2 #to do image tests, draw rectangles label image and stuff

KNOWN_FACES_DIR = "known_faces"
#UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.4 #LOWER THE TOLERANCE, more strict (more chances of known faces to be missed), HIGHER TOLERANCE, more chance for false positive (more faces incorrectly recognized)
FRAME_THICKNESS = 3 #RECTANGLE FRAME FOR FACE DETECTION, IN PIXELS
FONT_THICKNESS = 2
MODEL = "hog" #convolutional neural network, can also use hog (older way for image detection)
video = cv2.VideoCapture(0) #video feed

#load in known faces, have something for the unknowns to compare to
print("loading known faces")

known_faces = []
known_names = []

#iterate over known faces and store info
#each subfolder of known_faces become the label for found faces
for name in os.listdir(KNOWN_FACES_DIR):
    #load every file of faces of known person
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):

        #load an image
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name, filename))

        #get 128-dimension face encoding
        #always returns a list of found faces, 0th inex bc only want first face found since can have image with two people
        encoding = face_recognition.face_encodings(image)[0]

        #append encodings and name
        known_faces.append(encoding)
        known_names.append(name)



#iterate over unknown faces and try to identify
print("processing unknown faces...")

#loop over unknown faces folder that want to label
#for filename in os.listdir(UNKNOWN_FACES_DIR):
while True:
    #video feed
    ret, image = video.read()

    # resized_frame = cv2.resize(image, (0,0), fx=0.25, fy=0.25)

    #locate the faces in the unknown images
    locations = face_recognition.face_locations(image, model=MODEL)
    #encode the images
    encodings = face_recognition.face_encodings(image, locations)

    #if find a match with a known face, draw rectangle around face, using opencv

    #assume there might be more thaan one face in unknown image, find faces of different people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        #return array of true/false values in order of pased known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        print(results)

        #if found face, label of first matching known face within tolerance
        match = None
        #take index value of any Trues, and look for name from known_names variable
        if True in results:
            match = known_names[results.index(True)]
            print(f'Match found: {match}')

            #corners of face found to draw rectangle
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            #get colr by name
            #color = name_to_color(match)
            color = (0,0,255)

            #make rectangle on face
            cv2.rectangle(image, top_left, bottom_right,color, FRAME_THICKNESS)

            #need smaller rectangle for name of face
            #use bottom in both corners, start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            #create rectangle for label
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            #write name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)


    cv2.imshow(filename, image)
    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #cv2.waitKey(10000)
    #cv2.destroyWindow(filename)

video.release()
cv2.destroyAllWindows()