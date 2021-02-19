import face_recognition
import os #to iterate over directories
import cv2 #to do image tests, draw rectangles label image and stuff

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6 #LOWER THE TOLERANCE, more strict (more chances of known faces to be missed), HIGHER TOLERANCE, more chance for false positive (more faces incorrectly recognized)
FRAME_THICKNESS = 3 #RECTANGLE FRAME FOR FACE DETECTION, IN PIXELS
FONT_THICKNESS = 2
MODEL = "cnn" #convolutional neural network, can also use hog (older way for image detection)

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
for filename in os.listdir(UNKNOWN_FACES_DIR):
    #load images
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(os.path.join(UNKNOWN_FACES_DIR, filename))

    #locate the faces in the unknown images
    locations = face_recognition.face_locations(image, model=MODEL)
    #encode the images
    encodings = face_recognition.face_encodings(image, locations)

    #if find a match with a known face, draw rectangle around face, using opencv
    #first need to conver image from RGB to BGR, since opencv uses this
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

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
            print(f' -{match} from {results}')

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
cv2.waitKey(0)
cv2.destroyWindow(filename)

