import datetime
import os
import pickle
import cv2
import cvzone
import face_recognition
import firebase_admin
import numpy as np
from datetime import datetime



from firebase_admin import db, credentials, storage
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://clone-488bc-default-rtdb.firebaseio.com/" , # Replace <project-id>
    'storageBucket': "clone-488bc.appspot.com"
})


bucket = storage.bucket()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Importing mode images into the list
imgBackground = cv2.imread('Resources/background.jpg')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading encoded file...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode file loaded...")
# Define a threshold for face distance (adjustable based on accuracy required)
FACE_DISTANCE_THRESHOLD = 0.6
modeType = 0
counter=0
id=-1
imgStudent =[]
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    if faceCurFrame:
    # Loop over each detected face
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
           # print("Distances:", faceDis)  # Debugging distances for clarity
            # Find the best match based on the minimum face distance
            matchIndex = np.argmin(faceDis)  # Index of closest face
            if faceDis[matchIndex] < FACE_DISTANCE_THRESHOLD:  # Ensure it meets threshold
                matches = [False] * len(faceDis)  # Create a precise match array
                matches[matchIndex] = True
               # print("Matches:", matches)  # Example: [False, True, False]

                # Draw bounding box for matched face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                # Display matched student ID
                studentId = studentIds[matchIndex]
                #print(f"Matched Student ID: {studentId}")
                id = studentIds[matchIndex]

                if counter == 0:
                    counter = 1
                    modeType = 1


        if counter != 0:

            if counter == 1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                blob = bucket.get_blob(f'Images/{id}.jpg')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent =cv2.imdecode(array,cv2.COLOR_BGRA2BGR)

            datetime_now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Calculate seconds since the last attendance
            if studentInfo['last attendence']:
                datetimeObject = datetime.strptime(studentInfo['last attendence'], '%Y-%m-%d %H:%M:%S')
                seconds = (datetime.now() - datetimeObject).total_seconds()
                print("Seconds since last attendance:", seconds)
            else:
                print("No previous attendance recorded.")

            if seconds>15:

            # Update the attendance count and last attendance
                ref = db.reference(f'Students/{id}')
                studentInfo['Total Attendence'] += 1
                ref.child('Total Attendence').set(studentInfo['Total Attendence'])
                ref.child('last attendence').set(datetime_now_str)


                print("Attendance updated successfully!")
            else :
                modeType=3
                counter =0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                if 10<counter<20:
                    modeType =2
                imgBackground[44:44+633,808:808+414]= imgModeList[modeType]

                if counter <=10:
                    cv2.putText(imgBackground,str(studentInfo['Total Attendence']),(861,125),
                                cv2.FONT_HERSHEY_COMPLEX,1,(225,255,255),1)

                    cv2.putText(imgBackground, str(studentInfo['Department']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (225, 255, 255), 1)

                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100,100,100), 1)

                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (225, 255, 255), 1)

                    cv2.putText(imgBackground, str(studentInfo['Standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    cv2.putText(imgBackground, str(studentInfo['Starting year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 100, 100), 1)

                    (w,h),_ =cv2.getTextSize(studentInfo['Name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
                    offset =(414-w)//2

                    cv2.putText(imgBackground, str(studentInfo['Name']), (808+offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    #imgBackground[175:175+216,909:909+216] = imgStudent
                    height, width = imgStudent.shape[:2]
                    aspect_ratio = width / height
                    # Calculate new dimensions while preserving aspect ratio
                    new_width = 216
                    new_height = int(new_width / aspect_ratio)
                    if new_height > 216:
                        # If height exceeds the target, resize by height instead
                        new_height = 216
                        new_width = int(new_height * aspect_ratio)
                    # Resize the image
                    imgStudent = cv2.resize(imgStudent, (new_width, new_height))
                    # Calculate padding to center the image in a 216x216 frame
                    padding_top = max(0, (216 - new_height) // 2)
                    padding_bottom = 216 - new_height - padding_top
                    padding_left = max(0, (216 - new_width) // 2)
                    padding_right = 216 - new_width - padding_left

                    # Add padding to fit the target size
                    imgStudent = cv2.copyMakeBorder(
                        imgStudent,
                        padding_top, padding_bottom, padding_left, padding_right,
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Black border
                    )
                    # Place the padded image in the background
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                counter +=1
                if counter>=20:
                    counter =0
                    modeType =0
                    studentInfo=[]
                    imgStudent=[]
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType=0
        counter=0
    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()