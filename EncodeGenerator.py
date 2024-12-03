from mtcnn import MTCNN
import face_recognition
import cv2
import pickle
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://clone-488bc-default-rtdb.firebaseio.com/" , # Replace <project-id>
    'storageBucket': "clone-488bc.appspot.com"
})
# Path to the folder containing images
folderPath = 'Images'
PathList = os.listdir(folderPath)
print("Image Paths:", PathList)

# Lists to store images and student IDs
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
print("Student IDs:", studentIds)
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face detected in an image. Skipping...")
    return encodeList

print("Encoding started...")
encodeListKnown = findEncodings(imgList)
if len(encodeListKnown) == len(studentIds):
    print("All faces encoded successfully.")
else:
    print(f"Warning: Encoded {len(encodeListKnown)} faces but found {len(studentIds)} IDs.")

# Save encodings and IDs to a file
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding complete. Saving to file...")

with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)

print("File Saved: EncodeFile.p")