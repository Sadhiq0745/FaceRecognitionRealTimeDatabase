import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://clone-488bc-default-rtdb.firebaseio.com/"  # Replace <project-id>
})

# Reference the Students node
ref = db.reference('Students')

# Data to add
data = {
    "R190199": {
        "Name": "Abdul",
        "Department": "Cse",
        "year": "4th",
        "Total Attendence": 87,
        "Starting year": "2024",
        "Standing" :"10",
        "last attendence":'2024-11-16 10:00:00'


    },
    "R190239": {
        "Name": "Rajesh",
        "Department": "Cse",
        "year": "4th",
        "Total Attendence": 92,
        "Starting year": "2024",
        "Standing" :"10",
        "last attendence":'2024-11-16 10:00:00'
    },
    "R190779": {
        "Name": "Eswar",
        "Department": "Cse",
        "year": "4th",
        "Total Attendence": 88,
        "Starting year": "2024",
        "Standing": "10",
        "last attendence": '2024-11-16 10:00:00'
    },
    "R190029": {
        "Name": "Pavan",
        "Department": "Cse",
        "year": "4th",
        "Total Attendence": 10,
        "Starting year": "2024",
        "Standing": "10",
        "last attendence": '2024-11-16 10:00:00'
    }
}

# Add data to the database
for key, value in data.items():
    ref.child(key).set(value)
