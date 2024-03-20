# code  
import cv2
import dlib
from scipy.spatial import distance as dist

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Define constants for drowsiness detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract left and right eye landmarks
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate eye aspect ratio for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the eye aspect ratio
        ear = (left_ear + right_ear) / 2.0

        # Draw contours around eyes
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # If the eyes have been closed for a sufficient number of frames, sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    print("Drowsiness detected!")
                    # You can add code here to trigger an alarm (e.g., sound, notification)
        else:
            COUNTER = 0
            ALARM_ON = False

        # Display the calculated eye aspect ratio on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


