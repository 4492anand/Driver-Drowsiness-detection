# Description
Driver drowsiness detection is a crucial aspect of ensuring road safety, particularly for long-distance drivers or 
those operating vehicles for extended periods. This project implements a driver drowsiness detection system using a Facial Landmark (FL) algorithm, 
computer vision techniques, and machine learning.
The system monitors the driver's facial expressions and eye movements through a webcam or camera mounted on the vehicle's dashboard.
By analyzing changes in facial features and eye behavior, the system can detect signs of drowsiness, such as drooping eyelids, prolonged eye closure, and yawning.

# Features:
Real-time monitoring of driver's facial expressions and eye movements.
Detection of drowsiness based on predefined thresholds and rules.
Visual and/or audible alerts to notify the driver when drowsiness is detected.
Configurable parameters for sensitivity and alert thresholds.

# Key Components:
Facial Landmark Detection: Utilizes a Facial Landmark (FL) algorithm to detect key facial landmarks, such as the positions of the eyes, nose, and mouth.
Eye Aspect Ratio (EAR) Calculation: Calculates the Eye Aspect Ratio (EAR) based on the positions of the eye landmarks. EAR is a measure of eye openness and can be used to detect drowsiness.
Threshold-based Drowsiness Detection: Monitors changes in EAR over time and triggers an alert if the EAR falls below a predefined threshold for a certain duration, indicating drowsiness.
Alert Mechanism: Provides visual and/or audible alerts to notify the driver when drowsiness is detected, prompting them to take corrective action, such as taking a break or stopping the vehicle.
Prerequisites
Before running the driver drowsiness detection system, ensure you have the following prerequisites installed:

# pre-req

Python 3.x,

OpenCV (cv2),

dlib,

scipy,

numpy

# You can install these dependencies using pip:
pip install opencv-python dlib scipy numpy
