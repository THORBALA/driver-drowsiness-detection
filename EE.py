import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
import random

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Set up the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for drowsiness detection
EAR_THRESHOLD = 0.25
EYE_AR_CONSECUTIVE_FRAMES = 10
eye_closed_frames = 0
drowsy_start_time = None
drowsy_detection_duration = 0.1  # seconds

# Variables for yawning detection
MOUTH_WIDTH_THRESHOLD = 0.6
MOUTH_HEIGHT_THRESHOLD = 0.5
yawn_detected = False
yawn_start_time = None
yawn_duration_threshold = 2.0  # seconds

# Eye landmarks (left and right eyes)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Mouth landmarks (for yawning)
UPPER_LIP_LANDMARK = 13
LOWER_LIP_LANDMARK = 14
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291

# Colors for UI elements
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

# Load the alert sound
alert_sound_path = "alert.wav"

# Vehicle data simulation function
def generate_vehicle_data():
    speed = random.randint(40, 100)
    brake = random.randint(0, 1)

    # Simulate erratic driving 10% of the time
    if random.random() < 0.1:
        speed = random.randint(20, 40)  # sudden drop in speed
        brake = 1  # sudden braking

    return speed, brake

# Vehicle behavior monitoring function
def monitor_vehicle_behavior():
    while True:
        speed, brake = generate_vehicle_data()
        print(f"Speed: {speed} km/h, Brake: {'ON' if brake else 'OFF'}")
        if speed < 40 and brake:
            print("Warning: Erratic vehicle behavior detected! Possible drowsiness.")
            winsound.Beep(1000, 500)  # Sound alert
        time.sleep(1)

def play_alert_sound():
    winsound.Beep(1000, 1000)

def calculate_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness(ear):
    global eye_closed_frames, drowsy_start_time
    if ear < EAR_THRESHOLD:
        eye_closed_frames += 1
        if eye_closed_frames >= EYE_AR_CONSECUTIVE_FRAMES:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time > drowsy_detection_duration:
                play_alert_sound()  # Play sound when drowsiness is detected
                return True
    else:
        eye_closed_frames = 0
        drowsy_start_time = None
    return False

def detect_yawning(landmarks, frame_width, frame_height):
    global yawn_detected, yawn_start_time

    left_lip_corner = np.array([landmarks.landmark[LEFT_LIP_CORNER].x * frame_width, landmarks.landmark[LEFT_LIP_CORNER].y * frame_height])
    right_lip_corner = np.array([landmarks.landmark[RIGHT_LIP_CORNER].x * frame_width, landmarks.landmark[RIGHT_LIP_CORNER].y * frame_height])
    upper_lip = np.array([landmarks.landmark[UPPER_LIP_LANDMARK].x * frame_width, landmarks.landmark[UPPER_LIP_LANDMARK].y * frame_height])
    lower_lip = np.array([landmarks.landmark[LOWER_LIP_LANDMARK].x * frame_width, landmarks.landmark[LOWER_LIP_LANDMARK].y * frame_height])

    mouth_width = np.linalg.norm(left_lip_corner - right_lip_corner)
    mouth_height = np.linalg.norm(upper_lip - lower_lip)

    face_width = np.linalg.norm(left_lip_corner - right_lip_corner)
    adjusted_mouth_width_threshold = 0.4 * face_width
    adjusted_mouth_height_threshold = 0.3 * face_width

    if mouth_width > adjusted_mouth_width_threshold and mouth_height > adjusted_mouth_height_threshold:
        if not yawn_detected:
            yawn_detected = True
            yawn_start_time = time.time()
        elif time.time() - yawn_start_time > yawn_duration_threshold:
            play_alert_sound()
            return True
    else:
        yawn_detected = False
        yawn_start_time = None
    return False

# Run the camera-based drowsiness and yawning detection
def camera_based_drowsiness_detection():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Draw face mesh with subtle color
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=None,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # Eye aspect ratio (EAR) calculation
                left_eye_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in LEFT_EYE_LANDMARKS]
                right_eye_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in RIGHT_EYE_LANDMARKS]

                ear_left = calculate_eye_aspect_ratio(left_eye_points)
                ear_right = calculate_eye_aspect_ratio(right_eye_points)
                ear = (ear_left + ear_right) / 2

                # Detect drowsiness
                if detect_drowsiness(ear):
                    cv2.putText(frame, "Drowsiness Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                    cv2.rectangle(frame, (0, 0), (frame_width, frame_height), RED, 10)  # Red border when drowsiness detected

                # Detect yawning
                if detect_yawning(landmarks, frame_width, frame_height):
                    cv2.putText(frame, "Yawning Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
                    cv2.circle(frame, (int(frame_width / 2), int(frame_height / 2)), 100, GREEN, 5)  # Green circle for yawning

        # Display the frame
        cv2.imshow('Enhanced Drowsiness and Yawning Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to run both detections
def main():
    # Start vehicle behavior monitoring in a separate thread
    vehicle_thread = threading.Thread(target=monitor_vehicle_behavior)
    vehicle_thread.start()

    # Run camera-based drowsiness detection
    camera_based_drowsiness_detection()

    # Wait for the vehicle thread to finish
    vehicle_thread.join()

if __name__ == "__main__":
    main()
