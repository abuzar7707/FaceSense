import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screeninfo
import time

inactdelay = 10  # seconds of no face before lock
last_seen = time.time()

# ------------------ Minimize/Restore Functions ------------------
def minimize_all_windows(monitor=None):
    try:
        if monitor:
            pyautogui.moveTo(monitor.x + monitor.width//2, monitor.y + monitor.height//2)
        pyautogui.hotkey('win', 'd')
    except Exception as e:
        print("Minimize Error:", e)

def restore_last_window(monitor=None):
    try:
        if monitor:
            pyautogui.moveTo(monitor.x + monitor.width//2, monitor.y + monitor.height//2)
        pyautogui.hotkey('win', 'd')
    except Exception as e:
        print("Restore Error:", e)

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ------------------ Camera Setup ------------------
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ------------------ 3D Model Points ------------------
model_points = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -63.6, -12.5],       # Chin
    [-43.3, 32.7, -26.0],      # Left eye left corner
    [43.3, 32.7, -26.0],       # Right eye right corner
    [-28.9, -28.9, -20.0],     # Left mouth corner
    [28.9, -28.9, -20.0]       # Right mouth corner
], dtype=np.float64)

# ------------------ Variables ------------------
previous_status = "Focused"
locked_face_center = None
lock_threshold = 100
required_frames = 8
distraction_counter = 0
last_action_time = 0
action_cooldown = 2.0

# ------------------ Monitor Info ------------------
monitors = screeninfo.get_monitors()
primary_monitor = monitors[0]

# ------------------ Gaze Detection Helper ------------------
def is_gaze_centered(landmarks, w, h):
    """
    Returns True if gaze is roughly centered based on iris positions.
    """
    # Left eye
    left_iris = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in [468,469,470,471]])
    left_eye = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in [33, 133]])  # left/right corners
    left_ratio = (left_iris.mean(axis=0)[0] - left_eye[0][0]) / (left_eye[1][0] - left_eye[0][0])

    # Right eye
    right_iris = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in [473,474,475,476]])
    right_eye = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in [362, 263]])  # left/right corners
    right_ratio = (right_iris.mean(axis=0)[0] - right_eye[0][0]) / (right_eye[1][0] - right_eye[0][0])

    # Centered if both eyes are roughly 0.35 < ratio < 0.65
    return 0.35 < left_ratio < 0.65 and 0.35 < right_ratio < 0.65

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # ------------------ DRAW LANDMARKS ------------------
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Iris landmarks (highlight)
        iris_points = [468, 469, 470, 471, 472, 473, 474, 475, 476]
        for idx in iris_points:
            ix = int(landmarks[idx].x * w)
            iy = int(landmarks[idx].y * h)
            cv2.circle(frame, (ix, iy), 2, (0, 0, 255), -1)

        # Eye corners (for debugging gaze)
        eye_debug_points = [33, 133, 362, 263]
        for idx in eye_debug_points:
            ex = int(landmarks[idx].x * w)
            ey = int(landmarks[idx].y * h)
            cv2.circle(frame, (ex, ey), 3, (255, 0, 0), -1)

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        face_center = ((xmin+xmax)/2, (ymin+ymax)/2)
        face_width = xmax - xmin

        # Lock face at first detection
        if locked_face_center is None:
            locked_face_center = face_center

        # Ignore other faces
        dist = np.linalg.norm(np.array(face_center) - np.array(locked_face_center))
        if dist > lock_threshold:
            cv2.putText(frame, "Not Your Face", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            distraction_counter = 0
            cv2.imshow("FaceSense", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Ignore if too far
        if face_width < w * 0.15:
            cv2.putText(frame, "Too Far", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            distraction_counter = 0
            cv2.imshow("FaceSense", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ------------------ Head Pose ------------------
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],      # Nose tip
            [landmarks[152].x * w, landmarks[152].y * h],  # Chin
            [landmarks[263].x * w, landmarks[263].y * h],  # Left eye left corner
            [landmarks[33].x * w, landmarks[33].y * h],    # Right eye right corner
            [landmarks[287].x * w, landmarks[287].y * h],  # Left mouth corner
            [landmarks[57].x * w, landmarks[57].y * h]     # Right mouth corner
        ], dtype=np.float64)

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length,0,center[0]],
                                  [0,focal_length,center[1]],
                                  [0,0,1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))

        try:
            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            pitch = np.arctan2(-rmat[2,0], sy) * 180/np.pi
            yaw   = np.arctan2(rmat[1,0], rmat[0,0]) * 180/np.pi
        except:
            pitch = yaw = 0

        # ------------------ Distraction Logic ------------------
        current_time = time.time()
        gaze_centered = is_gaze_centered(landmarks, w, h)
        is_focused = abs(yaw) < 60 and abs(pitch) < 35 and gaze_centered

        if is_focused:
            status = "Focused"
            color = (0,255,0)
            distraction_counter = 0
            if previous_status == "Distracted" and (current_time - last_action_time) > action_cooldown:
                restore_last_window(primary_monitor)
                last_action_time = current_time
                previous_status = "Focused"
        else:
            status = "Distracted"
            color = (0,0,255)
            distraction_counter += 1
            if distraction_counter >= required_frames and previous_status != "Distracted":
                minimize_all_windows(primary_monitor)
                last_action_time = current_time
                previous_status = "Distracted"
                distraction_counter = 0


        cv2.putText(frame, status, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

    else:
        cv2.putText(frame, "No Face Detected", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        distraction_counter = 0
        if results.multi_face_landmarks:
            # face detected → reset timer
            last_seen = time.time()
        else:
            # no face detected → check how long it's been
            if time.time() - last_seen >= inactdelay:
                pyautogui.hotkey('win', 'l')


    cv2.imshow("FaceSense", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
