import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import screeninfo
import time


# ------------------ Minimize/Restore Functions ------------------
def minimize_all_windows(monitor=None):
    try:
        if monitor:
            pyautogui.moveTo(monitor.x + monitor.width // 2, monitor.y + monitor.height // 2)
        pyautogui.hotkey('win', 'd')
    except Exception as e:
        print("Minimize Error:", e)


def restore_last_window(monitor=None):
    try:
        if monitor:
            pyautogui.moveTo(monitor.x + monitor.width // 2, monitor.y + monitor.height // 2)
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
    [0.0, 0.0, 0.0],  # Nose tip
    [0.0, -63.6, -12.5],  # Chin
    [-43.3, 32.7, -26.0],  # Left eye left corner
    [43.3, 32.7, -26.0],  # Right eye right corner
    [-28.9, -28.9, -20.0],  # Left mouth corner
    [28.9, -28.9, -20.0]  # Right mouth corner
], dtype=np.float64)

# ------------------ Face Lock & Distraction ------------------
previous_status = None
locked_face_center = None
lock_threshold = 80
required_frames = 5
distraction_counter = 0
last_action_time = 0
action_cooldown = 1

# ------------------ Monitor Info ------------------
monitors = screeninfo.get_monitors()
primary_monitor = monitors[0]


# ------------------ Eye Detection Function ------------------
def eye_aspect_ratio(landmarks, left=True, w=640, h=480):
    # MediaPipe Face Mesh indices for eyes
    if left:
        points = [33, 160, 158, 133, 153, 144]  # left eye
    else:
        points = [263, 387, 385, 362, 380, 373]  # right eye

    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in points]
    # Vertical distances
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C)
    return ear


EAR_THRESHOLD = 0.25  # Threshold to detect closed eye

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

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        face_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        face_width = xmax - xmin

        # Lock face at first detection
        if locked_face_center is None:
            locked_face_center = face_center

        # Ignore other faces
        dist = np.linalg.norm(np.array(face_center) - np.array(locked_face_center))
        if dist > lock_threshold:
            cv2.putText(frame, "Not Your Face", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            distraction_counter = 0
            cv2.imshow("FaceSense", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Ignore if too far
        if face_width < w * 0.15:
            cv2.putText(frame, "Too Far", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            distraction_counter = 0
            cv2.imshow("FaceSense", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ------------------ Head Pose ------------------
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],  # Nose tip
            [landmarks[152].x * w, landmarks[152].y * h],  # Chin
            [landmarks[263].x * w, landmarks[263].y * h],  # Left eye left corner
            [landmarks[33].x * w, landmarks[33].y * h],  # Right eye right corner
            [landmarks[287].x * w, landmarks[287].y * h],  # Left mouth corner
            [landmarks[57].x * w, landmarks[57].y * h]  # Right mouth corner
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            pitch = np.arctan2(-rmat[2, 0], sy) * 180 / np.pi
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180 / np.pi
        except:
            pitch = yaw = 0

        # ------------------ Eye Detection ------------------
        left_ear = eye_aspect_ratio(landmarks, left=True, w=w, h=h)
        right_ear = eye_aspect_ratio(landmarks, left=False, w=w, h=h)
        eyes_open = left_ear > EAR_THRESHOLD and right_ear > EAR_THRESHOLD

        # ------------------ Distraction ------------------
        current_time = time.time()
        if abs(yaw) < 60 and abs(pitch) < 35 and eyes_open:
            status = "Focused"
            color = (0, 255, 0)
            distraction_counter = 0
        else:
            status = "Distracted"
            color = (0, 0, 255)
            distraction_counter += 1

        # ------------------ Trigger ------------------
        if distraction_counter >= required_frames and (current_time - last_action_time) > action_cooldown:
            minimize_all_windows(primary_monitor)
            last_action_time = current_time
            previous_status = "Distracted"
            distraction_counter = 0
        elif status == "Focused" and previous_status == "Distracted" and (
                current_time - last_action_time) > action_cooldown:
            restore_last_window(primary_monitor)
            last_action_time = current_time
            previous_status = "Focused"

        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Left EAR: {left_ear:.2f} Right EAR: {right_ear:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    else:
        cv2.putText(frame, "No Face Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("FaceSense", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
