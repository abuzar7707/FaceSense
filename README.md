# FaceSense

FaceSense is a computer vision project using **MediaPipe** and **Python** to detect and analyze facial features, gaze, and head pose for distraction detection and window focus automation.

---

## Features

- **Face Detection:** Real-time detection and tracking
- **Head Pose Tracking:** Pitch, yaw, roll orientation
- **Gaze Detection:** Detect where the user is looking
- **Window Automation:** Minimize/restore windows based on attention
- **Landmark Visualization:** Shows facial landmarks on screen

---

## Project Structure

FaceSense/
├── scripts/
│ ├── EyeDetect.py
│ ├── GazeDetect.py
│ ├── focus.py
│ └── ioscript.py
├── models/
│ └── emotion-ferplus-8.onnx
├── README.md
├── requirements.txt
└── .gitignore

yaml
Copy code

---

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- Pillow (optional, used by PyAutoGUI)

Install all dependencies:

```bash
pip install -r requirements.txt
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/abuzar7707/FaceSense.git
Navigate to project folder:

bash
Copy code
cd FaceSense
Run a script, e.g.:

bash
Copy code
python scripts/EyeDetect.py
Follow the on-screen landmark and gaze detection.

Notes
Ensure your camera is functioning.

Adjust PyAutoGUI permissions if needed for window control.

License
MIT License

yaml
Copy code

---

## **4. requirements.txt**

opencv-python>=4.7.0
mediapipe>=0.10.0
numpy>=1.25.0
pyautogui>=0.9.55
pillow>=10.0.0

markdown
Copy code

---

### ✅ **Next steps**

1. Organize your **existing scripts** into `scripts/` and move the model to `models/`.  
2. Add `.gitignore`, `README.md`, and `requirements.txt` to your repo.  
3. Commit and push:  

```bash
git add .
git commit -m "Reorganize project: add scripts, models, README, requirements, gitignore"
git push
