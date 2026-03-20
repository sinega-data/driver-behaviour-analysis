# driver-behaviour-analysis
Real-time driver drowsiness and attention monitoring using MediaPipe and OpenCV
# Driver Behaviour Monitoring System

A real-time driver behaviour analysis system using Computer Vision and MediaPipe.

## Features
- >**Drowsiness Detection** — Eye Aspect Ratio (EAR) algorithm detects eye closure
-> **Attention Monitoring** — Head pose estimation detects distracted driving
- >**Real-time Alerts** — Visual alert when drowsy or distracted state detected

## Tech Stack
-> Python 3.x
-> OpenCV
-> MediaPipe
- >NumPy

## Installation
```bash
pip install opencv-python mediapipe numpy scipy
```

## Run
```bash
python main.py
```

## How It Works
- >MediaPipe Face Mesh extracts 468 facial landmarks in real-time
- >EAR (Eye Aspect Ratio) calculated from 6 eye landmarks per eye
- >If EAR < 0.25 for 20 consecutive frames → DROWSY alert triggered
- >Head pitch angle estimated via solvePnP → DISTRACTED alert if pitch < -15°

## Output
1.ACTIVE --> Driver is focused and awake 
2.DROWSY-->Eyes closing — fatigue detected 
3.DISTRACTED-->Head tilted down — attention lost 

## Author
M.Sinega — B.Tech Artificial Intelligence & Data Science
EGS Pillay Engineering College, Nagapattinam


