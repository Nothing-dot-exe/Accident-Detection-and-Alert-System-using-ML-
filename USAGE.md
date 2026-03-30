# How to Use the Accident Detection System

The easiest way to run the system is by using the provided batch file or running the Python script directly.

## 1. Running the Application

### Using `run.bat` (Recommended for Windows)
1. Go into the `car-human-counting` folder.
2. Double-click on the `run.bat` file. 
3. A command window will pop up asking you a couple of questions:
   - **Input Source:** Choose `[1]` for your Webcam or `[2]` to pick a video file on your computer.
   - **Window Size:** Choose your preferred display resolution (Default, Full HD, or Small). 
4. After selecting these options, the tracking window will open. If you chose "Video File," a file picker dialog will let you select a video (you can select `crash.mp4` from your main folder to test it).

### Using the Python script directly
Alternatively, you can open a terminal/command prompt in the `car-human-counting` folder and run:
```bash
python car+human.py
```
By default, this will immediately open a graphical file dialog prompting you to select an `.mp4` video.

## 2. Pre-trained Models

The system relies on two different AI models under the hood, which are loaded automatically by `car+human.py`:

1. **`yolov8n.pt`** (General Object Detection Model)
   - **Location:** Inside your `car-human-counting` folder.
   - **Purpose:** This is the standard YOLOv8 "nano" model. The system uses this to track humans and general vehicles (cars, trucks, buses, motorcycles) across the screen.

2. **`accident_model.pt`** (Dedicated AI Accident Model)
   - **Location:** Inside the `car-human-counting` folder.
   - **Purpose:** This is your specially trained custom model. It is evaluating the frames specifically to detect crashes. When it recognizes an incident and two vehicles intersect, it triggers the red "ACCIDENT DETECTED!" UI overlay.
