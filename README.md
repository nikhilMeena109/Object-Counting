# Object-Counting

This project is an object-counting system built using YOLO (You Only Look Once) for object detection. It counts the number of specified objects in images or video feeds. This project is powered by Flask for backend processing and provides a simple web interface.

## Features
- Object detection using YOLO
- Real-time object counting
- Simple web interface built with Flask
- Configurable for different object classes

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/object-counting-system.git
   cd object-counting-system
2.**Set up a virtual environment**:
   python3 -m venv venv
   source venv/bin/activate
3.**Install the required packages**:
  pip install -r requirements.txt
4.**Download YOLO weights**:
  Download the YOLO weights file from Google Drive link .

## Folder Structure
- app.py - Main Flask application file.
- weights/ - Folder to store the YOLO weights file.
- static/ - Static files (CSS, JS).
- templates/ - HTML templates for the web interface.
- requirements.txt - List of dependencies.
