import cv2
import os
from cvzone.ClassificationModule import Classifier
import cvzone
from flask import Flask, render_template_string, Response

app = Flask(__name__)

# Define HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Computer Vision Output</title>
</head>
<body>
    <h1>Computer Vision Output</h1>
    <div>
        <img src="{{ video_source }}" width="640" height="480">
    </div>
</body>
</html>
"""

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
classifier = Classifier('Resources/Model/keras_model.h5','Resources/Model/labels.txt')
classIDBin = 0

# Import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the waste images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# Load arrow image
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

# 0 = Recyclable
# 1 = Hazardous
# 2 = Food
# 3 = Residual

classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            imgResize = cv2.resize(frame, (454, 340))
            imgBackground = cv2.imread('Resources/background.png')
            prediction = classifier.getPrediction(frame)
            classID = prediction[1]
            if classID != 0:
                imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
                imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
                classIDBin = classDic[classID]
            imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
            imgBackground[148:148 + 340, 159:159 + 454] = imgResize

            ret, buffer = cv2.imencode('.jpg', imgBackground)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    # Render HTML template
    return render_template_string(html_template, video_source="/video_feed")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
