import sys
import time
from datetime import datetime
import numpy as np
import cv2 as cv
import pyautogui as gui
import visfx

# Get sys args
FULLSCREEN = False
SCALE = 1

for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if arg == '-h' or arg == '--help':
        print("This scripts renders the default camera with an FX stack")
        print("Run with '-f' or '--fullscreen' to enable fullscreen")
        print("Run with '-s' or '--scale' followed by a value to set scale")
        exit()
    elif arg == '-f' or arg == '--fullscreen':
        FULLSCREEN = True
    elif arg == '-s' or arg == '--scale':
        SCALE = float(sys.argv[i+1])
        i += 1

# Create an FX Stack
FX = visfx.Stack(
    [
        visfx.effects.Denoise(),
        visfx.effects.FaceDetect(),
        visfx.effects.MoveDetect(),
        visfx.effects.Ghost(),
        visfx.effects.RemoveBG(),
        visfx.effects.ColorFilter(),
        visfx.effects.Symmetry(),
        visfx.effects.Tracers()
    ]
)

SCREEN_WIDTH, SCREEN_HEIGHT = gui.size()
ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT

# Capture default camera
cap = cv.VideoCapture(0)

FOURCC = cv.VideoWriter_fourcc(*'XVID')
OUT = None

RECORDING = False

LAST_INPUT = None
LAST_OUTPUT = None
LAST_OUTPUT_TIME = time.time()

FPS_OUT_HISTORY = []

TAKE_SNAPSHOT = False

if not cap.isOpened():
    print("Unable to connect to camera. Closing the program...")
    exit()

while True:
    # Get processing time
    fps_in = cap.get(cv.CAP_PROP_FPS)

    ttp = time.time() - LAST_OUTPUT_TIME
    LAST_OUTPUT_TIME = time.time()

    fps_out = 1 / max(0.00001, ttp)
    FPS_OUT_HISTORY.append(fps_out)
    avg_fps_out = 0
    if len(FPS_OUT_HISTORY) > 5:
        del FPS_OUT_HISTORY[0]
    for f in FPS_OUT_HISTORY:
        avg_fps_out += f
    avg_fps_out /= len(FPS_OUT_HISTORY)
    avg_fps_out = min(fps_in, avg_fps_out)

    # Capture frame
    ret, frame = cap.read()

    # Check for frame errors
    if not ret:
        print("Unable to properly read frame. Closing the program...")
        break

    frame_diff = 0
    if LAST_INPUT is not None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        last_gray = cv.cvtColor(LAST_INPUT, cv.COLOR_BGR2GRAY)

        frame_diff = cv.absdiff(gray, last_gray).max()
    LAST_INPUT = frame

    if frame_diff == 0 and LAST_OUTPUT is not None:
        output = LAST_OUTPUT
    else:
        # Apply FX Stack to frame
        output = FX.apply(frame)

        if RECORDING:
            OUT.write(output)

        if TAKE_SNAPSHOT:
            timestamp = datetime.now().isoformat(sep='@', timespec='seconds').replace(
                ':', 'h', 1).replace(':', 'm', 1) + 's'
            cv.imwrite('{}.jpg'.format(timestamp), output)
            TAKE_SNAPSHOT = False

        # Resize and pad output to fit screen
        height, width, channels = output.shape
        if SCALE != 1:
            output = cv.resize(
                output, (int(width * SCALE), int(height * SCALE)))
            height, width, channels = output.shape

        if FULLSCREEN:
            padded_width = int(height * ASPECT_RATIO)
            border_size = int((padded_width - width) * 0.5)
            padded_output = np.zeros((height, padded_width, 3), output.dtype)
            padded_output[:, border_size:border_size+width] = output
            output = padded_output
            height, width, channels = output.shape

        # Display info
        font = cv.FONT_HERSHEY_SIMPLEX
        font_size = height / 2000
        offset = height // 100
        cv.putText(output, "FPS IN: {:.1f} | FPS OUT: {:.1f}".format(fps_in, avg_fps_out), (height // 100,
                                                                                            height // 100 + offset), font, font_size, (255, 255, 255), 1, cv.LINE_AA)
        offset += height // 40
        cv.putText(output, "FX Layers:", (height // 100,
                                          height // 100 + offset), font, font_size, (255, 255, 255), 1, cv.LINE_AA)
        offset += height // 40

        for name in FX.getLayerNames():
            if 'ON' in name:
                color = (128, 255, 128)
            else:
                color = (128, 128, 255)

            cv.putText(output, name, (height // 100,
                                      height // 100 + offset), font, font_size, color, 1, cv.LINE_AA)
            offset += height // 40

        offset += height // 40
        for tip in FX.getTooltips():
            cv.putText(output, tip, (height // 100,
                                     height // 100 + offset), font, font_size, (255, 255, 255), 1, cv.LINE_AA)
            offset += height // 40

        offset += height // 40
        for readout in FX.getReadouts():
            cv.putText(output, readout, (height // 100,
                                         height // 100 + offset), font, font_size, (255, 255, 255), 1, cv.LINE_AA)
            offset += height // 40

        if RECORDING:
            cv.putText(output, "Recording...", (height // 100,
                                                height // 100 + offset), font, font_size, (255, 255, 255), 1, cv.LINE_AA)
            offset += height // 40

        LAST_OUTPUT = output

    # Display the resulting frame
    if FULLSCREEN:
        cv.namedWindow('VisFX Render', cv.WINDOW_NORMAL)
        cv.setWindowProperty(
            'VisFX Render', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('VisFX Render', output)

    # Get key press
    key = cv.waitKey(1)
    if key == ord('q'):
        # Close window if 'q' is pressed
        break
    if key == ord('r'):
        RECORDING = not RECORDING
        if RECORDING:
            timestamp = datetime.now().isoformat(sep='@', timespec='seconds').replace(
                ':', 'h', 1).replace(':', 'm', 1) + 's'
            OUT = cv.VideoWriter('{}.avi'.format(
                timestamp), FOURCC, 20, (frame.shape[1], frame.shape[0]))
        else:
            OUT = None

    if key == ord('i'):
        TAKE_SNAPSHOT = True
    # Pass keypress to the FX Stack
    FX.userInput(key)

# Release the capture
cap.release()
cv.destroyAllWindows()
