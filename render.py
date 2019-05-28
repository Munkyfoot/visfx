import numpy as np
import cv2 as cv
import pyautogui as gui
import visfx
import time
import sys

# Get sys args
FULLSCREEN = False
SCALE = 1

for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if arg == '-h' or arg == '--help':
        print("This scripts renders the default camera with an FX stack")
        print("Run with '-f' or '--fullscreen' to enable fullscreen")
        print("Run with '-s' or '--scale' followed by a value to set scale. No effect in Fullscreen mode.")
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

# Capture default camera
cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, gui.size())

RECORDING = False
LAST_OUTPUT_TIME = time.time()

FPS_OUT_HISTORY = []

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

    # Capture frame
    ret, frame = cap.read()

    # Check for frame errors
    if not ret:
        print("Unable to properly read frame. Closing the program...")
        break

    # Apply FX Stack to frame
    output = FX.apply(frame)

    # Resize and pad output to fit screen
    height, width, channels = output.shape    

    if FULLSCREEN:
        ratio = SCREEN_HEIGHT / height
        rescale = tuple([int(x*ratio) for x in (height, width)])
        output = cv.resize(output, (rescale[1], rescale[0]))
        border_size_x = int((SCREEN_WIDTH - rescale[1]) * 0.5)
        border_size_y = int((SCREEN_HEIGHT - rescale[0]) * 0.5)
        output = cv.copyMakeBorder(
            output, border_size_y, border_size_y, border_size_x, border_size_x, cv.BORDER_CONSTANT, value=[0, 0, 0])
        height, width, channels = output.shape
    elif SCALE != 1:        
        output = cv.resize(output, (int(width * SCALE), int(height * SCALE)))
        height, width, channels = output.shape
    # Add tooltips
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
        out.write(output)
        cv.putText(output, "Recording...", (height // 100,
                                            height // 100 + offset), font, font_size, (255, 255, 255), 1, cv.LINE_AA)
        offset += height // 40
    
    # Display the resulting frame
    if FULLSCREEN:
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('frame', output)

    # Get key press
    key = cv.waitKey(1)
    if key == ord('q'):
        # Close window if 'q' is pressed
        break
    if key == ord('r'):
        RECORDING = not RECORDING
    # Pass keypress to the FX Stack
    FX.userInput(key)

# Release the capture
cap.release()
cv.destroyAllWindows()
