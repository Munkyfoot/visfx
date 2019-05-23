import numpy as np
import cv2 as cv
import pyautogui as gui
import visfx

# Create an FX Stack
FX = visfx.Stack(
    [
        visfx.effects.Denoise(),
        visfx.effects.RemoveBG(),
        visfx.effects.Ghost(),
        visfx.effects.Symmetry(),
        visfx.effects.Tracers()
    ]
)

SCREEN_WIDTH, SCREEN_HEIGHT = gui.size()

# Capture default camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Unable to connect to camera. Closing the program...")
    exit()

while True:
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
    ratio = SCREEN_HEIGHT / height

    rescale = tuple([int(x*ratio) for x in (height, width)])

    output = cv.resize(output, (rescale[1], rescale[0]), cv.INTER_CUBIC)
    border_size = int((SCREEN_WIDTH - rescale[1]) * 0.5)
    output = cv.copyMakeBorder(
        output, 0, 0, border_size, border_size, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # Add tooltips
    font = cv.FONT_HERSHEY_SIMPLEX
    offset = 0
    cv.putText(output, "FX Layers:", (height // 10,
                                            height // 10 + offset), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    offset += height // 10

    for name in FX.getLayerNames():
        if 'ON' in name:
            color = (128,255,128)
        else:
            color = (128,128,255)

        cv.putText(output, name, (height // 10,
                                 height // 10 + offset), font, 0.5, color, 1, cv.LINE_AA)
        offset += height // 10

    offset += height // 10
    for tip in FX.getTooltips():
        cv.putText(output, tip, (height // 10,
                                 height // 10 + offset), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        offset += height // 10

    offset += height // 10
    for readout in FX.getReadouts():
        cv.putText(output, readout, (height // 10,
                                     height // 10 + offset), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        offset += height // 10

    # Display the resulting frame
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('frame', output)

    # Get key press
    key = cv.waitKey(1)
    if key == ord('q'):
        # Close window if 'q' is pressed
        break

    # Pass keypress to the FX Stack
    FX.userInput(key)

# Release the capture
cap.release()
cv.destroyAllWindows()
