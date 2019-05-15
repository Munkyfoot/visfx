import numpy as np
import cv2 as cv
import visfx

# Create an FX Stack
FX = visfx.Stack(
    [
        visfx.layers.Tracers(),
        visfx.layers.Symmetry()
    ]
)
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

    # Display the resulting frame
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
