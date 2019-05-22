import numpy as np
import cv2 as cv


class Layer:
    '''Visual FX Layer (Empty)'''

    def __init__(self):
        self.type = None

    def apply(self, frame):
        return frame

    def userInput(self, key):
        pass


class Ghost(Layer):
    '''Visual FX Layer (Ghost)'''

    def __init__(self):
        self.type = 'Ghost'
        self.background = None
        self.last_input = None
        self.last_output = None
        self.tooltips = ["Press 'B' to set the background"]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        if self.background is not None:
            output = cv.addWeighted(self.background, 1 / 3, frame, 2 / 3, 0)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('b'):
            self.background = self.last_input


class RemoveBG(Layer):
    '''Visual FX Layer (Remove BG)'''

    def __init__(self):
        self.type = 'Remove BG'
        self.background = None
        self.threshold = 32
        self.last_input = None
        self.last_output = None
        self.tooltips = ["Press 'B' to set the background",
        "Press 'T' to adjust BG detection threshold"]
        self.readouts = ["BG Detection Threshold:{}".format(self.threshold)]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape        

        if self.background is not None:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            hsv_bg = cv.cvtColor(self.background, cv.COLOR_BGR2HSV)
            output[(cv.absdiff(hsv[:,:,0], hsv_bg[:,:,0]) + cv.absdiff(hsv[:,:,1], hsv_bg[:,:,1]) + cv.absdiff(hsv[:,:,2], hsv_bg[:,:,2])) < self.threshold] = 0

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('b'):
            self.background = self.last_input

        if key == ord('t'):
            self.threshold += 16
            if self.threshold > 255:
                self.threshold = 16
            self.readouts[0] = "BG Detection Threshold:{}".format(self.threshold)


class Tracers(Layer):
    '''Visual FX Layer (Tracers)'''

    def __init__(self):
        self.type = 'Tracers'
        self.history = []
        self.last_input = None
        self.last_output = None

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        self.history.append(frame)
        if len(self.history) > 60:
            self.history.remove(self.history[0])

        output = self.history[0]

        for i in range(len(self.history)):
            output = cv.addWeighted(output, 0.9, self.history[i], 0.1, 0)

        self.last_input = frame
        self.last_output = output
        return output


class Symmetry(Layer):
    '''Visual FX Layer (Symmetry)'''

    def __init__(self):
        self.type = 'Symmetry'
        self.mode = 0
        self.last_input = None
        self.last_output = None
        self.tooltips = ["Press 'M' to cycle through symmetry modes"]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        # Create flipped images
        symmetry = output.copy()
        symmetry_flipped_y = np.flip(symmetry, 0)
        symmetry_flipped_x = np.flip(symmetry, 1)
        symmetry_flipped_xy = np.flip(symmetry_flipped_y, 1)

        if self.mode == 0:
            # Tile flipped images and resize to maintain original shape
            output = np.zeros((height * 2, width * 2, channels), 'uint8')
            output[0:height, 0:width, :] = symmetry[:, :, :]
            output[height:height * 2, 0:width, :] = symmetry_flipped_y[:, :, :]
            output[0:height, width:width*2, :] = symmetry_flipped_x[:, :, :]
            output[height:height * 2, width:width *
                   2, :] = symmetry_flipped_xy[:, :, :]
            output = cv.resize(output, (width, height),
                               interpolation=cv.INTER_AREA)
        elif self.mode == 1:
            # Take a corner of flipped images and tile
            output = np.zeros_like(symmetry)
            output[0:height // 2, 0:width // 2,
                   :] = symmetry[:height // 2, :width // 2, :]
            output[height // 2:, 0:width // 2,
                   :] = symmetry_flipped_y[height // 2:, :width // 2, :]
            output[0:height // 2, width // 2:,
                   :] = symmetry_flipped_x[:height // 2, width // 2:, :]
            output[height // 2:, width // 2:,
                   :] = symmetry_flipped_xy[height // 2:, width // 2:, :]
        elif self.mode == 2:
            # Combine flipped images for overlaid symmetry
            c1 = cv.addWeighted(symmetry, 0.5, symmetry_flipped_y, 0.5, 0)
            c2 = cv.addWeighted(symmetry_flipped_x, 0.5,
                                symmetry_flipped_xy, 0.5, 0)
            output = cv.addWeighted(c1, 0.5, c2, 0.5, 0)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('m'):
            # Rotate through symmetry modes
            self.mode += 1

            if self.mode >= 3:
                self.mode = 0
