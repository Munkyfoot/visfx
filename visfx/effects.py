import numpy as np
import cv2 as cv
import os
import sys


class Layer:
    '''Visual FX Layer (Empty)'''

    def __init__(self, start_active=False):
        self.type = None
        self.active = start_active
        self.last_input = None
        self.last_output = None
        self.tooltips = []
        self.readouts = []

    def apply(self, frame):
        return frame

    def userInput(self, key):
        pass


class Ghost(Layer):
    '''Visual FX Layer (Ghost)'''

    def __init__(self):
        super().__init__()
        self.type = 'Ghost'
        self.background = None
        self.tooltips = ["Press 'B' to set the background",
                         "Press 'C' to clear background"]

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

        if key == ord('c'):
            self.background = None


class RemoveBG(Layer):
    '''Visual FX Layer (Remove BG)'''

    def __init__(self):
        super().__init__()
        self.type = 'Remove BG'
        self.background = None
        self.show_bg = False
        self.threshold = 0.1
        self.history = []
        self.tooltips = ["Press 'B' to set the background",
                         "Press 'C' to clear background",
                         "Press 'V' to toggle background",
                         "Press 'T' to adjust BG detection threshold"]
        self.readouts = ["BG Detection Threshold:{}".format(self.threshold)]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        self.history.append(frame)
        if len(self.history) > 60:
            self.history.remove(self.history[0])

        if self.background is not None:
            mask = np.zeros_like(output)
            mask[cv.absdiff(output, self.background) >
                 255 * self.threshold] = 255

            flatmask = np.zeros((height, width), output.dtype)
            flatmask = cv.max(mask[:, :, 0], mask[:, :, 1])
            flatmask = cv.max(flatmask, mask[:, :, 2])

            if self.show_bg:
                output[flatmask == 0] = self.background[flatmask == 0]
            else:
                output[flatmask == 0] = 0

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('b'):
            average = self.history[-1]
            for frame in self.history[:-1]:
                average = cv.addWeighted(average, 0.5, frame, 0.5, 1)
            self.background = average

        if key == ord('c'):
            self.background = None

        if key == ord('t'):
            self.threshold += 0.1
            if self.threshold > 1:
                self.threshold = 0.1
            self.readouts[0] = "BG Detection Threshold:{}".format(
                self.threshold)

        if key == ord('v'):
            self.show_bg = not self.show_bg


class Tracers(Layer):
    '''Visual FX Layer (Tracers)'''

    def __init__(self):
        super().__init__()
        self.type = 'Tracers'
        self.history = []
        self.duration = 1
        self.tooltips = ["Press 'D' to change tracer duration"]
        self.readouts = ["Tracer Duration:{}".format(
            self.duration)]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        self.history.append(frame)
        if len(self.history) > 60:
            self.history.remove(self.history[0])

        output = self.history[0]

        for i in range(len(self.history)):
            output = cv.addWeighted(
                output, 0.5 + self.duration * 0.1, self.history[i], 0.5 - self.duration * 0.1, 1)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('d'):
            self.duration += 1
            if self.duration > 4:
                self.duration = 1
            self.readouts[0] = "Tracer Duration:{}".format(
                self.duration)


class Denoise(Layer):
    '''Visual FX Layer (Denoise)'''

    def __init__(self):
        super().__init__()
        self.type = 'Denoise'
        self.strength = 1
        self.tooltips = ["Press 'S' to change denoise strength"]
        self.readouts = ["Denoise Strength:{}".format(
            self.strength)]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        try:
            output = cv.addWeighted(frame, 0.5, self.last_input, 0.5, 1)
        except:
            pass

        if self.strength == 2:
            output = cv.bilateralFilter(output, 3, 45, 45)
        elif self.strength == 3:
            output = cv.bilateralFilter(output, 5, 60, 60)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('s'):
            self.strength += 1
            if self.strength > 3:
                self.strength = 1
            self.readouts[0] = "Denoise Strength:{}".format(
                self.strength)


class Symmetry(Layer):
    '''Visual FX Layer (Symmetry)'''

    def __init__(self):
        super().__init__()
        self.type = 'Symmetry'
        self.mode = 0
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


class FaceDetect(Layer):
    '''Visual FX Layer (Face Detection)'''

    def __init__(self):
        super().__init__()
        self.type = 'Face Detection'
        self.face_cascade = cv.CascadeClassifier(os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'cascades', 'standard', 'haarcascade_frontalface_default.xml'))
        self.pixelize = False
        self.tooltips = ["Press 'P' to pixelize faces"]
        self.readouts = ["Pixelize Faces: {}".format(str(self.pixelize))]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            (x, y, w, h) = (max(0, x - (w // 8)), max(0, y - (h // 8)),
                            min(width, w + (w // 4)), min(height, h + (h // 4)))
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = output[y:y+h, x:x+w]

            if self.pixelize:
                pixelized = cv.resize(roi_color, (w // 16, h // 16))
                pixelized = cv.resize(
                    pixelized, (w, h), interpolation=cv.INTER_NEAREST)
                output[y:y+h, x:x+w] = pixelized
            else:
                cv.rectangle(output, (x, y), (x+w, y+h), (200, 200, 200), 2)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('p'):
            self.pixelize = not self.pixelize
            self.readouts[0] = "Pixelize Faces: {}".format(str(self.pixelize))


class ColorFilter(Layer):
    '''Visual FX Layer (Color Filter)'''

    def __init__(self):
        super().__init__()
        self.type = 'Color Filter'
        self.filters = ['Grayscale', 'Sepia', 'Invert',
                        'Red Pass', 'Green Pass', 'Blue Pass',
                        'Two Tone']
        self.filter_id = 0
        self.tooltips = ["Press 'F' to change color filter"]
        self.readouts = ["Color Filter: {}".format(
            self.filters[self.filter_id])]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        grayscale = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        color_filter = self.filters[self.filter_id]

        if color_filter == 'Grayscale':
            output = grayscale
        if color_filter == 'Sepia':
            sepia = np.array(
                [
                    [0.272, 0.534, 0.131],
                    [0.349, 0.686, 0.168],
                    [0.393, 0.769, 0.189]
                ]
            )

            output = cv.transform(output, sepia)
        elif color_filter == 'Invert':
            output = 255-output
        elif color_filter == 'Red Pass':
            hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

            lower = np.array([0, 32, 16])
            upper = np.array([9, 255, 255])

            lower_a = np.array([169, 32, 16])
            upper_b = np.array([179, 255, 255])

            mask = cv.inRange(hsv, lower, upper)
            mask = cv.addWeighted(mask, 1.0, cv.inRange(
                hsv, lower, upper), 1.0, 0.0)

            output[mask == 0] = grayscale[mask == 0]
        elif color_filter == 'Green Pass':
            hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

            lower = np.array([40, 32, 16])
            upper = np.array([88, 255, 255])

            mask = cv.inRange(hsv, lower, upper)

            output[mask == 0] = grayscale[mask == 0]
        elif color_filter == 'Blue Pass':
            hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

            lower = np.array([92, 32, 16])
            upper = np.array([128, 255, 255])

            mask = cv.inRange(hsv, lower, upper)

            output[mask == 0] = grayscale[mask == 0]
        elif color_filter == 'Two Tone':
            lower = np.array([128, 128, 128])
            upper = np.array([255, 255, 255])

            mask = cv.inRange(grayscale, lower, upper)
            mask = cv.medianBlur(mask, 3)

            output = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('f'):
            self.filter_id += 1
            if self.filter_id >= len(self.filters):
                self.filter_id = 0
            self.readouts[0] = "Color Filter: {}".format(
                self.filters[self.filter_id])