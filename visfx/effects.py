import numpy as np
import cv2 as cv
import dlib
import os
import sys
import time


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


class MoveDetect(Layer):
    '''Visual FX Layer (Movement Detection)'''

    def __init__(self):
        super().__init__()
        self.type = 'Movement Detection'

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        try:
            output = cv.absdiff(self.last_input, frame)
        except:
            pass

        self.last_input = frame
        self.last_output = output
        return output


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
        self.threshold = 32
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
        if len(self.history) > 12:
            self.history.remove(self.history[0])

        if self.background is not None:
            mask = cv.absdiff(frame, self.background)

            diff = np.zeros_like(mask[:, :, 0])
            diff = cv.max(diff, mask[:, :, 0])
            diff = cv.max(diff, mask[:, :, 1])
            diff = cv.max(diff, mask[:, :, 2])
            diff[diff < self.threshold] = 0
            diff = cv.min(1, diff / (self.threshold + 32))
            diff = cv.blur(diff, (3, 3))

            if self.show_bg:
                output[:, :, 0] = (self.background[:, :, 0] * (1 - diff)) + (
                    output[:, :, 0] * diff)
                output[:, :, 1] = (self.background[:, :, 1] * (1 - diff)) + (
                    output[:, :, 1] * diff)
                output[:, :, 2] = (self.background[:, :, 2] * (1 - diff)) + (
                    output[:, :, 2] * diff)
            else:
                output[:, :, 0] = output[:, :, 0] * \
                    diff
                output[:, :, 1] = output[:, :, 1] * \
                    diff
                output[:, :, 2] = output[:, :, 2] * \
                    diff

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
            self.threshold += 4
            if self.threshold > 255:
                self.threshold = 4
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
            output = cv.bilateralFilter(output, 5, 45, 45)
        elif self.strength == 3:
            output = cv.bilateralFilter(output, 9, 60, 60)

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
        self.method = 0
        self.model_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'face-detection', 'opencv', 'opencv_face_detector_uint8.pb')
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'face-detection', 'opencv', 'opencv_face_detector.pbtxt')
        self.network = cv.dnn.readNetFromTensorflow(
            self.model_file, self.config_file)
        self.conf_threshold = 0.67
        self.detect_facemarks = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'face-detection', 'dlib', 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(self.predictor_file)
        self.pixelize = False
        self.tooltips = ["Press 'N' to change detection method",
                         "Press 'P' to pixelize faces",
                         "Press 'L' to detect facial landmarks"]
        self.readouts = ["Face Detection Method: {}".format(["dlib HoG", "OpenCV DNN"][self.method]),
                         "Pixelize Faces: {}".format(str(self.pixelize)),
                         "Detect Facemarks: {}".format(
                             str(self.detect_facemarks))]

    def apply(self, frame):
        output = frame.copy()
        height, width, channels = output.shape

        if self.method == 0:
            rects = self.detector(output, 0)

            for i, rect in enumerate(rects):
                x1 = rect.left()
                x2 = rect.right()
                y1 = rect.top()
                y2 = rect.bottom()
                w = x2 - x1
                h = y2 - y1
                roi = output[y1:y2, x1:x2]

                if self.detect_facemarks:
                    shape = self.predictor(output, rect)
                    points = np.zeros((68, 2), 'int32')
                    for pointIndex in range(68):
                        points[pointIndex] = (shape.part(
                            pointIndex).x, shape.part(pointIndex).y)
                    subdiv = cv.Subdiv2D((0, 0, width, height))
                    for point in points:
                        subdiv.insert((point[0], point[1]))
                    triangles = subdiv.getTriangleList()
                    hullIndex = cv.convexHull(points[:27], returnPoints=False)

                if self.pixelize:
                    pixelized = cv.resize(
                        roi, (w // 16, h // 16))
                    pixelized = cv.resize(
                        pixelized, (w, h), interpolation=cv.INTER_NEAREST)
                    output[y1:y2, x1:x2] = pixelized
                else:
                    cv.rectangle(output, (x1, y1), (x2, y2),
                                 (232, 64, 64), int(round(height/500)), 8)

                if self.detect_facemarks:
                    for t in triangles:
                        p1 = (t[0], t[1])
                        p2 = (t[2], t[3])
                        p3 = (t[4], t[5])

                        line_color = (64,200,64)

                        cv.line(output, p1, p2, line_color,
                                1, cv.LINE_AA, 0)
                        cv.line(output, p2, p3, line_color,
                                1, cv.LINE_AA, 0)
                        cv.line(output, p1, p3, line_color,
                                1, cv.LINE_AA, 0)
                    
                    for (x, y) in points:
                        cv.circle(output, (x, y), 1, (200,200,200), -1)
        else:
            blob = cv.dnn.blobFromImage(output, 1.0, (300, 300), [
                104, 117, 123], False, False)

            self.network.setInput(blob)
            faces = self.network.forward()
            for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    x1 = int(faces[0, 0, i, 3] * width)
                    y1 = int(faces[0, 0, i, 4] * height)
                    x2 = int(faces[0, 0, i, 5] * width)
                    y2 = int(faces[0, 0, i, 6] * height)

                    if self.pixelize:
                        w = x2 - x1
                        h = y2 - y1
                        y1_padded = max(0, y1 - h // 8)
                        y2_padded = min(height, y2 + h // 8)
                        x1_padded = max(0, x1 - w // 8)
                        x2_padded = min(width, x2 + w // 8)
                        w_padded = x2_padded - x1_padded
                        h_padded = y2_padded - y1_padded
                        roi = output[y1_padded:y2_padded, x1_padded:x2_padded]
                        pixelized = cv.resize(
                            roi, (w_padded // 16, h_padded // 16))
                        pixelized = cv.resize(
                            pixelized, (w_padded, h_padded), interpolation=cv.INTER_NEAREST)
                        output[y1_padded:y2_padded,
                               x1_padded:x2_padded] = pixelized
                    else:
                        conf_readout = "{:.2f}%".format(confidence * 100)
                        cv.putText(output, conf_readout, (x1, y1 - 5),
                                   cv.FONT_HERSHEY_SIMPLEX, height / 1500, (64, 232, 64), 1, cv.LINE_AA)
                        cv.rectangle(output, (x1, y1), (x2, y2),
                                     (232, 64, 64), int(round(height/500)), 8)

        self.last_input = frame
        self.last_output = output
        return output

    def userInput(self, key):
        if key == ord('n'):
            if self.method == 0:
                self.method = 1
                del self.readouts[2]
            else:
                self.method = 0
                self.readouts.append("Detect Facemarks: {}".format(
                    str(self.detect_facemarks)))

            self.readouts[0] = "Face Detection Method: {}".format(
                ["dlib HoG", "OpenCV DNN"][self.method])

        if key == ord('l') and self.method == 0:
            self.detect_facemarks = not self.detect_facemarks
            self.readouts[2] = "Detect Facemarks: {}".format(
                str(self.detect_facemarks))

        if key == ord('p'):
            self.pixelize = not self.pixelize
            self.readouts[1] = "Pixelize Faces: {}".format(str(self.pixelize))


class ColorFilter(Layer):
    '''Visual FX Layer (Color Filter)'''

    def __init__(self):
        super().__init__()
        self.type = 'Color Filter'
        self.filters = ['Grayscale', 'Sepia', 'Invert',
                        'Red Pass', 'Green Pass', 'Blue Pass',
                        'Two Tone', 'Colormap']
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
            hsv = cv.cvtColor(~output, cv.COLOR_BGR2HSV)

            lower = np.array([80, 48, 16])
            upper = np.array([100, 255, 255])

            mask = cv.inRange(hsv, lower, upper)

            output[mask == 0] = grayscale[mask == 0]
        elif color_filter == 'Green Pass':
            hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

            lower = np.array([40, 48, 16])
            upper = np.array([88, 255, 255])

            mask = cv.inRange(hsv, lower, upper)

            output[mask == 0] = grayscale[mask == 0]
        elif color_filter == 'Blue Pass':
            hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

            lower = np.array([92, 58, 16])
            upper = np.array([128, 255, 255])

            mask = cv.inRange(hsv, lower, upper)

            output[mask == 0] = grayscale[mask == 0]
        elif color_filter == 'Two Tone':
            lower = np.array([128, 128, 128])
            upper = np.array([255, 255, 255])

            mask = cv.inRange(grayscale, lower, upper)
            mask = cv.medianBlur(mask, 3)

            output = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        elif color_filter == 'Colormap':
            hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

            output[:, :, 0] = 255 - hsv[:, :, 2]
            output[:, :, 1] = hsv[:, :, 1]
            output[:, :, 2] = hsv[:, :, 2]

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
