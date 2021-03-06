class Stack:
    '''Visual FX Stack'''

    def __init__(self, layers=[]):
        self.layers = layers

    def apply(self, frame):
        '''Return the frame with the fx layers applied.'''
        output = frame.copy()

        for layer in self.layers:
            if layer.active:
                output = layer.apply(output)

        return output

    def userInput(self, key):
        '''Pass the user input to the fx layers.'''
        for i in range(len(self.layers)):
            if key == ord(str(i + 1)):
                self.layers[i].active = not self.layers[i].active

        for layer in self.layers:
            if layer.active:
                layer.userInput(key)

    def getTooltips(self):
        tooltips = ["Press 'Q' to quit",
                    "Press 'R' to record",
                    "Press 'I' to take a snapshot"]
        for layer in self.layers:
            if layer.active:
                try:
                    for tooltip in layer.tooltips:
                        if tooltip not in tooltips:
                            tooltips.append(tooltip)
                except:
                    continue

        return tooltips

    def getReadouts(self):
        readouts = []
        for layer in self.layers:
            if layer.active:
                try:
                    for readout in layer.readouts:
                        if readout not in readouts:
                            readouts.append(readout)
                except:
                    continue

        return readouts

    def getLayerNames(self):
        layerNames = []
        for l in range(len(self.layers)):
            if self.layers[l].active:
                status = "ON"
            else:
                status = "OFF"

            layerNames.append(
                "{}-{} {}".format(l + 1, self.layers[l].type, status))

        return layerNames
