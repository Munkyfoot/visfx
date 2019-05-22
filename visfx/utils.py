class Stack:
    '''Visual FX Stack'''

    def __init__(self, layers=[]):
        self.layers = layers

    def apply(self, frame):
        '''Return the frame with the fx layers applied.'''
        output = frame.copy()

        for layer in self.layers:
            output = layer.apply(output)

        return output

    def userInput(self, key):
        '''Pass the user input to the fx layers.'''
        for layer in self.layers:
            layer.userInput(key)

    def getTooltips(self):
        tooltips = []
        for layer in self.layers:
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
            try:
                for readout in layer.readouts:
                    if readout not in readouts:
                        readouts.append(readout)
            except:
                continue

        return readouts

    def getLayerNames(self):
        layerNames = ""
        for l in range(len(self.layers)):
            if l == 0:
                layerNames += self.layers[l].type
            else:
                layerNames += " | {}".format(self.layers[l].type)
                

        return layerNames