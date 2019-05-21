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
                tooltips.append(layer.tooltip)
            except:
                continue

        return tooltips