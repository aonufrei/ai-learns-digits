from PIL import Image
import sys

from neural import NeuralNetwork

class Detector:

    __memory_file = 'weights.w'

    def __init__(self) -> None:
        self.brain = NeuralNetwork()
        self.brain.set_weights(self.__memory_file)

    def prepare_image(self, image_path):
        img = Image.open(image_path).resize((8,8))
        pxl = img.load()
        rows = []
        for x in range(8):
            for y in range(8):
                rows.append(pxl[x, y][3] / 255)
        return rows

    def detect_digit(self, image_path):
        dataset = self.prepare_image(image_path)
        return self.to_digit(self.brain.predict(dataset))
    
    def to_digit(self, value):
        constrains = [0, 1/3, 2/3, 1]
        distances = [abs(x - value) for x in constrains]
        
        smalles = distances[0]
        smalles_index = 0
        for d in range(1, len(distances)):
            if distances[d] < smalles:
                smalles = distances[d]
                smalles_index = d

        return smalles_index


detector = Detector()
params = sys.argv
if (len(params) != 2):
    print('Script requires 2 params (second parameter is the image filepath)')
else:
    print(detector.detect_digit(params[1]))