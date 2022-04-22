from multiprocessing.spawn import prepare
from re import X
from PIL import Image
import glob
import os
import random

from neural import NeuralNetwork

def rgb_to_grayscale(color):
    return (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114) / 255

def prepare_image(path):
    img = Image.open(path)
    pxl = img.load()
    rows = []
    for x in range(8):
        for y in range(8):
            rows.append(pxl[x, y][3] / 255)

    return rows
    

def prepare_data(folder):
    prepared_data = []

    for img in glob.glob(f'{folder}\\*one*.png'):
        img = Image.open(img)
        pxl = img.load()
        rows = []
        for x in range(8):
            for y in range(8):
                rows.append(pxl[x, y][3] / 255)
        
        prepared_data.append((rows, 1))

    for img in glob.glob(f'{folder}\\*zero*.png'):
        img = Image.open(img)
        pxl = img.load()
        rows = []
        for x in range(8):
            for y in range(8):
                rows.append(pxl[x, y][3] / 255)
        
        prepared_data.append((rows, 0))

    return prepared_data



# print(prepare_data('dataset_resized'))

if __name__ == '__main__':
    # dataset = prepare_data('dataset_resized')
    # my_zero = prepare_image('converted_my_zero.png')
    ai = NeuralNetwork()
    ai.set_weights('weights.w')
    # epoch = 120000
    # for x in range(epoch):
    #     ai.train(random.choice(dataset), 0.01)
    
    # ai.output_weights()
    prediction_dataset = prepare_data('test')
    data = prediction_dataset[0]
    for x in prediction_dataset:
        result = ai.predict(x)
        print(f"actual {1 if result > 0.5 else 0}, expected {x[1]}")
    # result = ai.predict((my_zero, 0))
    # print(f"actual {1 if result > 0.5 else 0}, expected {0}")
    