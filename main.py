from PIL import Image
import glob
import os
import random
import sys
import numpy as np

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
    

def prepare_data(path_expected_list):
    prepared_data = []
    for pair in path_expected_list:
        prepared_data.extend(fetch_data_by_pattern(pair[0], pair[1]))
    return prepared_data

def fetch_data_by_pattern(pattern, expected):
    result = []
    for img in glob.glob(f'{pattern}'):
        img = Image.open(img)
        pxl = img.load()
        rows = []
        for x in range(8):
            for y in range(8):
                rows.append(pxl[x, y][3] / 255)
        
        result.append((rows, expected))
    return result

# print(prepare_data('dataset_resized'))

def define_number(value):
    constrains = [1/8, 3/8, 5/8, 7/8]
    distances = [abs(x - value) for x in constrains]
    
    smalles = distances[0]
    smalles_index = 0
    for d in range(1, len(distances)):
        if distances[d] < smalles:
            smalles = distances[d]
            smalles_index = d

    return smalles_index

def training_loss(actual_values, correct_values): 
    assert len(actual_values) == len(correct_values)
    return np.mean((np.array(actual_values).T - np.array(correct_values)) ** 2)

if __name__ == '__main__':
    dataset = prepare_data([('dataset_resized\\*zero*.png', 1/8), ('dataset_resized\\*one*.png', 3/8), ('dataset_resized\\*two*.png', 5/8), ('dataset_resized\\*three*.png', 7/8)])
    # my_zero = prepare_image('converted_my_zero.png')
    ai = NeuralNetwork()
    # ai.set_weights('weights.w')
    epoch = 300000
    last = []
    counter = 0
    # for x in range(epoch):
    max_of_ten = 1
    while counter < epoch and max_of_ten >= 0.01:
        dif = ai.train(random.choice(dataset), 0.1)
        last.append(abs(dif))
        
        if (len(last) > 100):
            last = last[-10:]

        max_of_ten = 1 if len(last[-10:]) < 10 else max(last[-10:])
        # correct_results = [x[1] for x in dataset]
        # actual_results = [ai.predict(x[0]) for x in dataset]
        # sys.stdout.write(f'\r Progress: {x + 1}/{epoch}; Training loss: {training_loss(actual_results, correct_results)}')
        sys.stdout.write(f'\r Epoch: {counter + 1}; error: {max_of_ten}')
        counter += 1
    
    print()
    print(last[-10:])   
    ai.output_weights('weights.w')
    # prediction_dataset = prepare_data([('test\\test_zero*.png', 0), ('test\\test_one*.png', 1), ('test\\test_two*.png', 2), ('test\\test_three*.png', 3)])
    prediction_dataset = dataset
    for x in prediction_dataset:
        result = ai.predict(x[0])
        print(f"actual {define_number(result)}, expected {define_number(x[1])}")
    # result = ai.predict((my_zero, 0))
    # print(f"actual {1 if result > 0.5 else 0}, expected {0}")