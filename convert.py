from PIL import Image
import glob
import os

def convert_images():
    out_dir = 'dataset_resized'
    cnt = 0
    for img in glob.glob('dataset\\one*.png'):
        Image.open(img).resize((8,8)).save(os.path.join(out_dir, f'one{cnt}.png'))
        cnt += 1

    cnt = 0
    for img in glob.glob('dataset\\zero*.png'):
        Image.open(img).resize((8,8)).save(os.path.join(out_dir, f'zero{cnt}.png'))
        cnt += 1

def convert_single(path, out):
    Image.open(path).resize((8,8)).save(out)

if __name__ == "__main__":
    convert_single(".\\test_zero_1.png", ".\\test\\test_zero_1.png")