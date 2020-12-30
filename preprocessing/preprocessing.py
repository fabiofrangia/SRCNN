import cv2
import numpy as np
import os 

input_folder = 'Data/train_data/291/'
### Create label figure and input figire with low resolution
os.chdir('../')
print(os.getcwd())
try:
    os.makedirs('Data/input')
    os.makedirs('Data/output')
except Exception as e:
    print(e)
scale_percent = 25
for img in os.listdir(input_folder):
    img_input = cv2.imread(input_folder + img)
    width = int(img_input.shape[1] * scale_percent / 100)
    height = int(img_input.shape[0] * scale_percent / 100)

    width_or = int(img_input.shape[1] )
    height_or = int(img_input.shape[0])


    dim = (width, height)
    dim_original = (width*4, height*4)
    res = cv2.resize(img_input, dsize=dim, interpolation=cv2.INTER_AREA)
    res = cv2.resize(res, dsize=dim_original)
    img_input = cv2.resize(img_input, dsize=dim_original, interpolation=cv2.INTER_AREA)


    cv2.imwrite('Data/output/'+ str(img), img_input)
    cv2.imwrite('Data/input/'+ str(img), res)