# build model and other function for neural network
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

# image processing 
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from scipy.misc import imread, imsave

# show the image
import matplotlib.pyplot as plt


# accessing images in directory 
import imghdr
import os 

# colored data in terminal
from termcolor import colored

# remove warnings
import warnings 


# return number of images in path directory 
def num_images(path):
    numberOfImages = 0    
    for root, dirs, files in os.walk(path):
        for name in files:
            lastPoint = name.rfind('.')     # index of last character '.'
            exstension = name[lastPoint : ]     # extension of file
            if exstension == ".png" or exstension == "jpg" or exstension == ".jpeg":
                numberOfImages = numberOfImages + 1
    return numberOfImages



def binarization_image(original_image_path, binarized_image_path):
        
    # read original image
    img = imread(original_image_path, mode = 'L')

    # specify a threshold 0-255
    threshold = 100

    # make all pixels < threshold black
    binarized = 1.0 * (img < threshold)
    
    # save the binarized image
    imsave(binarized_image_path, binarized)
    
        

def prediction_image(image):
    
    # load saved model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model('../model/my_model.h5')

    # expand dimension of picture
    image = (np.expand_dims(image,0))
    
    # prediction image
    predictions_single = model.predict(image)
    prediction_result = np.argmax(predictions_single[0])
    print(colored("Prediction: %s \n\n" % prediction_result, 'green'))
     


def main():
    
    # remove warnings
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # enter directory where we saved images
    directory_path = raw_input(colored("\nEnter folder path: \n", 'red'))
    directory_path = directory_path + "/"
    print(" ")
   
    # numbers of images in directory_path
    img_num = num_images(directory_path)
    print(colored('Number of images in directory "{}" is: {}\n\n'.format(directory_path, img_num), 'yellow'))
    
    # processing all image from directory
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            
            # define the path name
            original_image_path = directory_path + name
            binarized_image_path = directory_path + "1" + name
            
            # binarized image saved on path "directory_path + 1 + name"
            binarization_image(original_image_path, binarized_image_path)
            
            # open image on binarized_image_path
            with open(binarized_image_path, 'r') as f:
                with Image.open(f) as image:
                    print(colored('Processing image "{}".'.format(original_image_path), 'blue'))
                    
                    # change size of binarized image to 28x28 pixels
                    resized_image = resizeimage.resize_cover(image, [28, 28])
                    
                    #print("Show image on path %s." % binarized_image_path)
                    #plt.imshow(resized_image)
                    #plt.show()
                    
                    # prediction processed image
                    prediction_image(resized_image)
                    
                    # remove binarized image from directory
                    os.remove(binarized_image_path)
            
            
            
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    