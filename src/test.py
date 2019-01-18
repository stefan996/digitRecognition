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
import matplotlib.pyplot as plt

# colored data in terminal
from termcolor import colored

# remove warnings
import warnings
import os


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
    print(colored("Prediction: %s" % prediction_result, 'green'))
     

 
def main():
    
    # remove warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore")
    
    # enter directory where we saved images
    image_path = raw_input(colored("\nEnter folder path: \n", 'red'))
    print(" ")
   
    # for image "num1.png" we get "num1Binarized.png"
    lastPoint = image_path.rfind('.')
    binarized_image_path = image_path[ : lastPoint] + "Binarized" + image_path[lastPoint : ]
    binarization_image(image_path, binarized_image_path)
    
    # open image on binarized_image_path
    with open(binarized_image_path, 'r') as f:
        with Image.open(f) as image:
            # show image
            print(colored('Processing image "{}".'.format(image_path), 'blue'))
            plt.imshow(image)
            plt.show()
            
            # change size of binarized image to 28x28 pixels
            resized_image = resizeimage.resize_cover(image, [28, 28])
            
            # prediction processed image
            prediction_image(resized_image)
            
            # remove binarized image from directory
            os.remove(binarized_image_path)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    