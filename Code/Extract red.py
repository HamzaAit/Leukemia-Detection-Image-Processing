from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import cv2
import numpy as np
from scipy import misc
import scipy.io as sio
from skimage import data
from skimage.color import rgb2gray

# Opening the original image
im = Image.open("../Dataset/Multiple Cells/Blast/Original/Im001_1.jpg")

red, green, blue = im.split()

blue.show()
