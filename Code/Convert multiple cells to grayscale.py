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
im = cv2.imread("../Dataset/Multiple Cells/Blast/Original/Im001_1.jpg")

# Setting up the figures window
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
ax = axes.ravel()

# Adding the original image to the list of figures shown, titled "Original"
ax[0].imshow(im)
ax[0].set_title("Original")

# Converting the image to grayscale, then adding it to the list of figures titled "Grayscale"

# gim = rgb2gray(im)
# gim *= 255
gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ax[1].imshow(gim, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

# We save the grayscale image under a folder where we will keep the grayscale versions of the dataset
cv2.imwrite('../Dataset/Multiple Cells/Blast/Grayscale/Im001_1.png', gim)

# Converting the grayscale image to binary, then adding it to the list of figures under title "Binary"

thresh = 150

bimarr = np.copy(asarray(gim))

for i in range (bimarr.shape[0]):
    for j in range (bimarr.shape[1]):
        if(bimarr[i][j] > thresh):
            bimarr[i][j] = 255
        else:
            bimarr[i][j] = 0

bim = Image.fromarray(bimarr)


ax[2].imshow(bim, 'gray')
ax[2].set_title("Binary")
plt.show()

# th2 = cv2.adaptiveThreshold(gim,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# bim = cv2.GaussianBlur(gim, (9, 9), 0)
# bim = cv2.adaptiveThreshold(gim, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# ret, bim = cv2.threshold(gim,100,255,cv2.THRESH_BINARY)

# plt.figure(2)
# plt.imshow(bim, 'gray')
# plt.show()

# Saving the binary image under a folder called Threshold, where we will be keeping the binary version of the dataset
cv2.imwrite('../Dataset/Multiple Cells/Blast/Threshold/Im001_1.png', bimarr)



