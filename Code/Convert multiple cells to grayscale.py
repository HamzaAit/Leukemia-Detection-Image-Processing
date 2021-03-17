from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import cv2
import numpy as np
from scipy import misc
import scipy.io as sio
from skimage import data
from skimage.color import rgb2gray

# im = Image.open("../Dataset/Multiple Cells/Blast/Im001_1.jpg")
im = cv2.imread("../Dataset/Multiple Cells/Blast/Original/Im001_1.jpg")

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
ax = axes.ravel()

ax[0].imshow(im)
ax[0].set_title("Original")

# gim = rgb2gray(im)
# gim *= 255
gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# plt.figure(0)
# plt.imshow(gim, cmap=plt.cm.gray)
# plt.show()

ax[1].imshow(gim, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

cv2.imwrite('../Dataset/Multiple Cells/Blast/Grayscale/Im001_1.png', gim)

thresh = 150

bimarr = np.copy(asarray(gim))

for i in range (bimarr.shape[0]):
    for j in range (bimarr.shape[1]):
        if(bimarr[i][j] > thresh):
            bimarr[i][j] = 255
        else:
            bimarr[i][j] = 0

bim = Image.fromarray(bimarr)


# plt.figure(1)
# plt.imshow(bim, 'gray')

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

cv2.imwrite('../Dataset/Multiple Cells/Blast/Threshold/Im001_1.png', bimarr)



