from PIL import Image
import cv2
from numpy import asarray
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread("CLAHE4.png")
# im = cv2.imread("../Dataset/Multiple Cells/Blast/Original/Im001_1.jpg")
# im = Image.open("/CLAHE initial.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imwrite('clahe before.png', im)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

histbefore = cv2.calcHist([im], [0], None, [256],[0,256])

plt.plot(histbefore); plt.show()

imarr = np.copy(asarray(im))

imarr = clahe.apply(imarr)

im = Image.fromarray(imarr)

im.save("clahe after.png")

im = cv2.imread("clahe after.png")

histafter = cv2.calcHist([im], [0], None, [256],[0,256])

plt.plot(histafter); plt.show()

