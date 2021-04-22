from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


# im = Image.open("../Dataset/Single Cell/Blast/Im001_1.tif")
im = cv2.imread("../Dataset/Single Cell/Blast/Im001_1.tif")


imarr = np.copy(asarray(im))
newimarr = np.copy(asarray(im))

n = 257

newimarr.resize((257,257,4))


for i in range(n):
    for j in range(n):
        newimarr[i][j][0] = imarr[i][j][0]
        newimarr[i][j][1] = imarr[i][j][1]
        newimarr[i][j][2] = imarr[i][j][2]
        newimarr[i][j][3] = 255

# Setting up the figures window
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
ax = axes.ravel()

# Adding the original image to the list of figures shown, titled "Original"
ax[0].imshow(im)
ax[0].set_title("Original")

gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ax[1].imshow(gim, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

# Converting the grayscale image to binary, then adding it to the list of figures under title "Binary"

thresh = 110

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

# Area opening done manually using a sliding window, which will check if any black pixels are surrounded by white pixels, if so, make them white

wl = 20
for i in range(n - wl):
    for j in range(n - wl):
        valid = True
        for k in range (i, i + wl):
            if (bimarr[k][j] == 0):
                valid = False
                break
            if (bimarr[k][j + wl] == 0):
                valid = False
                break
        if (valid == False):
            continue
        for k in range (j, j + wl):
            if (bimarr[i][k] == 0):
                valid = False
                break
            if (bimarr[i + wl][k] == 0):
                valid = False
                break
        if (valid == False):
            continue
        for x in range(i, i + wl):
            for y in range (j, j + wl):
                bimarr[x][y] = 255

cleanimg = Image.fromarray(bimarr)

cleanimg.show()

# cv2.imwrite('../Dataset/Single Cell/Im001_1.png', cleanimg)
cleanimg.save('../Dataset/Single Cell/Im001_1.png')

# Implementation of bounding box: Get coordinate of highest and lowest black pixels, and leftmost and rightmost black pixels

up = 0
down = 0
left = 0
right = 0


done = False
for i in range (n):
    if(done):
        break
    for j in range (n):
        if(bimarr[i][j] == 0):
            up = i
            done = True
            break

done = False
for i in range (n):
    if(done):
        break
    for j in range (n):
        if(bimarr[n - i - 1][j] == 0):
            down = n - i - 1
            done = True
            break

done = False
for j in range (n):
    if(done):
        break
    for i in range (n):
        if(bimarr[i][j] == 0):
            left = j
            done = True
            break

done = False
for j in range (n):
    if(done):
        break
    for i in range (n):
        if(bimarr[i][n - j - 1] == 0):
            right = n - j - 1
            done = True
            break

im = cv2.imread("../Dataset/Single Cell/Im001_1.png")

print(up,down,left,right)
startpoint = (left, up)
endpoint = (right, down)

rectim = cv2.rectangle(im, startpoint, endpoint, (255,0,0), 1)
rectim = Image.fromarray(rectim)
rectim.show()

# Hole filling

wl = 20
for i in range(n - wl):
    for j in range(n - wl):
        valid = True
        for k in range (i, i + wl):
            if (bimarr[k][j] == 255):
                valid = False
                break
            if (bimarr[k][j + wl] == 255):
                valid = False
                break
        if (valid == False):
            continue
        for k in range (j, j + wl):
            if (bimarr[i][k] == 255):
                valid = False
                break
            if (bimarr[i + wl][k] == 255):
                valid = False
                break
        if (valid == False):
            continue
        for x in range(i, i + wl):
            for y in range (j, j + wl):
                bimarr[x][y] = 0

cleanimg = Image.fromarray(bimarr)

cleanimg.show()

# Calculate area and perimeter and circularity

# Area is the number of black pixels
# Perimeter is the number of border black pixels, i.e. black pixels with a neighboring white pixel

Area = 0
Perimeter = 0

for i in range(up, down):
    for j in range(left, right):
        if(bimarr[i][j] == 0):
            Area += 1
            if(bimarr[i-1][j] == 255 or bimarr[i+1][j] == 255 or bimarr[i][j-1] == 255 or bimarr[i][j+1] == 255):
                Perimeter += 1
    

Circularity = 4 * math.pi * Area / (Perimeter * Perimeter)

print(Area, Perimeter, Circularity)