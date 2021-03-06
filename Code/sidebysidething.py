from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math



path = "../Dataset/Single Cell/Blast/Im005_1.tif"

im = cv2.imread(path)

im = Image.open(path)
print(path)

imarr = np.copy(asarray(im.split()[0]))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

imarr = clahe.apply(imarr)

thresh = 130

n = len(imarr)

m = len(imarr[0])

for i in range (imarr.shape[0]):
    for j in range (imarr.shape[1]):
        if(imarr[i][j] > thresh):
            imarr[i][j] = 255
        else:
            imarr[i][j] = 0


bimg = Image.fromarray(imarr)

bimg.show()

wl = 35
for i in range(n - wl):
    for j in range(m - wl):
        valid = True
        for k in range (i, i + wl):
            if (imarr[k][j] == 255):
                valid = False
                break
            if (imarr[k][j + wl] == 255):
                valid = False
                break
        if (valid == False):
            continue
        for k in range (j, j + wl):
            if (imarr[i][k] == 255):
                valid = False
                break
            if (imarr[i + wl][k] == 255):
                valid = False
                break
        if (valid == False):
            continue
        for x in range(i, i + wl):
            for y in range (j, j + wl):
                imarr[x][y] = 0


for i in range(n):
    for j in range(m):
        if(imarr[i][j] == 0):
            imarr[i][j] = 255
        else:
            imarr[i][j] = 0




imarr = cv2.morphologyEx(imarr, cv2.MORPH_CLOSE, (10,10), iterations=3)

wl = 40
for i in range(n - wl):
    for j in range(m - wl):
        valid = True
        for k in range (i, i + wl):
            if (imarr[k][j] == 0 and i < n - wl):
                valid = False
                break
            if (imarr[k][j + wl] == 0 and i < n - wl):
                valid = False
                break
        if (valid == False):
            continue
        for k in range (j, j + wl):
            if (imarr[i][k] == 0 and j < n - wl):
                valid = False
                break
            if (imarr[i + wl][k] == 0 and j < n - wl):
                valid = False
                break
        if (j == 0 or j == n-1 or j + wl == m - 1):
            valid = True
        if (i == 0 or i == n-1 or i + wl >= n - 1):
            valid = True
        if (valid == False):
            continue
        for x in range(i, i + wl):
            for y in range (j, j + wl):
                imarr[x][y] = 255


# def baseCase(i, j, visited):
#     return (i>=n or j >= m or i < 0 or j < 0 or (i,j) in visited)


# def areaCalc(imarr, i, j, visited):
#     if(baseCase(i, j, visited) or imarr[i][j] == 0):
#         return 0
#     visited.append((i,j))
#     return 1 + areaCalc(imarr, i-1, j, visited) + areaCalc(imarr, i+1, j, visited) + areaCalc(imarr, i, j-1, visited) + areaCalc(imarr, i, j+1, visited)

# visited = []



imarr = cv2.morphologyEx(imarr, cv2.MORPH_OPEN, (10,10), iterations=5)

cleanimg = Image.fromarray(imarr)

cleanimg.show()

# print(areaCalc(imarr, n//2, m//2, visited))

def getArea(contours):
    maxArea = 0
    idx = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            maxArea = area
            idx = i
    maxPerimeter = len(np.array(contours[idx]))
    return maxArea, maxPerimeter

# image_src = cv2.imread("input.png")
# imarr = cv2.cvtColor(imarr, cv2.COLOR_BGR2GRAY)
ret, res = cv2.threshold(imarr, 250,255,0)

contours = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

print(getArea(contours))

area, perimeter = getArea(contours)

circularity = 4 * math.pi * area / (perimeter * perimeter)

print("Blast Cell: ", area, perimeter, circularity)

path = "../Dataset/Single Cell/Healthy/Im157_0.tif"

im = cv2.imread(path)

im = Image.open(path)
print(path)

imarr = np.copy(asarray(im.split()[0]))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

imarr = clahe.apply(imarr)

thresh = 130

n = len(imarr)

m = len(imarr[0])

for i in range (imarr.shape[0]):
    for j in range (imarr.shape[1]):
        if(imarr[i][j] > thresh):
            imarr[i][j] = 255
        else:
            imarr[i][j] = 0


bimg = Image.fromarray(imarr)

bimg.show()

wl = 35
for i in range(n - wl):
    for j in range(m - wl):
        valid = True
        for k in range (i, i + wl):
            if (imarr[k][j] == 255):
                valid = False
                break
            if (imarr[k][j + wl] == 255):
                valid = False
                break
        if (valid == False):
            continue
        for k in range (j, j + wl):
            if (imarr[i][k] == 255):
                valid = False
                break
            if (imarr[i + wl][k] == 255):
                valid = False
                break
        if (valid == False):
            continue
        for x in range(i, i + wl):
            for y in range (j, j + wl):
                imarr[x][y] = 0


for i in range(n):
    for j in range(m):
        if(imarr[i][j] == 0):
            imarr[i][j] = 255
        else:
            imarr[i][j] = 0




imarr = cv2.morphologyEx(imarr, cv2.MORPH_CLOSE, (10,10), iterations=3)

wl = 40
for i in range(n - wl):
    for j in range(m - wl):
        valid = True
        for k in range (i, i + wl):
            if (imarr[k][j] == 0 and i < n - wl):
                valid = False
                break
            if (imarr[k][j + wl] == 0 and i < n - wl):
                valid = False
                break
        if (valid == False):
            continue
        for k in range (j, j + wl):
            if (imarr[i][k] == 0 and j < n - wl):
                valid = False
                break
            if (imarr[i + wl][k] == 0 and j < n - wl):
                valid = False
                break
        if (j == 0 or j == n-1 or j + wl == m - 1):
            valid = True
        if (i == 0 or i == n-1 or i + wl >= n - 1):
            valid = True
        if (valid == False):
            continue
        for x in range(i, i + wl):
            for y in range (j, j + wl):
                imarr[x][y] = 255


# def baseCase(i, j, visited):
#     return (i>=n or j >= m or i < 0 or j < 0 or (i,j) in visited)


# def areaCalc(imarr, i, j, visited):
#     if(baseCase(i, j, visited) or imarr[i][j] == 0):
#         return 0
#     visited.append((i,j))
#     return 1 + areaCalc(imarr, i-1, j, visited) + areaCalc(imarr, i+1, j, visited) + areaCalc(imarr, i, j-1, visited) + areaCalc(imarr, i, j+1, visited)

# visited = []



imarr = cv2.morphologyEx(imarr, cv2.MORPH_OPEN, (10,10), iterations=5)

cleanimg = Image.fromarray(imarr)

cleanimg.show()

# print(areaCalc(imarr, n//2, m//2, visited))


# image_src = cv2.imread("input.png")
# imarr = cv2.cvtColor(imarr, cv2.COLOR_BGR2GRAY)
ret, res = cv2.threshold(imarr, 250,255,0)

contours = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

print(getArea(contours))

area, perimeter = getArea(contours)

circularity = 4 * math.pi * area / (perimeter * perimeter)

print("Healthy Cell: ", area, perimeter, circularity)