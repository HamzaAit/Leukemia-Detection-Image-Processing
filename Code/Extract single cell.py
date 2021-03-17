from PIL import Image
from numpy import asarray
import numpy as np

im = Image.open("../Dataset/Single Cell/Blast/Im001_1.tif")

imarr = np.copy(asarray(im))
newimarr = np.copy(asarray(im))

n = 257

newimarr.resize((257,257,4))
for i in range(n):
    for j in range(n):
        newimarr[i][j][0] = 255
        newimarr[i][j][1] = 255
        newimarr[i][j][2] = 255
        newimarr[i][j][3] = 0

for i in range(n):
    for j in range(n):
        if imarr[i][j][2] > 90 and (imarr[i][j][1] < 100 or imarr[i][j][0] < 100):
            newimarr[i][j][0] = imarr[i][j][0]
            newimarr[i][j][1] = imarr[i][j][1]
            newimarr[i][j][2] = imarr[i][j][2]
            imarr[i][j][2] = 0
            imarr[i][j][1] = 0
            imarr[i][j][0] = 0
            newimarr[i][j][3] = 255
            

blackcell = Image.fromarray(imarr)
isolatedcell = Image.fromarray(newimarr)
blackcell.show()
isolatedcell.show()
