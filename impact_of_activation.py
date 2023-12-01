# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread(r'E:\project\OpenAccess\Net\34\ves\DRIVE_0.png', 0)
print(img.max())
processedImg = np.exp(-np.abs(img / 255 - 0.5)) - np.exp(-0.5)
processedImg = processedImg * 255
processedImg = np.uint8(processedImg / processedImg.max() * 255)
# processedImg = processedImg - img
print(processedImg.max())
# cv2.imshow('img', processedImg)
# cv2.waitKey()
cv2.imwrite('D:/vc.png', processedImg)
