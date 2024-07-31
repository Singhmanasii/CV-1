import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("C:/Users/singh/OneDrive/Pictures/cherry blossom.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols, channels = img_rgb.shape
ptsl= np.float32([[133,34], [226,16], [133,206], [226,219]])
pts2= np.float32([[0,0],[300,0], [0,300],[300,300]])
M = cv2.getPerspectiveTransform(ptsl,pts2)
dst= cv2.warpPerspective (img, M, (300,300))
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original Image')
plt.subplot(122), plt.imshow(dst), plt.title('Output Image')
plt.show()
