import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/singh/OneDrive/Pictures/cherry blossom.jpg")
img_rqb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols, channels = img_rqb.shape
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img_rqb, M, (cols, rows))

fig, axs = plt.subplots(1, 2, figsize=(7, 4))
axs[0].imshow(img_rqb)
axs[0].set_title('Original image')
axs[1].imshow(dst)
axs[1].set_title('Translated image')
plt.tight_layout()
plt.show()
