import cv2
import numpy as nm
import matplotlib.pyplot as plt
from random import randrange

img = cv2.imread("C:/Users/singh/OneDrive/Documents/cherry blossom 1.jpg")
imgl = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = cv2.imread("C:/Users/singh/OneDrive/Documents/cherry blossom 2.jpg")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kpl, desl = sift.detectAndCompute(imgl, None)
kp2, des2 =sift.detectAndCompute (img2, None)

bf = cv2.BFMatcher()

matches= bf.knnMatch (desl,des2, k=2)
good = []

for m in matches:

  if m[0].distance < 0.5*m[1].distance:
   good.append(m)
matches = nm.asarray(good)

if len (matches[:,0]) >= 4:
    src = nm.float32([kpl[m.queryIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
    dst = nm.float32([kp2[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)

else:
  raise AssertionError("Can't find enough keypoints.")


dst = cv2.warpPerspective(img, H, (img.shape[1] + img.shape[1], img.shape[0]))
plt.subplot (122), plt.imshow(dst),plt.title("Warped Image")

plt.show()

plt.figure()

dst [0:img. shape[0], 0:img.shape[1]] = img
cv2.imwrite("resultant_stitched_panorama.jpg",dst)

plt.imshow(dst)

plt.show()
