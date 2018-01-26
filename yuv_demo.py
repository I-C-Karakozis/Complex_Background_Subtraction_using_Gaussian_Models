import cv2
import numpy as np
import sys

def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)

img = cv2.imread(sys.argv[1])

b = np.zeros(img.shape)
b[:, :, 0] = img[:, :, 0]

g = np.zeros(img.shape)
g[:, :, 1] = img[:, :, 1]

r = np.zeros(img.shape)
r[:, :, 2] = img[:, :, 2]

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

lut_u, lut_v = make_lut_u(), make_lut_v()

# Convert back to BGR so we can apply the LUT and stack the images
y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

u_mapped = cv2.LUT(u, lut_u)
v_mapped = cv2.LUT(v, lut_v)

result = np.hstack([img, y, u_mapped, v_mapped, r, g, b])

cv2.imwrite(sys.argv[2], result)
