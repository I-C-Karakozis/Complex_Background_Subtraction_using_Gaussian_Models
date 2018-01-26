import numpy as np
import cv2

# creates foreground mask from rgb frame
def create_fgmask(frame, fgbg):

    #yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    fgmask = fgbg.apply(frame)
    if fgmask is None:
        return None

    fgmask[fgmask < 255] = 0
    return fgmask

# convert frame to yuv and creates foreground mask from it
def create_fgmask_yuv(frame, fgbg):

    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # yuv_frame[...,0] = 128
    fgmask = fgbg.apply(yuv_frame)
    if fgmask is None:
        return None

    fgmask[fgmask < 255] = 0
    
    return fgmask

# nullifies zero mask pixel values from frame
def apply_mask(mask, frame):

    dims = np.shape(mask)
    applied = np.copy(frame)

    for row in range(dims[0]):
        for col in range(dims[1]):
            if mask[row][col] < 255:
                applied[row][col][0] = 0
                applied[row][col][1] = 0
                applied[row][col][2] = 0

    return applied
