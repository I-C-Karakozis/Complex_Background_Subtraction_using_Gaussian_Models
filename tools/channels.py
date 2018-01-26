import numpy as np
import cv2

def center_luma(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame_yuv[:,:,0] = 128
    frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB)
    return frame

def get_uv(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    return frame_yuv[:,:,1:]

def get_yuv(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    return frame_yuv

def get_bgr(frame):
    return frame

def gray_to_color(graydata, minval, maxval):
    # Based on https://stackoverflow.com/questions/43983265/rgb-to-yuv-conversion-and-accessing-y-u-and-v-channels
    colormap = np.array([[[0, i, 255-i] for i in range(256)]], dtype=np.uint8)
    scaled_data = np.clip((graydata + minval) / (maxval - minval), 0.0, 1.0)
    if scaled_data.dtype == float:
        scaled_data = (255 * scaled_data).astype(np.uint8)
    colorized = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2BGR)
    return cv2.LUT(colorized, colormap)
