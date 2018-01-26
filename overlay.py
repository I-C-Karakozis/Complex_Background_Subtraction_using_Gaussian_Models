import numpy as np
import argparse
import cv2
import sys

import skvideo.io
import imageio

from tools.dir_manager import *

def main(args): 

    print("Extracting Background Frame:", args.bg_mov)
    cap = skvideo.io.vreader(args.bg_mov)    
    bg = next(cap, None)

    print("Extracting Foreground Frame:", args.fg_mov)
    cap = skvideo.io.vreader(args.fg_mov)  
    frame_count = 1  
    fg = next(cap, None)
    while fg is not None and frame_count < args.fg_frame:
        fg = next(cap, None)
        frame_count = frame_count + 1

    print("Reading Mask:", args.mask)
    mask = imageio.imread(args.mask)
    if len(np.shape(mask)) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask>=128] = 255
        mask[mask<128] = 0

    print("Overlaying:", args.composite_path)
    overlay = np.where(np.repeat(np.expand_dims(mask, 2), 3, axis=2) > 0, fg, bg)
    imageio.imwrite(args.composite_path, overlay)

    # show background, composite and mask frame
    cv2.imshow('Mask', mask)
    cv2.imshow("Composite", overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Windows Setup using XLaunch:
# setting up x server; run XLaunch before your first run this program
# launch x server and then execute:
# $ export DISPLAY=:0

'''
Sample executions: 
python overlay.py test_data/BG/stage.avi test_data/Composite/shia_jadwin.avi 87 test_data/gmm/jadwin/shia/yuv/seg000087.png survey/gmm_jadwin_shia_yuv.png
python overlay.py test_data/BG/stage.avi test_data/Composite/shia_jadwin.avi 87 test_data/gmm/jadwin/shia/medfilt/yuv_3_1/seg000087.png survey/gmm_jadwin_shia_medfilt_yuv_3_1.png

Ground Truth Extraction
python overlay.py test_data/BG/stage.avi test_data/Composite/shia_jadwin.avi 87 test_data/shia_gt_jadwin/gt000087.png survey/gt.png
'''

DESCRIPTION = "Creating Composite Frame using Mask Frame, Foreground and Background Video"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('bg_mov', help='Background movie filename')
    parser.add_argument('fg_mov', help='Foreground movie filename')
    parser.add_argument('fg_frame', help='Frame number', type=int)
    parser.add_argument('mask',   help='Mask to apply foreground extraction with')
    parser.add_argument('composite_path', help='File path for the composite file to be stored in')
    args = parser.parse_args()
    main(args)
