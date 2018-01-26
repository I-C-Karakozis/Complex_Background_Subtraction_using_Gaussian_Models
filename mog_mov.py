import numpy as np
import imageio
import argparse
import cv2
import sys

from os import listdir, path
import skvideo.io

from tools.dir_manager import *
from tools import stats
from tools import masks

def main(args):

    get_channels = args_setup(args)

    # reset segmentation directory
    clear_dir(args.segmentation_dir)
    create_dir(args.segmentation_dir)  

    # Give MOG some background frames to train history on for initialization
    print("Training")
    cap_train = skvideo.io.vreader(args.training_mov)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=2, history=1000)

    orig_frame = next(cap_train, None)
    while orig_frame is not None:

        # compute canny from despeckled mask for current frame
        frame = get_channels(orig_frame)
        fgmask = masks.create_fgmask(frame, fgbg)
        if fgmask is None:
            print("fgmask is None")
            break

        orig_frame = next(cap_train, None)
    print("Training complete")

    # Load video with foreground object
    cap = skvideo.io.vreader(args.movie_file)

    # setup frame sequence
    frame_count = 1
    orig_frame = next(cap, None)
    print("Frames Dimensions:", orig_frame.shape)
    black_frame = np.zeros(orig_frame.shape, dtype=np.uint8)

    while orig_frame is not None:

        # compute canny from despeckled mask for current frame
        frame = get_channels(orig_frame)
        fgmask = masks.create_fgmask(frame, fgbg)
        if fgmask is None:
            print("fgmask is None")
            break

        # despeckle
        for j in range(args.medfilt_iters):
            fgmask = cv2.medianBlur(fgmask, args.kernel_size)

        # show original, composite and masked frame
        cv2.imshow('Original Frame', orig_frame)
        cv2.imshow('MOG Mask', np.matrix(fgmask))
        cv2.imshow("Composite", np.where(np.repeat(np.expand_dims(fgmask, 2), 3, axis=2) > 0, orig_frame, black_frame))

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # save mask in segmentation directory
        print(path.join(args.segmentation_dir, 'seg{:06d}.png'.format(frame_count)))
        imageio.imwrite(path.join(args.segmentation_dir, 'seg{:06d}.png'.format(frame_count)), fgmask)

        frame_count = frame_count + 1
        orig_frame = next(cap, None)

    cv2.destroyAllWindows()

# Windows Setup usin XLaunch:
# setting up x server; run XLaunch before your first run this program
# launch x server and then execute:
# $ export DISPLAY=:0

'''
Sample execution: 
python mog_mov.py test_data/BG/jadwin.avi test_data/Composite/shia_jadwin.avi test_data/mog/jadwin/shia/uv uv 5 1
python mog_mov.py test_data/BG/jadwin.avi test_data/Composite/shia_jadwin.avi test_data/mog/jadwin/shia/medfilt/yuv_3_1 yuv 3 1
'''
DESCRIPTION = "MOG Model for Background Subtraction with repeated Median Filter Application for Despeckling"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('training_mov', help='Input movie filename for training')
    parser.add_argument('movie_file', help='Input movie filename')
    parser.add_argument('segmentation_dir', help='Directory to put segmentation output images')
    parser.add_argument('chan', help='''Determine color space and channels to use; Options: uv yuv no_luma bgr; 
                                        Defaults to bgr''', default="bgr")
    parser.add_argument('kernel_size', help='Size of median filter kernel', type=int)
    parser.add_argument('medfilt_iters', help='Number of times to apply the median filter to each frame', type=int)
    args = parser.parse_args()
    main(args)
