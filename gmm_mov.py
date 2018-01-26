import numpy as np
import cv2
import argparse
from os import listdir, path

import skvideo.io
import imageio

from tools.dir_manager import *
from tools import stats

def main(args):

    get_channels = args_setup(args)

    # setup training video sequence; backgroud only footage    
    cap_train = skvideo.io.vreader(args.training_mov)
    orig_frame = next(cap_train, None)
    black_frame_1d = np.zeros((orig_frame.shape[0], orig_frame.shape[1]), dtype=np.uint8)
    black_frame = np.zeros(orig_frame.shape, dtype=np.uint8)

    # Load training data
    training_data = np.zeros((args.num_train_frames,) + orig_frame.shape)
    if args.chan == "uv":
        training_data = training_data[:,:,:,:2]
    for i in range(args.num_train_frames):
        training_data[i] = get_channels(orig_frame)
        orig_frame = next(cap_train, None)
    print("Training frames:", training_data.shape)
    print("Training complete")

    # Calculate Gaussian model
    training_mean = np.mean(training_data, axis=0)
    training_std = np.std(training_data, axis=0)

    np.clip(training_std, 1.0, None, training_std)
    print("Model assembled")

    white_frame = np.ones(orig_frame.shape, dtype=np.uint8)
    black_frame = np.zeros(orig_frame.shape, dtype=np.uint8)

    # Evaluate rest of frames
    cap = skvideo.io.vreader(args.movie_file)
    orig_frame = next(cap, None)
    i = 1
    while orig_frame is not None:

        frame = get_channels(orig_frame)
        
        # Zscore is number of standard devations an observation is from the mean
        zscore = np.square((frame - training_mean).astype(np.float) / training_std)
        zscore = np.apply_over_axes(np.sum, zscore, [2])
        chroma_decision = np.where(stats.h_test(zscore, confidence=99, dof=np.shape(frame)[2]), 0, 255).astype(np.uint8)   

        # despeckling
        for j in range(args.medfilt_iters):
            chroma_decision = cv2.medianBlur(chroma_decision, args.kernel_size)

        # show original, composite and masked frame
        orig_frame = orig_frame[:,:,::-1]
        
        cv2.imshow("Chroma Score", chroma_decision)
        cv2.imshow('Original Frame', orig_frame)
        cv2.imshow("Composite", np.where(np.repeat(np.expand_dims(chroma_decision, 2), 3, axis=2) > 0, orig_frame, black_frame))

        # Play at half speed, Press q to quit
        if cv2.waitKey(int(1)) & 0xFF == ord('q'):
            break

        print(path.join(args.segmentation_dir, 'seg{:06d}.png'.format(i)))
        imageio.imwrite(path.join(args.segmentation_dir, 'seg{:06d}.png'.format(i)), chroma_decision)

        i = i + 1
        orig_frame = next(cap, None)

    cv2.destroyAllWindows()

# Windows Setup using XLaunch:
# setting up x server; run XLaunch before your first run this program
# launch x server and then execute:
# $ export DISPLAY=:0

'''
Sample execution: 
python gmm_mov.py test_data/BG/jadwin.avi test_data/Composite/shia_jadwin.avi 60 test_data/gmm/jadwin/shia/yuv yuv 5 1
python gmm_mov.py test_data/BG/jadwin.avi test_data/Composite/shia_jadwin.avi 60 test_data/gmm/jadwin/shia/medfilt/yuv_3_1 yuv 3 1
'''
DESCRIPTION = "Use bayesian estimation to produce a simple Gaussian Multivariate Model for background masking"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('training_mov', help='Input movie filename for training')
    parser.add_argument('movie_file', help='Input movie filename')
    parser.add_argument('num_train_frames', help='Number of training frames', type=int)
    parser.add_argument('segmentation_dir', help='Directory to put segmentation output images')
    parser.add_argument('chan', help='''Determine color space and channels to use; Options: uv yuv no_luma bgr; 
                                        Defaults to bgr''', default="bgr")
    parser.add_argument('kernel_size', help='Size of median filter kernel', type=int)
    parser.add_argument('medfilt_iters', help='Number of times to apply the median filter to each frame', type=int)
    args = parser.parse_args()
    main(args)

