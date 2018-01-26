import numpy as np
import argparse
import imageio
import cv2
import time
from os import listdir, path

def main(args):
    print("Segmentations:", args.segmentation_path)
    print("Ground truth:", args.ground_truth_path)

    seg_filenames = listdir(args.segmentation_path)
    gt_filenames = listdir(args.ground_truth_path)

    print(len(seg_filenames), "files found")

    false_positive_rate = np.zeros(len(seg_filenames), dtype=float)
    false_negative_rate = np.zeros(len(seg_filenames), dtype=float)
    fg_recall = np.zeros(len(seg_filenames), dtype=float)
    fg_precision = np.zeros(len(seg_filenames), dtype=float)
    accuracy = np.zeros(len(seg_filenames), dtype=float)

    for i in range(len(seg_filenames)):
        seg_im = imageio.imread(path.join(args.segmentation_path, seg_filenames[i]))
        gt_im = imageio.imread(path.join(args.ground_truth_path, gt_filenames[i]))
 
        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2GRAY)
        gt_im[gt_im>=128] = 255
        gt_im[gt_im<128] = 0
        seg_im[np.nonzero(seg_im)] = 255

        num_fg_gt = float(np.count_nonzero(gt_im))
        num_bg_gt = float(gt_im.size - num_fg_gt)

        num_fg_seg = float(np.count_nonzero(seg_im))

        # Positive: Foreground
        # Negative: Background
        # Sanity check: Recall should be complement of false negative

        if num_bg_gt > 0:
            # False positive:  background marked as foreground
            false_positive_rate[i] = np.count_nonzero(seg_im > gt_im) / num_bg_gt
        else:
            false_positive_rate[i] = None

        if num_fg_gt > 0:
            # False negative: foreground marked as background 
            false_negative_rate[i] = np.count_nonzero(seg_im < gt_im) / num_fg_gt

            # FG Recall: percentage of foreground that are marked as foreground
            fg_recall[i] = np.count_nonzero(np.logical_and(seg_im == gt_im,seg_im > 0)) / num_fg_gt
        else:
            false_negative_rate[i] = None
            fg_recall[i] = None

        if num_fg_seg > 0:
            # FG Precision: percentage of marked as foreground that are actually foreground
            fg_precision[i] = np.count_nonzero(np.logical_and(seg_im == gt_im,seg_im > 0)) / num_fg_seg            
        else:
            fg_precision[i] = None
        
        # Accuracy: fraction of pixels correctly labeled
        accuracy[i] = np.count_nonzero(seg_im == gt_im) / float(seg_im.size)

    print("---------------------------------------------------------------------")
    print("Mean Rates:")
    print("Accuracy:", accuracy.mean())
    print("False Positive Rate:", np.nanmean(false_positive_rate))
    print("False Negative Rate:", np.nanmean(false_negative_rate))
    print("Foreground Recall:", np.nanmean(fg_recall))
    print("Foreground Precision:", np.nanmean(fg_precision))    
    print("---------------------------------------------------------------------")
    
'''
Sample execution: 
python testing.py test_data/mog/jadwin/shia/bgr test_data/shia_gt_jadwin
python testing.py test_data/gmm/jadwin/shia/bgr test_data/shia_gt_jadwin
python testing.py test_data/mog/jadwin/shia/medfilt/yuv_3_1 test_data/shia_gt_jadwin
python testing.py test_data/gmm/jadwin/shia/medfilt/yuv_5_100 test_data/shia_gt_jadwin
'''
DESCRIPTION = """Evaluate performance of segmentation compared to provided ground truth"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('segmentation_path', help='Path to folder containing segmentation output image series')
    parser.add_argument('ground_truth_path', help='Path to folder containing ground truth segementation')
    args = parser.parse_args()
    main(args)