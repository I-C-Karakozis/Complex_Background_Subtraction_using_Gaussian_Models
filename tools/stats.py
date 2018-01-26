import numpy as np
import cv2
'''
 z_stat is the test statistic
 dof should equal the number of independent components
 confidence is the confidence level of the test

 dof accommodated: 1, 2 and 3
 confidence levels accommodated: 95%, 98%, 99% 

 source: https://pdfs.semanticscholar.org/7c21/22e26a55c16866a3d5c674dcbfe36278d469.pdf
'''
def h_test(z_stat, dof=3, confidence=95):

    # compute test statistic
    # z_stat = np.linalg.norm((sample-mean).astype(np.float) / std_matrix_diag) ** 2

    # perform chi_square_test; returns true if accept, false if reject
    # chisquare test values taken from: http://math.hws.edu/javamath/ryan/ChiSquare.html
    if dof == 1:
        if confidence == 95:
            return z_stat < 3.841   
        elif confidence == 98:
             return z_stat < 5.412   
        elif confidence == 99:
            return z_stat < 6.635
        else:
            print("Bad confidence level input.")
            return None
    elif dof == 2:
        if confidence == 95:
            return z_stat < 5.991
        elif confidence == 98:
             return z_stat < 7.824
        elif confidence == 99:
            return z_stat < 9.210
        else:
            print("Bad confidence level input.")
            return None
    elif dof == 3:
        if confidence == 95:
            return z_stat < 7.815   
        elif confidence == 98:
             return z_stat < 9.837   
        elif confidence == 99:
            return z_stat < 11.345
        else:
            print("Bad confidence level input.")
            return None
    else:
        print("Bad dof input.")
        return None


def train_fgbg(training_size, cap_train):

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)

    ## training for first k frames
    count = 0
    for frame in cap_train:

        if frame is None:
            print("frame is None")
            break
        
        # compute canny from despeckled mask
        fgmask = masks.create_fgmask_rgb(frame, fgbg)
        if fgmask is None:
            print("fgmask is None")
            break
        print("training: " + count)
        count = count + 1
        if count > training_size:
            break

    return fgbg
