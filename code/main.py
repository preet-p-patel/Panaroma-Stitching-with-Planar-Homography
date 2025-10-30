"""
Objective: To continously identify a laptop screen, find homography for it
and we can add displays parallel to it in overlay. 

Steps: 
1. Compute matching points correspondences between each frame and a template laptop screen
2. Use these points to compute homography from laptop template to the screen. 
3. Translate the homography by set distance. 
4. Overlay a frame from somewhere continously using this homography. 
"""
from utils import *

def get_frame()

def main():
    # initialise require variables
    template_path = ''
    template = cv2.imread(template_path)
    tranform = np.array([[1, 0, 100],
                         [0, 1, 0],
                         [0, 0, 1]])
    # initialise camera

    while True:
        # grab frame
        frame = get_frame()

        # compute matching points
        matches, locs1, locs2 = matchPics(frame, template)