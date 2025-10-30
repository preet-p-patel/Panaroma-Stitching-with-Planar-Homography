import os
import numpy as np
import cv2
import skimage.color
import pickle
from matplotlib import pyplot as plt
import scipy
from skimage.util import montage
import time
import skimage


PATCHWIDTH = 9

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def briefMatch(desc1,desc2,ratio):

    matches = skimage.feature.match_descriptors(desc1,desc2,
                                                'hamming',
                                                cross_check=True,
                                                max_ratio=ratio)
    return matches

def plotMatches(img1,img2,matches,locs1,locs2):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    # skimage.feature.plot_matches(ax,img1,img2,locs1,locs2,
    #                              matches,matches_color='r',only_matches=True)
    skimage.feature.plot_matched_features(image0=img1,image1=img2,keypoints0=locs1,keypoints1=locs2,
                                 matches=matches,keypoints_color='r',only_matches=True, ax=ax)
    plt.show()
    return

def makeTestPattern(patchWidth, nbits):

    np.random.seed(0)
    compareX = patchWidth*patchWidth * np.random.random((nbits,1))
    compareX = np.floor(compareX).astype(int)
    np.random.seed(1)
    compareY = patchWidth*patchWidth * np.random.random((nbits,1))
    compareY = np.floor(compareY).astype(int)

    return (compareX, compareY)

def computePixel(img, idx1, idx2, width, center):

    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0

def computeBrief(img, locs):

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])

    return desc, locs

def corner_detection(img, sigma):

    # fast method
    result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    return locs

def loadVid(path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)

    # get fps, width, and height
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Append frames to list
    frames = []

    # Check if camera opened successfully
    if cap.isOpened()== False:
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            #Store the resulting frame
            frames.append(frame)
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    frames = np.stack(frames)

    return frames, fps, width, height

def matchPics(I1, I2, ratio, sigma):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images (RGB or Grayscale uint8)
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    # ===== your code here! =====

    # TODO: Convert images to GrayScale
    # Input images can be either RGB or Grayscale uint8 (0 -> 255). Both need
    # to be supported.
    # Input images must be converted to normalized Grayscale (0.0 -> 1.0)
    # skimage.color.rgb2gray may be useful if the input is RGB.
    if len(I1.shape) == 3 and I1.shape[2] == 3:
        I1 = skimage.color.rgb2gray(I1)
        # I1 = I1.astype(np.float32) / 255.0
    else: 
        I1 = I1.astype(np.float32) / 255.0
    
    if len(I2.shape) == 3 and I2.shape[2] == 3:
        I2 = skimage.color.rgb2gray(I2)
    else: 
        I2 = I2.astype(np.float32) / 255.0

    # TODO: Detect features in both images
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)

    # TODO: Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    # TODO: Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    # ==== end of code ====

    return matches, locs1, locs2

def computeH(x1, x2):
    """
    Compute the homography between two sets of points

    Input
    -----
    x1, x2: Sets of points

    Returns
    -------
    H2to1: 3x3 homography matrix that best transforms x2 to x1
    """

    if x1.shape != x2.shape:
        raise RuntimeError('number of points do not match')
    
    # ===== your code here! =====
    # TODO: Compute the homography between two sets of points
    # row1 = np.array([x2[:,0], x2[:,1], np.ones(shape=(len(x2))), np.zeros(shape=(len(x2))), np.zeros(shape=(len(x2))), np.zeros(shape=(len(x2))), -x2[:,0]*x1[:,0], -x2[:,1]*x1[:,0], -x1[:,0]])
    # row2 = np.array([np.zeros(shape=(len(x2))), np.zeros(shape=(len(x2))), np.zeros(shape=(len(x2))), x2[:,0], x2[:,1], np.ones(shape=(len(x2))), -x2[:,0]*x1[:,1], -x2[:,1]*x1[:,1], -x1[:,1]])
    
    # A = np.concatenate([row1, row2], axis=1).T
    A = []
    for x, u in zip(x1, x2):
        A.append(np.array([u[0], u[1], 1, 0, 0, 0, -u[0]*x[0], -u[1]*x[0], -x[0]]))
        A.append(np.array([0, 0, 0, u[0], u[1], 1, -u[0]*x[1], -u[1]*x[1], -x[1]]))
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    V = Vh.T
    h = V[:,-1]
    H2to1 = h.reshape(3,3)
    # ==== end of code ====

    return H2to1


def computeH_norm(x1, x2):
    """
    Compute the homography between two sets of points using normalization

    Input
    -----
    x1, x2: Sets of points

    Returns
    -------
    H2to1: 3x3 homography matrix that best transforms x2 to x1
    """

    # ===== your code here! =====

    # TODO: Compute the centroid of the points
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # TODO: Shift the origin of the points to the centroid
    x1 = x1 - x1_centroid
    x2 = x2 - x2_centroid

    # TODO: Normalize the points so that the largest distance from the
    # origin is equal to sqrt(2)
    f1 = np.sqrt(2)/(np.max(np.linalg.norm(x1, axis=1)))
    f2 = np.sqrt(2)/(np.max(np.linalg.norm(x2, axis=1)))

    x1 = x1*f1
    x2 = x2*f2

    # TODO: Similarity transform 1
    T1 = np.array([[f1, 0, -f1*x1_centroid[0]], 
                   [0, f1, -f1*x1_centroid[1]], 
                   [0, 0, 1]])

    # TODO: Similarity transform 2
    T2 = np.array([[f2, 0, -f2*x2_centroid[0]], 
                   [0, f2, -f2*x2_centroid[1]], 
                   [0, 0, 1]])

    # TODO: Compute homography
    H2to1 = computeH(x1, x2)

    # TODO: Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2

    # ==== end of code ====

    return H2to1

def computeH_ransac(locs1, locs2, max_iters, inlier_tol):
    """
    Estimate the homography between two sets of points using ransac

    Input
    -----
    locs1, locs2: Lists of points
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    Returns
    -------
    bestH2to1: 3x3 homography matrix that best transforms locs2 to locs1
    inliers: indices of RANSAC inliers

    """

    # ===== your code here! =====

    # TODO:
    # Compute the best fitting homography using RANSAC
    # given a list of matching points locs1 and loc2
    locs1 = np.hstack([locs1, np.ones([len(locs1), 1])])
    locs2 = np.hstack([locs2, np.ones([len(locs2), 1])])

    most_inliers = 0
    for _ in range(max_iters):
        idx = np.random.choice(locs1.shape[0], size=4, replace=False)
        x1 = locs1[idx]
        x2 = locs2[idx]
        H = computeH_norm(x1, x2)
        locs1_ = locs2 @ H.T
        locs1_ = locs1_ / locs1_[:, [2]]
        err = np.linalg.norm(locs1[:, :2] - locs1_[:, :2], axis=1)
        inliers = err <= inlier_tol

        inlier_count = np.sum(inliers)

        if inlier_count >= most_inliers:
            most_inliers = inlier_count
            bestH2to1 = H
            best_inliers = np.where(inliers==1)[0]

    # ==== end of code ====

    return bestH2to1, best_inliers


def compositeH(H2to1, template, img):
    """
    Returns the composite image.

    Input
    -----
    H2to1: Homography from image to template
    template: template image to be warped
    img: background image

    Returns
    -------
    composite_img: Composite image

    """

    # ===== your code here! =====
    # TODO: Create a composite image after warping the template image on top
    # of the image using the homography
    warped_template = cv2.warpPerspective(template, M=H2to1, dsize=(img.shape[1],img.shape[0]))
    mask = np.where(warped_template != 0)
    img[mask] = warped_template[mask]
    # ==== end of code ====

    return img

def warpImage(ratio, sigma, max_iters, inlier_tol):
    """
    Warps hp_cover.jpg onto the book cover in cv_desk.png.

    Input
    -----
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    """

    hp_cover = skimage.io.imread(os.path.join(DATA_DIR, 'hp_cover.jpg'))
    cv_cover = skimage.io.imread(os.path.join(DATA_DIR, 'cv_cover.jpg'))
    cv_desk = skimage.io.imread(os.path.join(DATA_DIR, 'cv_desk.png'))
    cv_desk = cv_desk[:, :, :3]

    # ===== your code here! =====

    # TODO: match features between cv_desk and cv_cover using matchPics
    matches, locs1, locs2 = matchPics(cv_desk, cv_cover, ratio, sigma)

    locs1 = locs1[matches[:,0]]
    locs2 = locs2[matches[:,1]]

    locs1[:,[1, 0]] = locs1[:,[0, 1]]
    locs2[:,[1, 0]] = locs2[:,[0, 1]]

    # TODO: Scale matched pixels in cv_cover to size of hp_cover
    # factor = np.array([hp_cover.shape[0]/cv_cover.shape[0], hp_cover.shape[1]/cv_cover.shape[1]])
    # locs_scaled = locs2 * factor
    
    template = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]), interpolation= cv2.INTER_AREA)

    # TODO: Get homography by RANSAC using computeH_ransac
    H2to1, _ = computeH_ransac(locs1, locs2, max_iters, inlier_tol)
    # print("H2to1: \n", H2to1)
    transform = np.array([[1, 0, 100],
                          [0, 1, 100],
                          [0, 0, 1]])
    H2to1 = H2to1 @ transform
    # H2to1[-1,-1] = 1
    # TODO: Overlay using compositeH to return composite_img
    composite_img = compositeH(H2to1, template, cv_desk)

    # ==== end of code ====

    plt.imshow(composite_img)
    plt.show()