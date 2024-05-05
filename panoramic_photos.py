import time
from enum import Enum

import numpy as np
import imutils
import cv2


class DetectionMethods(Enum):
    SIFT = 'sift'
    ORB = 'orb'


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images: list, detection_method: DetectionMethods, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA, detection_method)
        (kpsB, featuresB) = self.detectAndDescribe(imageB, detection_method)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)
            # return a tuple of the stitched image and the
            # visualization
            return result, vis
        # return the stitched image
        return result

    def detectAndDescribe(self, image: np.ndarray, detection_method: DetectionMethods):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            if detection_method == DetectionMethods.SIFT:
                # detect and extract features from the image
                descriptor = cv2.SIFT_create()
                kps, features = descriptor.detectAndCompute(gray, None)
            elif detection_method == DetectionMethods.ORB:
                orb = cv2.ORB_create()
                kps, features = orb.detectAndCompute(gray, None)
            else:
                raise NotImplementedError
        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            kps, features = extractor.compute(gray, kps)
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsA: np.ndarray, kpsB: np.ndarray, featuresA: np.ndarray, featuresB: np.ndarray,
                       ratio: float, reprojThresh: float):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return matches, H, status
            # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA: np.ndarray, imageB: np.ndarray, kpsA: np.ndarray, kpsB: np.ndarray, matches: list,
                    status: list) -> np.ndarray:
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis


def go(images, detection_method: DetectionMethods, ratio: float):
    result = None
    for idx in range(len(images) - 1):
        # load the two images and resize them to have a width of 400 pixels
        # (for faster processing)
        if result is None:
            imageA = cv2.imread(images[idx])
            imageB = cv2.imread(images[idx + 1])
            imageA = imutils.resize(imageA, width=400)
            imageB = imutils.resize(imageB, width=400)
        else:
            imageA = result
            imageB = cv2.imread(images[idx + 1])
            imageB = imutils.resize(imageB, width=400)

        # stitch the images together to create a panorama
        stitcher = Stitcher()
        result, vis = stitcher.stitch([imageA, imageB], detection_method, ratio=ratio, showMatches=True)
        # show the images
    # cv2.imshow(f"Image A {idx}", imageA)
    # cv2.imshow(f"Image B {idx}", imageB)
    # cv2.imshow(f"Keypoint Matches {idx}", vis)
    # cv2.imshow(f"Result {idx}", result)
    # cv2.waitKey(0)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    first_file = r"D:\Learning\Computer_Vision\stitch\first.jpg"
    second_file = r"D:\Learning\Computer_Vision\stitch\second.jpg"
    third_file = r"D:\Learning\Computer_Vision\stitch\third.jpg"
    input_images = [first_file, second_file]
    detection_method = DetectionMethods.ORB
    ratio = 0.78
    s = time.time()
    go(input_images, detection_method, ratio)
    print(f"Running time {time.time() - s}")
