# coding=utf-8
import streamlit as st
import cv2 as cv
from enum import Enum

class Extract_Features_Method(Enum):
    METHOD_SIFT = 0
    METHOD_SURF = 1
    METHOD_BRISH = 2
    METHOD_ORB = 3

class ImgUtils():

    @staticmethod
    def get_positive(src, thresh=100, maxval=120):
        if src is None:
            return src

        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        
        # 二值化
        ret, binary = cv.threshold(gray, thresh, maxval, cv.THRESH_BINARY)

        return binary

    @staticmethod
    def readimage(path, flags = None):
        if flags == None:
            return cv.imread(path)

        return cv.imread(path, flags)

    @staticmethod
    def detectAndDescribe(image, method):
        keypoints = None;
        descriptor = None;

        if method == Extract_Features_Method.METHOD_SIFT:
            detector = cv.SIFT_create(800);
            keypoints, descriptor = detector.detectAndCompute(image, None);
            return keypoints, descriptor;
        # 新版本由于专利影响, OpenCV已经删除了此算法
        elif method == Extract_Features_Method.METHOD_SURF:
            detector = cv.xFeatures2d.SURF_create(400);
            keypoints, descriptor = detector.detectAndCompute(image, None);
            return keypoints, descriptor;
        elif method == Extract_Features_Method.METHOD_BRISH:
            detector = cv.BRISK.create(400);
            keypoints, descriptor = detector.detectAndCompute(image, None);
            return keypoints, descriptor;
        elif method == Extract_Features_Method.METHOD_ORB:
            detector = cv.ORB.create();
            keypoints, descriptor = detector.detectAndCompute(image, None);
            return keypoints, descriptor;
        else:
            return keypoints, descriptor;

    @staticmethod
    def findFeatures(src):
        if src is None:
            return None

        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        
        # 获取特征点
        keypoint1, describe1 = ImgUtils.detectAndDescribe(gray, Extract_Features_Method.METHOD_SIFT);

        FLANN_INDEX_KDTREE = 0;
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5);
        searchParams = dict(checks=50);
        flann = cv.FlannBasedMatcher(indexParams, searchParams);
        
        keypoint_img1 = cv.drawKeypoints(src, keypoint1, src.copy(), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);

        return keypoint_img1, keypoint1, describe1

ImgUtils.switch = {
    1: ImgUtils.get_positive(src=None, thresh=100, maxval=120),
    2: ImgUtils.readimage(path=None, flags=None),
    3: ImgUtils.detectAndDescribe(image=None, method=None),
    4: ImgUtils.findFeatures(src=None)
}