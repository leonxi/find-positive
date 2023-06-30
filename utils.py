# coding=utf-8
import streamlit as st
import cv2 as cv

class ImgUtils():

    @staticmethod
    def get_positive(src, thresh=100, maxval=120):
        if src is None:
            return src

        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        
        # 二值化
        ret, binary = cv.threshold(gray, thresh, maxval, cv.THRESH_BINARY)

        return binary

ImgUtils.switch = {
    1: ImgUtils.get_positive(src=None, thresh=100, maxval=120)
}