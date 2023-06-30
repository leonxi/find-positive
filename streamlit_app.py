# coding=utf-8
import streamlit as st
from PIL import Image
import cv2 as cv
import tempfile
import numpy as np
from utils import ImgUtils 
import random

st.write("""
# 识别目标膜条中的阳性条带
""")

col1, col2, col3 = st.columns(3)

image = Image.open('samples/src.jpg')
with col1:
    st.image(image)

contours = Image.open('samples/dst.png')
with col2:
    st.image(contours)

positive = Image.open('samples/dst.png')
with col3:
    st.image(positive)

st.write("""
## 识别条带
""")

inSrc = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg', 'bmp'])

if inSrc is None:
    src = cv.imread('samples/src.jpg')
else:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(inSrc.read())

    src = cv.imread(tfile.name)

clonedSrc = src.copy()

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

thresh = st.slider("二值化阈值", 0, 255, 108, step=1)
maxval = st.slider("二值化最大值", 0, 255, 200, step=1)

# 二值化
ret, binary = cv.threshold(gray, thresh, maxval, cv.THRESH_BINARY)

erode_kernel_sizex = st.slider("腐蚀核尺寸 (x)", 3, 255, 3, step=1)
erode_kernel_sizey = st.slider("腐蚀核尺寸 (y)", 3, 255, 30, step=1)

# 腐蚀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (erode_kernel_sizex, erode_kernel_sizey))
erode = cv.erode(binary, kernel, iterations=1)

dilate_kernel_sizex = st.slider("膨胀核尺寸 (x)", 3, 255, 210, step=1)
dilate_kernel_sizey = st.slider("膨胀核尺寸 (y)", 3, 255, 80, step=1)

# 膨胀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernel_sizex, dilate_kernel_sizey))
dilate = cv.dilate(erode, kernel, iterations=1)

# 查找轮廓
foundContours, h = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# ouCol1, ouCol2 = st.columns(2)

black = np.zeros(src.shape)
output = cv.drawContours(black, foundContours, -1, (0,225,0), 3)
forOutput = output.astype("uint8")

toImage = Image.fromarray(cv.cvtColor(forOutput, cv.COLOR_BGR2RGB))

# with ouCol1:
#     st.image(toImage)

srcOutput = cv.drawContours(src, foundContours, -1, (0,225,0), 3)

toSrcImage = Image.fromarray(cv.cvtColor(srcOutput, cv.COLOR_BGR2RGB))
#cv.imwrite("samples/new.png", srcOutput)
# with ouCol2:
#     st.image(toSrcImage)

found = len(foundContours)

st.write("""
## 分割图片中的{}根膜条
""".format(found))

# 裁剪边距
margin = 5

for i in range(found):
    # 检测轮廓最小外接矩形，得到最小外接矩形的（中心（x, y），（宽，高），旋转角度）
    rect = cv.minAreaRect(foundContours[i])
    # 获取最小外接矩形的4个顶点坐标
    box = np.intp(cv.boxPoints(rect))

    # 原图像的高和宽
    h, w = clonedSrc.shape[:2]
    # 最小外接矩形的宽和高
    rect_w, rect_h = int(rect[1][0]) + 1, int(rect[1][1]) + 1

    if rect_w > rect_h:
        # 旋转中心
        x, y = int(box[1][0]), int(box[1][1])
        M2 = cv.getRotationMatrix2D((x, y), rect[2], 1)
        rotated_image = cv.warpAffine(clonedSrc, M2, (w * 2, h * 2))
        y1, y2 = y - margin if y - margin > 0 else 0, y + rect_h + margin + 1
        x1, x2 = x - margin if x - margin > 0 else 0, x + rect_w + margin + 1
        rotated_canvas = rotated_image[y1 : y2, x1 : x2]
        he, wi = rotated_canvas.shape[:2]
    else:
        # 旋转中心
        x, y = int(box[2][0]), int(box[2][1])
        M2 = cv.getRotationMatrix2D((x, y), rect[2] + 90, 1)
        rotated_image = cv.warpAffine(clonedSrc, M2, (w * 2, h * 2))
        y1, y2 = y - margin if y - margin > 0 else 0, y + rect_w + margin + 1
        x1, x2 = x - margin if x - margin > 0 else 0, x + rect_h + margin + 1
        rotated_canvas = rotated_image[y1 : y2, x1 : x2]
        # 旋转180度，恢复原始图像的方向
        rotated_canvas = cv.flip(rotated_canvas, -1)
        he, wi = rotated_canvas.shape[:2]
    
    to_rotated_canvas = Image.fromarray(cv.cvtColor(rotated_canvas, cv.COLOR_BGR2RGB))
    st.write("第", i + 1, "根 尺寸: (", he, ", ", wi, ")")
    st.image(to_rotated_canvas)

    thresh = st.slider("二值化阈值", 0, 255, 100, step=1, key=i+100)
    maxval = st.slider("二值化最大值", 0, 255, 120, step=1, key=i+200)
    binary = ImgUtils.get_positive(rotated_canvas.copy(), thresh, maxval)
    to_binary = Image.fromarray(cv.cvtColor(binary, cv.COLOR_BGR2RGB))
    st.image(to_binary)
    
    # 腐蚀参数
    erode_kernel_sizex = st.slider("腐蚀核尺寸 (x)", 3, 255, 3, step=1, key=i+300)
    erode_kernel_sizey = st.slider("腐蚀核尺寸 (y)", 3, 255, 3, step=1, key=i+400)

    # 膨胀参数
    dilate_kernel_sizex = st.slider("膨胀核尺寸 (x)", 3, 255, 3, step=1, key=i+500)
    dilate_kernel_sizey = st.slider("膨胀核尺寸 (y)", 3, 255, 160, step=1, key=i+600)

    filter = ImgUtils.dilate_erode(rotated_canvas.copy(), (dilate_kernel_sizex, dilate_kernel_sizey), (erode_kernel_sizex, erode_kernel_sizey))
    to_filter = Image.fromarray(cv.cvtColor(filter, cv.COLOR_BGR2RGB))
    st.image(to_filter)
