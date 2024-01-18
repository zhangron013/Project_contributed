# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import os
import cv2 as cv
import tqdm

min_thred = 180 #二值化的阈值
min_area = 120 #矩形框最小的面积，小于此面积的矩形框判定不认为是答案
max_area = 400 #矩形框最大的面积，大于此面积的矩形框判定不认为是答案
approx_num = 20 #矩形框的顶点数，小于此数的矩形框判定不认为是答案
pw = 0.6   #矩形框中白色像素值的比例，大于此比例的矩形框判定不认为是答案
top = 160  #矩形框的y坐标，小于此坐标的矩形框判定不认为是答案
# bottom = 270 #矩形框的y坐标，大于此坐标的矩形框判定不认为是答案
def main(input_file,out_file):
    # 加载一个图片到opencv中

    img = cv.imread(input_file)

    # 打印原图
    # cv.imshow("orgin", img)

    # 灰度化
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 打印灰度图
    # cv.imshow("gray", gray)
    # 自适应二值化算法
    ret,thresh2 = cv.threshold(gray,min_thred,255,cv.THRESH_BINARY)
    gray_deal = thresh2.copy()
    # 打印二值化后的图
    # cv.imshow("thresh2", thresh2)
    cv.imwrite("thresh2.jpg", thresh2)
    cts, _ = cv.findContours(thresh2,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 寻找轮廓
    # cts, hierarchy = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 打印找到的轮廓
    # print("轮廓数：", len(cts))

    # 对拷贝的原图进行轮廓标记
    contour_flagged = cv.drawContours(img.copy(), cts, -1, (0, 0, 255), 3)
    # 打印轮廓图
    # cv.imshow("contours_flagged", contour_flagged)
    cv.imwrite("contours_flagged.jpg", contour_flagged)
    # 按像素面积降序排序
    list = sorted(cts, key=cv.contourArea, reverse=True)
    # print(len(list))
    # 遍历轮廓
    question_list = []
    for ct in list:
        # 周长，第1个参数是轮廓，第二个参数代表是否是闭环的图形
        peri = 0.01 * cv.arcLength(ct, True)
        # print("周长：", peri)
        #计算面积
        area = cv.contourArea(ct)
        # print("面积：", area)
        if area < min_area or area > max_area:
            continue
        else:
            approx = cv.approxPolyDP(ct, peri, True)
            # print("approx:", len(approx))
            if len(approx)<= approx_num:
                x, y, w, h = cv.boundingRect(ct)
                #判断矩形是否宽度大于高度
                if w >h and w>=20 and w/h<5:
                    area_ori = gray_deal[y:y+h, x:x+w]
                    #统计area_ori中白色像素值的个数
                    white_pix = np.count_nonzero(area_ori)
                    # print("白色像素值的个数：", white_pix)
                    # print("面积：", area_ori.shape[0]*area_ori.shape[1])
                    # print("比例：", white_pix/(area_ori.shape[0]*area_ori.shape[1]))
                    if white_pix/(area_ori.shape[0]*area_ori.shape[1])<pw and y>top :
                        # print("x:{},y:{},w:{},h:{}".format(x, y, w, h))
                        question_list.append(ct) 

            # print("答案总数：", len(question_list))

            # 按坐标从上到下排序
    questionCnts = contours.sort_contours(question_list, method="top-to-bottom")[0]
    cv.drawContours(img, questionCnts, -1, (0, 0, 255), 2)
    cv.imwrite(out_file, img)

if __name__ == "__main__":
    input_dir = './data/'
    output_dir = './data_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_files = os.listdir(input_dir)
    for file in tqdm.tqdm(sorted(input_files)):
        if file.endswith('.jpg'):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file)
            main(input_file, output_file)