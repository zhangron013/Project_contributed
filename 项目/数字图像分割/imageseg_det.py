import math
import numpy as np
import cv2
import tqdm
import os
#------------------------------ canny算子 ------------------------------
def gaussian(img):
    """对图像进行高斯滤波

    Args:
        img (numpy矩阵): 待处理图像矩阵

    Returns:
        numpy矩阵: 高斯滤波后的图像
    """
    sigma1 = sigma2 = 1
    sum = 0
    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i,j] = math.exp(-1/2 * (np.square(i-3)/np.square(sigma1)           #生成二维高斯分布矩阵
                            + (np.square(j-3)/np.square(sigma2)))) / (2*math.pi*sigma1*sigma2)
            sum = sum + gaussian[i, j]
    W, H = img.shape
    new_img = np.zeros([W-5, H-5])
    for i in range(W-5):
        for j in range(H-5):
            new_img[i,j] = np.sum(img[i:i+5,j:j+5]*gaussian)   # 与高斯矩阵卷积实现滤波
    return new_img
def grant(new_img):
    """对图像进行梯度计算

    Args:
        new_img (numpy矩阵): 高斯滤波处理后的图像

    Returns:
        d,dx,dy: 梯度矩阵 
    """
    W1, H1 = new_img.shape # 图像尺寸
    dx = np.zeros([W1-1, H1-1])# 梯度矩阵
    dy = np.zeros([W1-1, H1-1])# 梯度矩阵
    d = np.zeros([W1-1, H1-1])# 梯度矩阵
    for i in range(W1-1):
        for j in range(H1-1):   
            dx[i,j] = new_img[i, j+1] - new_img[i, j]
            dy[i,j] = new_img[i+1, j] - new_img[i, j]        
            d[i,j] = np.sqrt(np.square(dx[i,j]) + np.square(dy[i,j]))   # 图像梯度幅值作为图像强度值
    return d,dx,dy
def NMS_detect(d,dx,dy):
    """非极大值抑制

    Args:
        d,dx,dy: 梯度矩阵
    Returns:
        numpy矩阵: 非极大值抑制后的图像
    """
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0,:] = NMS[W2-1,:] = NMS[:,0] = NMS[:, H2-1] = 0
    for i in range(1, W2-1):
        for j in range(1, H2-1):      
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]
                
                # 如果Y方向幅度值较大
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i-1, j+1]
                        grad3 = d[i+1, j-1]
                        
                # 如果X方向幅度值较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i+1, j-1]
                        grad3 = d[i-1, j+1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
            
                gradTemp1 = weight * grad1 + (1-weight) * grad2
                gradTemp2 = weight * grad3 + (1-weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0
    return NMS
def edge_detector(NMS):
    """双阈值检测

    Args:
        NMS (numpy矩阵):非极大值抑制后的图像

    Returns:
        numpy矩阵: 双阈值检测后的图像
    """
    W3, H3 = NMS.shape
    DT = np.zeros([W3, H3])               
    # 定义高低阈值
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    for i in range(1, W3-1):
        for j in range(1, H3-1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1]).any() 
                or (NMS[i, [j-1, j+1]] < TH).any()):
                DT[i, j] = 1
    return DT
def canny_detector(img,name,output_dir):
    new_img = gaussian(img)
    d,dx,dy = grant(new_img)
    NMS = NMS_detect(d,dx,dy)
    DT = edge_detector(NMS)
    cv2.imwrite(os.path.join(output_dir,'canny_detector_'+name), DT*255)

#------------------------------ robert算子 ------------------------------
def cv_filter(kernelx, kernely, grayImage):
  """_summary_

    Args:
        kernelx (_type_):Robert算子核
        kernely (_type_): Robert算子核
        grayImage (_type_): 待处理图像

    Returns:
        _type_: _description_
  """
  # 算子
  x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
  y = cv2.filter2D(grayImage, cv2.CV_16S, kernely) 
  # 调用convertScaleAbs()函数计算绝对值，
  # 并将图像转换为8位图进行显示，然后进行图像融合
  absX = cv2.convertScaleAbs(x)
  absY = cv2.convertScaleAbs(y)
  Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
  return Roberts
def robert_detector(img,name,output_dir):
    # 使用 Numpy 构建卷积核，并对灰度图像在 x 和 y 的方向上做一次卷积运算
    # Roberts 算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    cv_Roberts = cv_filter(kernelx, kernely, img)
    cv2.imwrite(os.path.join(output_dir,'robert_detector_'+name), cv_Roberts)
#------------------------------ 主函数 ------------------------------
if __name__ =='__main__':
    input_dir = './input'
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name in tqdm.tqdm(sorted(os.listdir(input_dir))):
        if not name.endswith('.jpg'):
            continue
        print(name)
        img = cv2.imread(os.path.join(input_dir,name),0)
        canny_detector(img,name,output_dir)
        robert_detector(img,name,output_dir)