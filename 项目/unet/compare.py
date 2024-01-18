# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
# import cv2
# import numpy as np
# import glob
# import os
# from medpy.metric.binary import dc
# # torch.set_deterministic(True)

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class SegmentationMetric(object):
#     def __init__(self, numClass):
#         self.numClass = numClass
#         self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

#     def addBatch(self, imgPredict, imgLabel, ignore_labels):
#         assert imgPredict.shape == imgLabel.shape
#         self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
#         return self.confusionMatrix

#     def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
#         """
#         同FCN中score.py的fast_hist()函数,计算混淆矩阵
#         :param imgPredict:
#         :param imgLabel:
#         :return: 混淆矩阵
#         """
#         # remove classes from unlabeled pixels in gt image and predict
#         mask = (imgLabel >= 0) & (imgLabel < self.numClass)
#         # for IgLabel in ignore_labels:
#         #     mask &= (imgLabel != IgLabel)
#         label = self.numClass * imgLabel[mask] + imgPredict[mask]
#         count = torch.bincount(label, minlength=self.numClass ** 2)
#         confusionMatrix = count.view(self.numClass, self.numClass)
#         # print(confusionMatrix)
#         return confusionMatrix



#     def IntersectionOverUnion(self):
#         # Intersection = TP Union = TP + FP + FN
#         # IoU = TP / (TP + FP + FN)
#         intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
#         union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
#             self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
#         IoU = intersection / union  # 返回列表，其值为各个类别的IoU
#         return IoU


#     def meanIntersectionOverUnion(self):
#         IoU = self.IntersectionOverUnion()
#         mIoU = IoU[IoU < float('inf')].mean()  # 求各类别IoU的平均
#         return mIoU
# if __name__ == '__main__':
#     true_dir = './test_img/true_mask/'
#     pred_dir = './test_img/pred_mask/'
#     true_files = glob.glob(true_dir+'*.jpg')
#     dice_score = 0

#     # pred_files = glob.glob(pred_dir+'*.png')
#     for true_file in tqdm(true_files):
#         img_true = cv2.imread(true_file)#加载gt

#         img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY).astype(int)#转为灰度图
#         true_name = true_file.split('/')[-1]#获取文件名
#         only_name,suffix = os.path.splitext(true_name)#获取文件名和后缀
#         pred_name = only_name+'.png'
#         img_pred = cv2.imread(os.path.join(pred_dir, pred_name))#加载预测结果
#         img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY).astype(int)#转为灰度图
#         image_height, image_width = img_pred.shape[0:2]
#         # 计 算G-mean
#         dice_metrics = dc(img_true, img_pred)
#         tp_idx, tp_idy = np.where(img_true > 0)
#         TP = np.sum(img_true[tp_idx, tp_idy] - img_pred[tp_idx, tp_idy] == 0)
#         FN = np.sum(img_true[tp_idx, tp_idy] - img_pred[tp_idx, tp_idy] != 0)
#         idx2, idy2 = np.where(img_true == 0)
#         TN = np.sum(img_true[idx2, idy2] - img_pred[idx2, idy2] == 0)
#         FP = np.sum(img_true[idx2, idy2] - img_pred[idx2, idy2] != 0)
#         jaccardIndex = (TP+0.00001)/(TP+FN+FP+0.00001)
#         Precision = (TP+0.00001)/(TP+FP+0.00001)
#         Recall = (TP+0.00001)/(TP+FN+0.00001)
#         f1 = (2*Precision*Recall+0.00001)/(Precision+Recall+0.00001)
#         OA = (TP+TN+0.00001)/(TP+TN+FP+FN+0.00001)
#         FF = (TN+0.00001)/(TN+FP+0.00001)
#         metric = SegmentationMetric(2)
#         # hist = metric.addBatch(torch.tensor(img_pred).long(), torch.tensor(img_true).long(),None)
#         mIoU = metric.meanIntersectionOverUnion()

#         with open('./metrics.txt', "a",encoding='utf-8') as f:
#                 # 记录每个epoch对应的train_loss、lr以及验证集各指标
#                 predict_info =f"处理图片: {true_name}.png\n" \
#                               f"[DICE: {dice_metrics:.4f}]\n" \
#                               f"[jaccardIndex: {jaccardIndex:.4f}]\n" \
#                               f"[ Precision : {Precision:.4f}]\n"\
#                                f"[ Recall: { Recall:.4f}]\n" \
#                                f"[f1: {f1:.4f}]\n" \
#                                f"[OA: {OA:.4f}]\n" \
#                               f"MIou: {mIoU :.10f}\n"
#                 f.write(predict_info + "\n\n")


