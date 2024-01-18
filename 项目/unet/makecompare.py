import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image  # 如果还未安装PIL库，可以使用 pip install pillow 进行安装
import glob
import os
img_dir = './test1/test/img/'
pred_dir = './test1/test/pred/'
true_dir = './test1/test/mask/'
res_dir = './test1/test/res/'
img_files = glob.glob(img_dir + '*.jpg')
for img_file in img_files:
    img = Image.open(img_file)
    _,real_name = os.path.split(img_file)
    only_name,_ = os.path.splitext(real_name)
    pred_path = pred_dir + only_name+'_mask.png'
    true_path = true_dir + only_name+'_mask.jpg'
    pred = Image.open(pred_path)
    true = Image.open(true_path)
# 读取三张图片

# 创建一个2行3列的图像网格
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])

    # 在图像网格的第一行显示img
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.set_title('img')
    ax0.axis('off')

    # 在图像网格的第一行显示pred
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(pred)
    ax1.set_title('pred')
    ax1.axis('off')

    # 在图像网格的第一行显示true
    ax2 = plt.subplot(gs[0, 2])
    ax2.imshow(true)
    ax2.set_title('true')
    ax2.axis('off')

    # 在图像网格的第二行创建一张空白图像，用于显示列名
    ax3 = plt.subplot(gs[1, :])
    ax3.axis('off')
    ax3.text(0.5, 0.5, 'img', va='center', ha='center', fontweight='bold')
    ax3.text(1.5, 0.5, 'pred', va='center', ha='center', fontweight='bold')
    ax3.text(2.5, 0.5, 'true', va='center', ha='center', fontweight='bold')
    # 调整布局
    plt.tight_layout()
    # 保存合并后的图片
    plt.savefig(res_dir + only_name+'_show.png')

