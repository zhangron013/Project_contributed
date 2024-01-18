import os
import shutil
import tqdm
import glob

# source_folder = './imgs'
# dir2 = './masks/'
# all_files = os.listdir(source_folder)
# #files2 = os.listdir(dir2)
# files2 = glob.glob(dir2+'*.jpg')
# print(files2)
# print(len(files2))
# ex_list = []
# for files in tqdm.tqdm(all_files):
#      _,name = os.path.split(files)

#      real_name= name.split('.jpg')[0]
#      mask_name = os.path.join(dir2,real_name+'.jpg')
#      if mask_name not in files2:
#          ex_list.append(mask_name)
# print(ex_list)
# print(len(ex_list))
import os

def check_matching_files(folder1, folder2):
    # 获取文件夹2中所有文件的文件名
    files_in_folder2 = set(os.listdir(folder2))

    # 遍历文件夹1中的文件
    for file_name in tqdm.tqdm(os.listdir(folder1)):
        #print(file_name)
        real_name = file_name.split('.jpg')[0]
        mask_name = real_name + '_mask.jpg'
        if mask_name not in files_in_folder2:
            file_path = os.path.join(folder1, file_name)
            os.remove(file_path)
            print(f'{file_name} exists in both folders.')

# 替换这里的路径为你的实际路径
folder1_path = './imgs'
folder2_path = './masks'

check_matching_files(folder1_path, folder2_path)


