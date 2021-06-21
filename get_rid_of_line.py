import os
import shutil
import cv2
import numpy as np

# 注释掉的是提取SRNTT图的代码
# img_folder = '/media/zy/55d6108f-5507-4552-977b-a5fbda209f8d/DocuClassLin/lsy/SRNTT-master/demo_testing_patchgan20_force_cutline/'
# count = 0
# new_path = '/media/zy/55d6108f-5507-4552-977b-a5fbda209f8d/DocuClassLin/lsy/SRNTT-master/data_line/input_truth/'
# for folder in os.listdir(img_folder):
#     img_all_path = os.path.join(img_folder, folder)
#     for index in os.listdir(os.path.join(img_folder, folder)):
#         img_single_path = os.path.join(img_all_path, index)
#         name_list = os.listdir(img_single_path)
#         for name in name_list:
#             if name=='hr.png':
#                 shutil.copy(os.path.join(img_single_path, name),
#                             os.path.join(new_path, str(count).zfill(5) + '.png'))
#         count += 1
#     print(count)

path = '/media/zy/55d6108f-5507-4552-977b-a5fbda209f8d/DocuClassLin/lsy/SRNTT-master/data_line/input/'
srntt_list = os.listdir(path)
img_path = os.path.join(path, srntt_list[0])
img = cv2.imread(img_path)
print(srntt_list[0])
print(img.shape)


# block = np.zeros((960, 4, 3))
for x in range(0, 1920):
    img[x, 240, :]=(img[x, 241, :]+img[x, 239, :])/2
