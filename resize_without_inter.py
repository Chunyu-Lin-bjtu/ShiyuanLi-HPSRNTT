import cv2
import numpy as np
import os
import tensorflow as tf
import cv2
import math
from tensorlayer import *
from tensorlayer.layers import *
from os.path import join, exists, split, isfile
from os import makedirs, environ
from shutil import rmtree
from glob import glob
from scipy.misc import imread, imresize, imsave, imrotate
from bicubic_kernel import back_projection_loss
import logging
from scipy.io import savemat
from scipy.signal import convolve2d

path = '/home/cylin/lsy/srntt-master/ori_result_4k_13_ori_nomap2_369/'
indexxx = 1
# path = r'C:\Users\71966\Desktop\figure\temp'
filelist = os.listdir(path)
csv_data = []
def write_csv(psnr_all):
    # path = "img_psnr_2k_ori18_test_362.csv"
    path = "psnr_near_SRNTT.csv"
    # path = "test_no_map1.csv"
    # 每次测试需要改path，保存目录，swap最后为0的map
    # 测完把cutline改回去，现在全设置为0了
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = psnr_all
        csv_write.writerow(psnr_all)


with open('name.csv', 'r') as csv_file:
    # next(csv_file)
    # next(csv_file)
    reader = csv.reader(csv_file)
    for i in range(indexxx - 1):
        next(csv_file)
    for row in reader:
        csv_data.append(row[0])
for csv_item in csv_data:
    row = []
    row.append(csv_item)
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    self.sess = tf.Session(config=config)
    print("当前处理文件夹", csv_item)
    src = os.path.join(os.path.abspath(path), csv_item)
    image_list = os.listdir(src)
    print(image_list)
    print(image_list)
    dddddd
    image_hr = os.path.join(os.path.abspath(src), 'hr.png')
    image_srntt = os.path.join(os.path.abspath(src), 'SRNTT.png')
    img_hr = cv2.imread(image_hr)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    img_srntt = cv2.imread(image_srntt)
    img_srntt = cv2.cvtColor(img_srntt, cv2.COLOR_BGR2RGB)

    img_hr_resize = imresize(img_hr, .25, interp='nearest')
    img_hr_near = imresize(img_hr_resize, 4, interp='nearest')
    hr_top = img_hr[0:480, :, :]
    hr_left = img_hr[480:1440, 0:1920, :]
    hr_bot = img_hr[1440:1920, :, :]
    hr_right = img_srntt[480:1440, 1920:3840, :]
    # image_no_inter = resize_pyramid(image_hr)
    image_no_inter = cv2.resize(image_hr, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    image_no_inter = cv2.resize(image_no_inter, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    image_no_inter_center = np.concatenate((hr_left, hr_right),axis=1)
    image_no_inter_center_top = np.concatenate((hr_top, image_no_inter_center), axis=0)
    image_no_inter_all = np.concatenate((image_no_inter_center_top, hr_bot), axis=0)

    image_no_inter_all = np.array(image_no_inter_all).squeeze().round().clip(0, 255).astype(np.uint8)
    imsave(join(src, 'nearest.png'), image_no_inter_all)

    psnr_near = sess.run(tf.image.psnr(img_hr, image_no_inter_all, max_val=255))
    psnr_SRNTT = sess.run(tf.image.psnr(img_hr, img_SRNTT, max_val=255))
    row.append(psnr_near)
    row.append(psnr_SRNTT)
    sess.close()




