import cv2
import numpy as np
import os
import tensorflow as tf
import cv2
import math
import csv
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

# path = '/home/cylin/lsy/srntt-master/ori_result_4k_13_onlygan_nomap2_369/'
path = '/home/cylin/lsy/srntt-master/ori_result_2k_13_onlygan_nomap2_369/'
indexxx = 1
# path = r'C:\Users\71966\Desktop\figure\temp'
filelist = os.listdir(path)
csv_data = []
def write_csv(psnr_all):
    # path = "img_psnr_2k_ori18_test_362.csv"
    path = "psnr_wspsnr_near_SRNTT_2K.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = psnr_all
        csv_write.writerow(psnr_all)


def cal_psnr(img1, img2):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCR_CB)
    height = img1.shape[0]
    width = img1.shape[1]
    total_wmse = 0
    total_wij = 0
    for i in range(height):
        for j in range(width):
            wij = math.cos((i + 0.5 - height / 2) * math.pi / height)
            wmse = wij * math.pow((int(img1[i, j, 0]) - int(img2[i, j, 0])), 2)
            total_wmse = total_wmse + wmse
            total_wij = total_wij + wij
    w_mse = total_wmse / total_wij
    ws_psnr = 10 * math.log((65025 / w_mse), 10)
    return ws_psnr


with open('4k_all_selected2.csv', 'r') as csv_file:
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
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("now folder", csv_item)
    src = os.path.join(os.path.abspath(path), csv_item)
    image_list = os.listdir(src)
    print(image_list)
    image_hr = os.path.join(os.path.abspath(src), 'hr.png')
    image_srntt = os.path.join(os.path.abspath(src), 'SRNTT.png')
    img_hr = cv2.imread(image_hr)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    img_srntt = cv2.imread(image_srntt)
    img_srntt = cv2.cvtColor(img_srntt, cv2.COLOR_BGR2RGB)
    height, width = img_hr.shape[0:2]

    img_hr_resize = imresize(img_hr, .25, interp='bicubic')
    img_hr_near = imresize(img_hr_resize, 4., interp='nearest')
    # hr_top = img_hr_near[0:480, :, :]
    # hr_left = img_hr_near[480:1440, 0:1920, :]
    # hr_bot = img_hr_near[1440:1920, :, :]
    # hr_right = img_srntt[480:1440, 1920:3840, :]

    hr_top = img_hr_near[0:int(height/4), :, :]
    hr_left = img_hr_near[int(height/4):int(3*height/4), 0:int(width/2), :]
    hr_bot = img_hr_near[int(3*height/4):height, :, :]
    hr_right = img_srntt[int(height/4):int(3*height/4), int(width/2):width, :]

    # image_no_inter = resize_pyramid(image_hr)
    image_no_inter_center = np.concatenate((hr_left, hr_right),axis=1)
    image_no_inter_center_top = np.concatenate((hr_top, image_no_inter_center), axis=0)
    image_no_inter_all = np.concatenate((image_no_inter_center_top, hr_bot), axis=0)

    image_no_inter_all = np.array(image_no_inter_all).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr = np.array(img_hr).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt = np.array(img_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
    # print("img_hr", img_hr.dtype)
    # print("image_no_inter_all", image_no_inter_all.dtype)
    imsave(join(src, 'nearest.png'), image_no_inter_all)

    psnr_near = sess.run(tf.image.psnr(img_hr, image_no_inter_all, max_val=255))
    psnr_SRNTT = sess.run(tf.image.psnr(img_hr, img_srntt, max_val=255))
    print("psnr_near",psnr_near)
    print("psnr_SRNTT", psnr_SRNTT)
    wspsnr_near = cal_psnr(img_hr, image_no_inter_all)
    wspsnr_SRNTT = cal_psnr(img_hr, img_srntt)
    print("wspsnr_near", wspsnr_near)
    print("wspsnr_SRNTT", wspsnr_SRNTT)

    row.append(psnr_near)
    row.append(psnr_SRNTT)
    row.append(wspsnr_near)
    row.append(wspsnr_SRNTT)

    if (image_list.count('compare')>0):
        print("need compare")
        src = os.path.join(os.path.abspath(path), csv_item)
        src = os.path.join(os.path.abspath(src), 'compare')
        image_list = os.listdir(src)
        print(image_list)
        image_hr = os.path.join(os.path.abspath(src), 'hr.png')
        image_srntt = os.path.join(os.path.abspath(src), 'SRNTT.png')
        img_hr = cv2.imread(image_hr)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_srntt = cv2.imread(image_srntt)
        img_srntt = cv2.cvtColor(img_srntt, cv2.COLOR_BGR2RGB)

        img_hr_resize = imresize(img_hr, .25, interp='bicubic')
        img_hr_near = imresize(img_hr_resize, 4., interp='nearest')
        # hr_top = img_hr_near[0:480, :, :]
        # hr_left = img_hr_near[480:1440, 0:1920, :]
        # hr_bot = img_hr_near[1440:1920, :, :]
        # hr_right = img_srntt[480:1440, 1920:3840, :]
        hr_top = img_hr_near[0:int(height / 4), :, :]
        hr_left = img_hr_near[int(height / 4):int(3 * height / 4), 0:int(width / 2), :]
        hr_bot = img_hr_near[int(3 * height / 4):height, :, :]
        hr_right = img_srntt[int(height / 4):int(3 * height / 4), int(width / 2):width, :]
        # image_no_inter = resize_pyramid(image_hr)
        image_no_inter_center = np.concatenate((hr_left, hr_right), axis=1)
        image_no_inter_center_top = np.concatenate((hr_top, image_no_inter_center), axis=0)
        image_no_inter_all = np.concatenate((image_no_inter_center_top, hr_bot), axis=0)

        image_no_inter_all = np.array(image_no_inter_all).squeeze().round().clip(0, 255).astype(np.uint8)
        img_hr = np.array(img_hr).squeeze().round().clip(0, 255).astype(np.uint8)
        img_srntt = np.array(img_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
        # print("img_hr", img_hr.dtype)
        # print("image_no_inter_all", image_no_inter_all.dtype)
        # imsave(join(src, 'nearest.png'), image_no_inter_all)
        psnr_near = sess.run(tf.image.psnr(img_hr, image_no_inter_all, max_val=255))
        psnr_SRNTT = sess.run(tf.image.psnr(img_hr, img_srntt, max_val=255))
        wspsnr_near = cal_psnr(img_hr, image_no_inter_all)
        wspsnr_SRNTT = cal_psnr(img_hr, img_srntt)
        print("psnr_near", psnr_near)
        print("psnr_SRNTT", psnr_SRNTT)
        print("wspsnr_near", wspsnr_near)
        print("wspsnr_SRNTT", wspsnr_SRNTT)
        row.append(psnr_near)
        row.append(psnr_SRNTT)
        row.append(wspsnr_near)
        row.append(wspsnr_SRNTT)
    else:
        row.append(psnr_near)
        row.append(psnr_SRNTT)
        row.append(wspsnr_near)
        row.append(wspsnr_SRNTT)

    write_csv(row)
    sess.close()




