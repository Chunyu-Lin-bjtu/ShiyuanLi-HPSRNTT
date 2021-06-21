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

path = '/home/cylin/lsy/srntt-master/ori_result_4k_13_onlygan_nomap2_369/'
# path = '/home/cylin/lsy/srntt-master/ori_result_2k_13_onlygan_nomap2_369/'
indexxx = 1
# path = r'C:\Users\71966\Desktop\figure\temp'
filelist = os.listdir(path)
csv_data = []


def write_csv(psnr_all):
    # path = "img_psnr_2k_ori18_test_362.csv"
    path = "near_region_SRNTT_4k_part.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = psnr_all
        csv_write.writerow(psnr_all)


def cal_wspsnr(img1, img2):
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


def cal_psnr(img1, img2):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCR_CB)
    height = img1.shape[0]
    width = img1.shape[1]
    total_wmse = 0
    for i in range(height):
        for j in range(width):
            wmse = math.pow((int(img1[i, j, 0]) - int(img2[i, j, 0])), 2)
            total_wmse = total_wmse + wmse
    w_mse = total_wmse
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

    # img_hr1 = img_hr[int(height / 4):int(3 * height / 4), 0:int(width / 16), :]
    # img_hr2 = img_hr[int(height / 4):int(3 * height / 4), 0:int(2*width / 16), :]
    # img_hr3 = img_hr[int(height / 4):int(3 * height / 4), 0:int(3*width / 16), :]
    # img_hr4 = img_hr[int(height / 4):int(3 * height / 4), 0:int(4*width / 16), :]
    # img_hr5 = img_hr[int(height / 4):int(3 * height / 4), 0:int(5*width / 16), :]
    # img_hr6 = img_hr[int(height / 4):int(3 * height / 4), 0:int(6 * width / 16), :]
    # img_hr7 = img_hr[int(height / 4):int(3 * height / 4), 0:int(7 * width / 16), :]
    # img_hr8 = img_hr[int(height / 4):int(3 * height / 4), 0:int(8 * width / 16), :]

    img_hr1 = img_hr[int(height / 4):int(3 * height / 4), 0:int(width / 16), :]
    img_hr2 = img_hr[int(height / 4):int(3 * height / 4), int(width / 16):int(2 * width / 16), :]
    img_hr3 = img_hr[int(height / 4):int(3 * height / 4), int(2 * width / 16):int(3 * width / 16), :]
    img_hr4 = img_hr[int(height / 4):int(3 * height / 4), int(3 * width / 16):int(4 * width / 16), :]
    img_hr5 = img_hr[int(height / 4):int(3 * height / 4), int(4 * width / 16):int(5 * width / 16), :]
    img_hr6 = img_hr[int(height / 4):int(3 * height / 4), int(5 * width / 16):int(6 * width / 16), :]
    img_hr7 = img_hr[int(height / 4):int(3 * height / 4), int(6 * width / 16):int(7 * width / 16), :]
    img_hr8 = img_hr[int(height / 4):int(3 * height / 4), int(7 * width / 16):int(8 * width / 16), :]

    img_hr1 = np.array(img_hr1).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr2 = np.array(img_hr2).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr3 = np.array(img_hr3).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr4 = np.array(img_hr4).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr5 = np.array(img_hr5).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr6 = np.array(img_hr6).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr7 = np.array(img_hr7).squeeze().round().clip(0, 255).astype(np.uint8)
    img_hr8 = np.array(img_hr8).squeeze().round().clip(0, 255).astype(np.uint8)
    imsave(join(src, '1.png'), img_hr1)
    imsave(join(src, '2.png'), img_hr2)
    imsave(join(src, '3.png'), img_hr3)
    imsave(join(src, '4.png'), img_hr4)
    imsave(join(src, '5.png'), img_hr5)
    imsave(join(src, '6.png'), img_hr6)
    imsave(join(src, '7.png'), img_hr7)
    imsave(join(src, '8.png'), img_hr8)
    hhhhhhhhhh

    # img_srntt1 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(width / 16), :]
    # img_srntt2 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(2 * width / 16), :]
    # img_srntt3 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(3 * width / 16), :]
    # img_srntt4 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(4 * width / 16), :]
    # img_srntt5 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(5 * width / 16), :]
    # img_srntt6 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(6 * width / 16), :]
    # img_srntt7 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(7 * width / 16), :]
    # img_srntt8 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(8 * width / 16), :]

    img_srntt1 = img_srntt[int(height / 4):int(3 * height / 4), 0:int(width / 16), :]
    img_srntt2 = img_srntt[int(height / 4):int(3 * height / 4), int(width / 16):int(2 * width / 16), :]
    img_srntt3 = img_srntt[int(height / 4):int(3 * height / 4), int(2 * width / 16):int(3 * width / 16), :]
    img_srntt4 = img_srntt[int(height / 4):int(3 * height / 4), int(3 * width / 16):int(4 * width / 16), :]
    img_srntt5 = img_srntt[int(height / 4):int(3 * height / 4), int(4 * width / 16):int(5 * width / 16), :]
    img_srntt6 = img_srntt[int(height / 4):int(3 * height / 4), int(5 * width / 16):int(6 * width / 16), :]
    img_srntt7 = img_srntt[int(height / 4):int(3 * height / 4), int(6 * width / 16):int(7 * width / 16), :]
    img_srntt8 = img_srntt[int(height / 4):int(3 * height / 4), int(7 * width / 16):int(8 * width / 16), :]

    img_srntt1 = np.array(img_srntt1).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt2 = np.array(img_srntt2).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt3 = np.array(img_srntt3).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt4 = np.array(img_srntt4).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt5 = np.array(img_srntt5).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt6 = np.array(img_srntt6).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt7 = np.array(img_srntt7).squeeze().round().clip(0, 255).astype(np.uint8)
    img_srntt8 = np.array(img_srntt8).squeeze().round().clip(0, 255).astype(np.uint8)

    # print("img_hr", img_hr.dtype)

    psnr_SRNTT1 = sess.run(tf.image.psnr(img_hr1, img_srntt1, max_val=255))
    psnr_SRNTT2 = sess.run(tf.image.psnr(img_hr2, img_srntt2, max_val=255))
    psnr_SRNTT3 = sess.run(tf.image.psnr(img_hr3, img_srntt3, max_val=255))
    psnr_SRNTT4 = sess.run(tf.image.psnr(img_hr4, img_srntt4, max_val=255))
    psnr_SRNTT5 = sess.run(tf.image.psnr(img_hr5, img_srntt5, max_val=255))
    psnr_SRNTT6 = sess.run(tf.image.psnr(img_hr6, img_srntt6, max_val=255))
    psnr_SRNTT7 = sess.run(tf.image.psnr(img_hr7, img_srntt7, max_val=255))
    psnr_SRNTT8 = sess.run(tf.image.psnr(img_hr8, img_srntt8, max_val=255))

    wspsnr_SRNTT1 = cal_wspsnr(img_hr1, img_srntt1)
    wspsnr_SRNTT2 = cal_wspsnr(img_hr2, img_srntt2)
    wspsnr_SRNTT3 = cal_wspsnr(img_hr3, img_srntt3)
    wspsnr_SRNTT4 = cal_wspsnr(img_hr4, img_srntt4)
    wspsnr_SRNTT5 = cal_wspsnr(img_hr5, img_srntt5)
    wspsnr_SRNTT6 = cal_wspsnr(img_hr6, img_srntt6)
    wspsnr_SRNTT7 = cal_wspsnr(img_hr7, img_srntt7)
    wspsnr_SRNTT8 = cal_wspsnr(img_hr8, img_srntt8)
    row.append(psnr_SRNTT1)
    row.append(psnr_SRNTT2)
    row.append(psnr_SRNTT3)
    row.append(psnr_SRNTT4)
    row.append(psnr_SRNTT5)
    row.append(psnr_SRNTT6)
    row.append(psnr_SRNTT7)
    row.append(psnr_SRNTT8)

    row.append(wspsnr_SRNTT1)
    row.append(wspsnr_SRNTT2)
    row.append(wspsnr_SRNTT3)
    row.append(wspsnr_SRNTT4)
    row.append(wspsnr_SRNTT5)
    row.append(wspsnr_SRNTT6)
    row.append(wspsnr_SRNTT7)
    row.append(wspsnr_SRNTT8)
    print("row", row)
    write_csv(row)
