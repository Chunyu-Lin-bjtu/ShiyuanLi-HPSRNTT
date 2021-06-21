import tensorflow as tf
import os
import cv2
import math
import csv
import numpy as np


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


def write_csv(psnr_all):
    path = "part_psnr_test.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(psnr_all)


result_path = r'/home/cylin/lsy/srntt-master/result_2k_13_patchgan20_369/'
result_list = os.listdir(result_path)
print("num of result_list", len(result_list))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    for file_num in result_list:
        print(file_num)
        psnr_all = []
        imgs_path = os.path.join(result_path, file_num)
        img_bicubic = cv2.imread(os.path.join(imgs_path, 'Bicubic.png'))
        img_bicubic = cv2.cvtColor(img_bicubic, cv2.COLOR_BGR2RGB)
        h = img_bicubic.shape[0]
        w = img_bicubic.shape[1]
        img_bicubic_part = np.concatenate((img_bicubic[:,(int(w/2)-int(h/4)):w,:],img_bicubic[:,0:int(h/4),:]),axis=1)
        # cv2.imwrite("bicubic_part.png", img_bicubic_part)
        img_hr = cv2.imread(os.path.join(imgs_path, 'hr.png'))
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_hr_part = np.concatenate(
            (img_hr[:, (int(w / 2) - int(h / 4)):w, :], img_hr[:, 0:int(h / 4), :]), axis=1)
        # cv2.imwrite("hr_part.png", img_hr_part)
        img_upscale = cv2.imread(os.path.join(imgs_path, 'upscale.png'))
        img_upscale = cv2.cvtColor(img_upscale, cv2.COLOR_BGR2RGB)
        img_upscale_part = np.concatenate(
            (img_upscale[:, (int(w / 2) - int(h / 4)):w, :], img_upscale[:, 0:int(h / 4), :]), axis=1)
        # cv2.imwrite("upscale_part.png", img_upscale_part)

        img_SRNTT = cv2.imread(os.path.join(imgs_path, 'SRNTT.png'))
        img_SRNTT = cv2.cvtColor(img_SRNTT, cv2.COLOR_BGR2RGB)
        img_SRNTT_part = np.concatenate(
            (img_SRNTT[:, (int(w / 2) - int(h / 4)):w, :], img_SRNTT[:, 0:int(h / 4), :]), axis=1)
        # cv2.imwrite("SRNTT_part.png", img_SRNTT_part)

        # tf.reset_default_graph()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        psnr1 = sess.run(tf.image.psnr(img_hr_part, img_bicubic_part, max_val=255))
        psnr2 = sess.run(tf.image.psnr(img_hr_part, img_upscale_part, max_val=255))
        psnr3 = sess.run(tf.image.psnr(img_hr_part, img_SRNTT_part, max_val=255))
        full_hr_tensor = tf.convert_to_tensor(img_hr_part)
        full_bicubic_tensor = tf.convert_to_tensor(img_bicubic_part)
        full_upscale_tensor = tf.convert_to_tensor(img_upscale_part)
        full_SRNTT_tensor = tf.convert_to_tensor(img_SRNTT_part)
        ssim1 = sess.run(tf.image.ssim(full_hr_tensor, full_bicubic_tensor, max_val=255))
        ssim2 = sess.run(tf.image.ssim(full_hr_tensor, full_upscale_tensor, max_val=255))
        ssim3 = sess.run(tf.image.ssim(full_hr_tensor, full_SRNTT_tensor, max_val=255))
        psnr_all.append(file_num)
        psnr_all.append(psnr1)
        psnr_all.append(psnr2)
        psnr_all.append(psnr3)
        # psnr_all.append(ws_psnr_bic)
        # psnr_all.append(ws_psnr_up)
        # psnr_all.append(ws_psnr_srntt)

        psnr_all.append(ssim1)
        psnr_all.append(ssim2)
        psnr_all.append(ssim3)

        compare_path = os.path.join(result_path, file_num, 'compare')
        if (os.path.exists(compare_path)):
            img_bicubic = cv2.imread(os.path.join(compare_path, 'Bicubic.png'))
            img_bicubic = cv2.cvtColor(img_bicubic, cv2.COLOR_BGR2RGB)
            h = img_bicubic.shape[0]
            w = img_bicubic.shape[1]
            img_bicubic_part = img_bicubic[int(h / 4):h, :, :]
            img_bicubic_top = np.concatenate(
                (img_bicubic[0:int(h / 4), 0:int(w / 2), :], img_bicubic[0:int(h / 4), int(w / 2):w, :]), axis=0)
            img_bicubic_part[0:int(h / 2), int(w / 2):w, :] = img_bicubic_top

            img_hr = cv2.imread(os.path.join(compare_path, 'hr.png'))
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
            img_hr_part = img_hr[int(h / 4):h, :, :]
            img_hr_top = np.concatenate(
                (img_hr[0:int(h / 4), 0:int(w / 2), :], img_hr[0:int(h / 4), int(w / 2):w, :]), axis=0)
            img_hr_part[0:int(h / 2), int(w / 2):w, :] = img_hr_top

            img_upscale = cv2.imread(os.path.join(compare_path, 'upscale.png'))
            img_upscale = cv2.cvtColor(img_upscale, cv2.COLOR_BGR2RGB)
            img_upscale_part = img_upscale[int(h / 4):h, :, :]
            img_upscale_top = np.concatenate(
                (img_upscale[0:int(h / 4), 0:int(w / 2), :], img_upscale[0:int(h / 4), int(w / 2):w, :]), axis=0)
            img_upscale_part[0:int(h / 2), int(w / 2):w, :] = img_upscale_top

            img_SRNTT = cv2.imread(os.path.join(compare_path, 'SRNTT.png'))
            img_SRNTT = cv2.cvtColor(img_SRNTT, cv2.COLOR_BGR2RGB)
            img_SRNTT_part = img_SRNTT[int(h / 4):h, :, :]
            img_SRNTT_top = np.concatenate(
                (img_SRNTT[0:int(h / 4), 0:int(w / 2), :], img_SRNTT[0:int(h / 4), int(w / 2):w, :]), axis=0)
            img_SRNTT_part[0:int(h / 2), int(w / 2):w, :] = img_SRNTT_top
            # cv2.imwrite("bicubic_part2.png", img_bicubic_part)
            # cv2.imwrite("hr_part2.png", img_hr_part)
            # cv2.imwrite("upscale_part2.png", img_upscale_part)
            # cv2.imwrite("SRNTT_part2.png", img_SRNTT_part)
            psnr1 = sess.run(tf.image.psnr(img_hr_part, img_bicubic_part, max_val=255))
            psnr2 = sess.run(tf.image.psnr(img_hr_part, img_upscale_part, max_val=255))
            psnr3 = sess.run(tf.image.psnr(img_hr_part, img_SRNTT_part, max_val=255))
            full_hr_tensor = tf.convert_to_tensor(img_hr_part)
            full_bicubic_tensor = tf.convert_to_tensor(img_bicubic_part)
            full_upscale_tensor = tf.convert_to_tensor(img_upscale_part)
            full_SRNTT_tensor = tf.convert_to_tensor(img_SRNTT_part)
            ssim1 = sess.run(tf.image.ssim(full_hr_tensor, full_bicubic_tensor, max_val=255))
            ssim2 = sess.run(tf.image.ssim(full_hr_tensor, full_upscale_tensor, max_val=255))
            ssim3 = sess.run(tf.image.ssim(full_hr_tensor, full_SRNTT_tensor, max_val=255))

            psnr_all.append(psnr1)
            psnr_all.append(psnr2)
            psnr_all.append(psnr3)

            psnr_all.append(ssim1)
            psnr_all.append(ssim2)
            psnr_all.append(ssim3)
        else:
            psnr_all.append(psnr1)
            psnr_all.append(psnr2)
            psnr_all.append(psnr3)

            psnr_all.append(ssim1)
            psnr_all.append(ssim2)
            psnr_all.append(ssim3)
        write_csv(psnr_all)
