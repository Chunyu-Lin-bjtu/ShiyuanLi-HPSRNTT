import os
import cv2
import csv
import tensorflow as tf
import numpy as np
from SRNTT.model import *
import argparse
from vgg19_trainable import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='SRNTT')

# init parameters
parser.add_argument('--is_train', type=str2bool, default=False)
parser.add_argument('--srntt_model_path', type=str, default='SRNTT/models/SRNTT')
parser.add_argument('--vgg19_model_path', type=str, default='SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat')
parser.add_argument('--save_dir', type=str, default=None, help='dir of saving intermediate training results')
parser.add_argument('--num_res_blocks', type=int, default=16, help='number of residual blocks')

# train parameters
# parser.add_argument('--input_dir', type=str, default='data/train/input', help='dir of input images')
parser.add_argument('--input_dir', type=str, default='data/train/input', help='dir of input images')
parser.add_argument('--input_truth_dir', type=str, default='data_line/input_truth', help='dir of input images')
# 测试阶段把input_dir自动划分，不需要输入ref_dir.
parser.add_argument('--ref_dir', type=str, default='data/train/ref', help='dir of reference images')
parser.add_argument('--map_dir', type=str, default='data/train/map_321', help='dir of texture maps of reference images')
parser.add_argument('--batch_size', type=int, default=9)
parser.add_argument('--num_init_epochs', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--use_pretrained_model', type=str2bool, default=True)
parser.add_argument('--use_init_model_only', type=str2bool, default=False,
                    help='effect if use_pretrained_model is true')
parser.add_argument('--w_per', type=float, default=1e-4,
                    help='weight of perceptual loss between output and ground truth')
parser.add_argument('--w_tex', type=float, default=1e-4, help='weight of texture loss between output and texture map')
parser.add_argument('--w_adv', type=float, default=1e-6, help='weight of adversarial loss')
parser.add_argument('--w_bp', type=float, default=0.0, help='weight of back projection loss')
parser.add_argument('--w_rec', type=float, default=1.0, help='weight of reconstruction loss')
parser.add_argument('--vgg_perceptual_loss_layer', type=str, default='relu5_1',
                    help='the VGG19 layer name to compute perceptrual loss')
parser.add_argument('--is_WGAN_GP', type=str2bool, default=True, help='whether use WGAN-GP')
parser.add_argument('--is_L1_loss', type=str2bool, default=True, help='whether use L1 norm')
parser.add_argument('--param_WGAN_GP', type=float, default=10, help='parameter for WGAN-GP')
parser.add_argument('--input_size', type=int, default=40)
parser.add_argument('--use_weight_map', type=str2bool, default=False)
parser.add_argument('--use_lower_layers_in_per_loss', type=str2bool, default=False)

# test parameters
parser.add_argument('--result_dir', type=str, default='result', help='dir of saving testing results')
parser.add_argument('--ref_scale', type=float, default=1.0)
parser.add_argument('--is_original_image', type=str2bool, default=True)

# predict parameters
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

args = parser.parse_args()


# def create_csv():
#     path = "img_psnr_2k_patchgan20_good_change.csv"
#     with open(path, 'wb') as f:
#         csv_write = csv.writer(f)
#     return path
#
#
# def write_csv(name, bic, srgan, srntt, ws_psnr_bic, ws_psnr_up, ws_psnr_srntt, bic2, srgan2, srntt2, ws_psnr_bic2, ws_psnr_up2, ws_psnr_srntt2, cut_line,
#               flag):
#     path = "img_psnr_2k_patchgan20_good_change.csv"
#     with open(path, 'a+') as f:
#         csv_write = csv.writer(f)
#         data_row = [name, bic, srgan, srntt, ws_psnr_bic, ws_psnr_up, ws_psnr_srntt, bic2, srgan2, srntt2, ws_psnr_bic2, ws_psnr_up2, ws_psnr_srntt2, cut_line,
#               flag]
#         csv_write.writerow(data_row)

# def create_csv_force():
#     path = "force_cut_line.csv"
#     with open(path, 'wb') as f:
#         csv_write = csv.writer(f)
#     return path


def write_csv_force(name, bic, srgan, srntt, ws_psnr_bic, ws_psnr_up, ws_psnr_srntt, cut_line):
    path = "force_cut_line.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [name, bic, srgan, srntt, ws_psnr_bic, ws_psnr_up, ws_psnr_srntt, cut_line]
        csv_write.writerow(data_row)


def write_csv(psnr_all):
    # path = "img_psnr_2k_ori18_test_362.csv"
    path = "img_psnr_ssim_3k_13_onlygan_300_all.csv"
    # path = "test_no_map2"
    # 记得改下面的index，改成从1开始，现在是114
    # 测完后把cutline改回去，把swap的map改回去
    # 如果要测2的2k情况，还需要改图片文件路径、切割线的csv文件、图片大小
    # path = "img_psnr_ssim_2k_1_patchgan20_overlap_301_all.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = psnr_all
        csv_write.writerow(psnr_all)



if args.is_train:

    # record parameters to file
    if args.save_dir is None:
        args.save_dir = 'default_save_dir'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'arguments.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            line = '{:>30}\t{:<10}\n'.format(arg, getattr(args, arg))
            bar = ''
            f.write(line)
        f.close()

    srntt = SRNTT(
        srntt_model_path=args.srntt_model_path,
        vgg19_model_path=args.vgg19_model_path,
        save_dir=args.save_dir,
        num_res_blocks=args.num_res_blocks
    )

    # srntt.train_get_rid_of_line(
    #     input_dir=args.input_dir,
    #     input_dir_truth=args.input_truth_dir,
    #     batch_size=args.batch_size,
    #     num_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     beta1=args.beta1,
    #     use_pretrained_model=args.use_pretrained_model,
    #     is_L1_loss=args.is_L1_loss,
    #     input_size=args.input_size,
    # )

    srntt.train(
        input_dir=args.input_dir,
        ref_dir=args.ref_dir,
        map_dir=args.map_dir,
        batch_size=args.batch_size,
        num_init_epochs=args.num_init_epochs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        use_pretrained_model=args.use_pretrained_model,
        use_init_model_only=args.use_init_model_only,
        weights=(args.w_per, args.w_tex, args.w_adv, args.w_bp, args.w_rec),
        vgg_perceptual_loss_layer=args.vgg_perceptual_loss_layer,
        is_WGAN_GP=args.is_WGAN_GP,
        is_L1_loss=args.is_L1_loss,
        param_WGAN_GP=args.param_WGAN_GP,
        input_size=args.input_size,
        use_weight_map=args.use_weight_map,
        use_lower_layers_in_per_loss=args.use_lower_layers_in_per_loss
    )
else:
    if args.save_dir is not None:
        # read recorded arguments
        fixed_arguments = ['srntt_model_path', 'vgg19_model_path', 'save_dir', 'num_res_blocks', 'use_weight_map']
        if os.path.exists(os.path.join(args.save_dir, 'arguments.txt')):
            with open(os.path.join(args.save_dir, 'arguments.txt'), 'r') as f:
                for arg, line in zip(sorted(vars(args)), f.readlines()):
                    arg_name, arg_value = line.strip().split('\t')
                    if arg_name in fixed_arguments:
                        fixed_arguments.remove(arg_name)
                        try:
                            if isinstance(getattr(args, arg_name), bool):

                                setattr(args, arg_name, str2bool(arg_value))
                            else:
                                setattr(args, arg_name, type(getattr(args, arg_name))(arg_value))
                        except:
                            print('Unmatched arg_name: %s!' % arg_name)

    srntt = SRNTT(
        srntt_model_path=args.srntt_model_path,
        vgg19_model_path=args.vgg19_model_path,
        save_dir=args.save_dir,
        num_res_blocks=args.num_res_blocks,
    )
    # csv_psnr = create_csv_force()
    folder_ref = '/home/cylin/lsy/srntt-master/test_8k_3/'
    # folder_ref = '/home/cylin/lsy/srntt-master/test_8k_2/'
    result_path = args.result_dir
    filelist = os.listdir(folder_ref)
    print("num of test image", len(filelist))
    # indexxx = 1
    indexxx = 1
    psnr1 = []
    psnr2 = []
    psnr3 = []
    psnr11 = []
    psnr22 = []
    psnr33 = []
    csv_data = []
    # with open('cut_8k_13_relu31_8500_369.csv', 'r') as csv_file:
    # with open('cut_8k_2_relu31_8500_301.csv', 'r') as csv_file:
    # name has 300 images
    with open('name.csv', 'r') as csv_file:
        # next(csv_file)
        # next(csv_file)
        reader = csv.reader(csv_file)
        for i in range(indexxx - 1):
            next(csv_file)
        for row in reader:
            csv_data.append(row)

    # for item in filelist:
    for csv_item in csv_data:
        psnr_all = []
        tf.reset_default_graph()
        item = csv_item[0]
        result_path = args.result_dir
        # create_path = os.path.join(result_path, str(indexxx))
        # if not os.path.exists(create_path):  # 如果路径不存在
        #     os.makedirs(create_path)
        #     print("result文件夹已创建", create_path)
        img_path = os.path.join(folder_ref, item)
        print("img path", img_path)
        image_ori_8k = cv2.imread(img_path)
        h_ori = image_ori_8k.shape[0]
        w_ori = image_ori_8k.shape[1]
        # image_ori = cv2.resize(image_ori, (int(w_ori / 2), int(h_ori / 2)))
        # image_ori = cv2.resize(image_ori_8k, (1920, 960))
        # image_ori = cv2.resize(image_ori_8k, (3840, 1920))
        image_ori = cv2.resize(image_ori_8k, (2880, 1440))
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        img_ori_h = image_ori.shape[0]
        img_ori_w = image_ori.shape[1]
        half_img_ori_w = int(img_ori_w/2)
        a_third_img_ori_h = int(img_ori_h / 4)
        two_third_img_ori_h = int(img_ori_h*3 / 4)
        level = int(img_ori_w/10)
        print("img_ori_h", img_ori_h)
        print("img_ori_w ", img_ori_w )

        # 960*1920测试代码---begin
        # cut_line = int(int(csv_item[1]) * 192)
        # print("cut_line", cut_line)
        cut_index = 0

        # for cut_line in range(0, 1920, 192):
        # cut_line = int(int(csv_item[2]) * level)
        cut_line = int((int(csv_item[1]) / 384) * level)
        # cut_line = 0
        print("now cut_line:", cut_line)
        # # 跑完了就把这些删掉
        # if(cut_line==0):
        #     print("abandon the 0 cut_line")
        #     continue
        # cut_line = 0
        # print("now cut_line:", cut_line)
        # # -----------------end
        image_left = image_ori[:, 0:cut_line, :]
        image_right = image_ori[:, cut_line:img_ori_w, :]
        img_new = np.concatenate((image_right, image_left), axis=1)
        # 默认右边的做ref
        # img_1 = img_new[240:720, 0:960, :]
        # img_2 = img_new[240:720, 960:1920, :]
        # LR_top = np.concatenate((img_new[0:240, 0:960, :], img_new[0:240, 960:1920, :]), axis=0)
        # LR_bottom = np.concatenate((img_new[720:960, 0:960, :], img_new[720:960, 960:1920, :]), axis=0)
        overlap = 8
        # img_1 = img_new[240 - overlap:720 + overlap, 0:960 + overlap, :]
        # img_2 = img_new[240 - overlap:720 + overlap, 960 - overlap:img_ori_w, :]
        img_1 = img_new[a_third_img_ori_h - overlap:two_third_img_ori_h + overlap, 0:half_img_ori_w + overlap, :]
        img_2 = img_new[a_third_img_ori_h - overlap:two_third_img_ori_h + overlap, half_img_ori_w - overlap:img_ori_w, :]

        # LR_top = np.concatenate(
        #     (img_new[0:240 + overlap, 0:960 + overlap, :], img_new[0:240 + overlap, 960 - overlap:1920, :]), axis=0)
        # LR_bottom = np.concatenate(
        #     (img_new[720 - overlap:960, 0:960 + overlap, :], img_new[720 - overlap:960, 960 - overlap:1920, :]), axis=0)
        LR_top = np.concatenate(
            (img_new[0:a_third_img_ori_h + overlap, 0:half_img_ori_w + overlap, :], img_new[0:a_third_img_ori_h + overlap, half_img_ori_w - overlap:img_ori_w, :]), axis=0)
        LR_bottom = np.concatenate(
            (img_new[two_third_img_ori_h - overlap:img_ori_h, 0:half_img_ori_w + overlap, :], img_new[two_third_img_ori_h - overlap:img_ori_h, half_img_ori_w - overlap:img_ori_w, :]), axis=0)
        print("img_1",img_1.shape)
        print("img_2", img_2.shape)
        print("img_top", LR_top.shape)
        print("img_bot", LR_bottom.shape)
        # hhh
        img_LR = np.concatenate((img_1, LR_top, LR_bottom), 1)
        img_ref = img_2
        flag = 2
        print("img_LR",img_LR.shape)
        print("img_ref",img_ref.shape)

        result_path = os.path.join(result_path, str(indexxx))
        if os.path.exists(result_path):
            print("result folder exist", result_path)
        else:
            os.mkdir(result_path)
            print("result folder is buliding", result_path)

        psnr_bic, psnr_up, psnr_srntt, ws_psnr_bic, ws_psnr_up, ws_psnr_srntt, ssim1, ssim2, ssim3, wssim1, wssim2, wssim3 = srntt.test(
            input_dir=img_LR,
            ref_dir=img_ref,
            overlap=overlap,
            hr_full=img_new,
            use_pretrained_model=args.use_pretrained_model,
            use_init_model_only=args.use_init_model_only,
            use_weight_map=args.use_weight_map,
            result_dir=result_path,
            ref_scale=args.ref_scale,
            is_original_image=args.is_original_image
        )
        psnr_all.append(item)
        psnr_all.append(cut_line)
        psnr_all.append(psnr_bic)
        psnr_all.append(psnr_up)
        psnr_all.append(psnr_srntt)
        psnr_all.append(ws_psnr_bic)
        psnr_all.append(ws_psnr_up)
        psnr_all.append(ws_psnr_srntt)

        psnr_all.append(ssim1)
        psnr_all.append(ssim2)
        psnr_all.append(ssim3)
        psnr_all.append(wssim1)
        psnr_all.append(wssim2)
        psnr_all.append(wssim3)



        if (cut_line != 0):
            print("zero is needed")
            result_path = args.result_dir
            result_path = os.path.join(result_path, str(indexxx), "compare")
            #
            # LR_top = np.concatenate((image_ori[0:240, 0:960, :], image_ori[0:240, 960:1920, :]), axis=0)
            # LR_bottom = np.concatenate((image_ori[720:960, 0:960, :], image_ori[720:960, 960:1920, :]), axis=0)
            # img_LR2 = np.concatenate((image_ori[240:720, 0:960, :], LR_top, LR_bottom), 1)
            # img_ref2 = image_ori[240:720, 960:1920:]

            img_1 = image_ori[a_third_img_ori_h - overlap:two_third_img_ori_h + overlap, 0:half_img_ori_w + overlap, :]
            img_2 = image_ori[a_third_img_ori_h - overlap:two_third_img_ori_h + overlap, half_img_ori_w - overlap:img_ori_w, :]
            # LR_top = np.concatenate(
            #     (image_ori[0:240 + overlap, 0:960 + overlap, :], image_ori[0:240 + overlap, 960 - overlap:1920, :]), axis=0)
            # LR_bottom = np.concatenate(
            #     (image_ori[720 - overlap:960, 0:960 + overlap, :], image_ori[720 - overlap:960, 960 - overlap:1920, :]),
            #     axis=0)
            LR_top = np.concatenate(
                (image_ori[0:a_third_img_ori_h + overlap, 0:half_img_ori_w + overlap, :],
                 image_ori[0:a_third_img_ori_h + overlap, half_img_ori_w - overlap:img_ori_w, :]), axis=0)
            LR_bottom = np.concatenate(
                (image_ori[two_third_img_ori_h - overlap:img_ori_h, 0:half_img_ori_w + overlap, :],
                 image_ori[two_third_img_ori_h - overlap:img_ori_h, half_img_ori_w - overlap:img_ori_w, :]), axis=0)

            img_LR2 = np.concatenate((img_1, LR_top, LR_bottom), 1)
            img_ref2 = img_2

            # 960*1920测试代码--end
            psnr_bic2, psnr_up2, psnr_srntt2, ws_psnr_bic2, ws_psnr_up2, ws_psnr_srntt2, ssim11, ssim22, ssim33, wssim11, wssim22, wssim33 = srntt.test(
                input_dir=img_LR2,
                ref_dir=img_ref2,
                overlap=overlap,
                hr_full=image_ori,
                use_pretrained_model=args.use_pretrained_model,
                use_init_model_only=args.use_init_model_only,
                use_weight_map=args.use_weight_map,
                result_dir=result_path,
                ref_scale=args.ref_scale,
                is_original_image=args.is_original_image
            )
            psnr_all.append(psnr_bic2)
            psnr_all.append(psnr_up2)
            psnr_all.append(psnr_srntt2)
            psnr_all.append(ws_psnr_bic2)
            psnr_all.append(ws_psnr_up2)
            psnr_all.append(ws_psnr_srntt2)
            psnr_all.append(ssim11)
            psnr_all.append(ssim22)
            psnr_all.append(ssim33)
            psnr_all.append(wssim11)
            psnr_all.append(wssim22)
            psnr_all.append(wssim33)
        else:
            psnr_all.append(psnr_bic)
            psnr_all.append(psnr_up)
            psnr_all.append(psnr_srntt)
            psnr_all.append(ws_psnr_bic)
            psnr_all.append(ws_psnr_up)
            psnr_all.append(ws_psnr_srntt)

            psnr_all.append(ssim1)
            psnr_all.append(ssim2)
            psnr_all.append(ssim3)
            psnr_all.append(wssim1)
            psnr_all.append(wssim2)
            psnr_all.append(wssim3)

        cut_index = cut_index + 1
        # write_csv_force(item, psnr_bic, psnr_up, psnr_srntt, ws_psnr_bic, ws_psnr_up, ws_psnr_srntt, cut_line)
        write_csv(psnr_all)
        indexxx = indexxx + 1


