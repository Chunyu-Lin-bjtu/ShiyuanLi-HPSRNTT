import tensorflow as tf
from SRNTT.tensorlayer import *
import numpy as np
from glob import glob
from os.path import exists, join, split, realpath, dirname
from os import makedirs
from SRNTT.model import *
from SRNTT.vgg19 import *
from SRNTT.swap360 import *
from scipy.misc import imread, imresize
import argparse
import csv

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser('offline_patchMatch_textureSwap')
parser.add_argument('--data_folder', type=str, default='data/train/CUFED', help='The dir of dataset: CUFED or DIV2K')
args = parser.parse_args()

data_folder = '/media/zy/55d6108f-5507-4552-977b-a5fbda209f8d/DocuClassLin/lsy/contextualLoss-master/move/new_2k/'
input_size_w = 240
input_size_h = 120
input_path = data_folder
# 是否要去掉2_1？在比较特征时，是否需要先下采样再上采样？
# matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
matching_layer = ['relu3_1', 'relu1_1']
input_files = sorted(glob(join(input_path, '*.jpg')))
n_files = len(input_files)
print('长度:', n_files)

vgg19_model_path = 'SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat'
tf_input = tf.placeholder(dtype=tf.float32, shape=[1, input_size_h, input_size_w, 3])
srntt = SRNTT(vgg19_model_path=vgg19_model_path)
net_upscale, _ = srntt.model(tf_input / 127.5 - 1, is_train=False)
net_vgg19 = VGG19(model_path=vgg19_model_path)
swaper = Swap()


def write_corr_list(row):
    path = "corr_list_normalize.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(row)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    model_path = join(dirname(realpath(__file__)), 'SRNTT', 'models', 'SRNTT', 'upscale.npz')
    if files.load_and_assign_npz(
            sess=sess,
            name=model_path,
            network=net_upscale) is False:
        raise Exception('FAILED load %s' % model_path)
    else:
        print('SUCCESS load %s' % model_path)
    print_format = '%%0%dd/%%0%dd' % (len(str(n_files)), len(str(n_files)))
    corr_list = []
    csv_data_name = []
    i = 0
    # for i in range(n_files):
    with open('label_cut_full_wspsnr.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            csv_data_name.append(row[0])
        print("csv文件数量", len(csv_data_name))
    for item in csv_data_name:
        print(print_format % (i + 1, n_files))
        corr_list.append(item)
        img_path = os.path.join(data_folder, item)
        full_image = imread(img_path, mode='RGB')
        height = full_image.shape[0]
        width = full_image.shape[1]
        lr_top = np.concatenate((full_image[0:int(height / 4), 0:int(width / 2), :],
                                 full_image[0:int(height / 4), int(width / 2):width, :]), axis=0)
        lr_bottom = np.concatenate((full_image[int(3 * height / 4):height, 0:int(width / 2), :],
                                    full_image[int(3 * height / 4):height, int(width / 2):width, :]), axis=0)
        for cut_line in range(0, width, int(width / 10)):
            print("当前分割线：", cut_line)
            img_left = full_image[:, 0:cut_line, :]
            img_right = full_image[:, cut_line:width, :]
            full_image_after_move = np.concatenate((img_right, img_left), axis=1)
            # 他们四个都是480*960
            lr_center = full_image_after_move[int(height / 4):int(3 * height / 4), 0:int(width / 2), :]
            # lr_top = np.concatenate((full_image[0:int(height / 4), 0:int(width / 2), :],
            #                          full_image[0:int(height / 4), int(width / 2):width, :]), axis=0)
            # lr_bottom = np.concatenate((full_image[int(3 * height / 4):height, 0:int(width / 2), :],
            #                             full_image[int(3 * height / 4):height, int(width / 2):width, :]), axis=0)
            ref = full_image_after_move[int(height / 4):int(3 * height / 4), int(width / 2):width, :]
            # imsave('center_%02d.png' % cut_line, lr_center)
            # imsave('top_%02d.png' % cut_line, lr_top)
            # imsave('bottom_%02d.png' % cut_line, lr_bottom)
            # imsave('ref_%02d.png' % cut_line, ref)

            # 低分辨率输入
            img_in_lr_center = imresize(lr_center, (input_size_h, input_size_w), interp='bicubic')
            img_in_lr_top = imresize(lr_top, (input_size_h, input_size_w), interp='bicubic')
            img_in_lr_bottom = imresize(lr_bottom, (input_size_h, input_size_w), interp='bicubic')
            # 原大小和缩小四倍的ref
            img_ref = imresize(ref, (input_size_h * 4, input_size_w * 4), interp='bicubic')
            img_ref_lr = imresize(img_ref, (input_size_h, input_size_w), interp='bicubic')
            # 网络插值得到的LR/ref（长宽各四倍）
            img_in_center_sr = (net_upscale.outputs.eval({tf_input: [img_in_lr_center]})[0] + 1) * 127.5
            img_in_top_sr = (net_upscale.outputs.eval({tf_input: [img_in_lr_top]})[0] + 1) * 127.5
            img_in_bottom_sr = (net_upscale.outputs.eval({tf_input: [img_in_lr_bottom]})[0] + 1) * 127.5
            img_ref_sr = (net_upscale.outputs.eval({tf_input: [img_ref_lr]})[0] + 1) * 127.5

            # get feature maps via VGG19
            # [1,40,40,256] [3,40,40,256] [1,40,40,256]
            # matching_layer的取值['relu3_1', 'relu2_1', 'relu1_1']
            # map_ref[0] [1,40,40,256] ; map_ref[1] [1,80,80,128] ; map_ref[2] [1,160,160,64]
            map_in_center_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_in_center_sr,
                                                          layer_name=matching_layer)
            map_in_top_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_in_top_sr,
                                                       layer_name=matching_layer)
            map_in_bottom_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_in_bottom_sr,
                                                          layer_name=matching_layer)

            map_ref = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref, layer_name=matching_layer[0])
            map_ref_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref_sr, layer_name=matching_layer)

            # patch matching and swapping
            # [2,80,80,128]
            # other_style = []
            # for m in map_ref[1:]:
            #     other_style.append([m])

            other_condition = []
            for s in map_ref_sr[1:]:
                other_condition.append([s])

            other_center = []
            for n in map_in_center_sr[1:]:
                other_center.append([n])

            other_top = []
            for e in map_in_top_sr[1:]:
                other_top.append([e])

            other_bottom = []
            for r in map_in_bottom_sr[1:]:
                other_bottom.append([r])

            # print("shape of map_in_sr", len(map_in_sr))
            # print("shape of map_ref", len(map_ref))
            # print("shape of map_ref[0]", len(map_ref[0]))
            # print("shape of map_ref_sr", len(map_ref_sr))
            # print("shape of other_style", len(other_style))
            # print("shape of lr_center", lr_center.shape)
            # print("shape of lr_top", lr_top.shape)
            # print("shape of lr_bottom", lr_bottom.shape)
            # print("shape of ref", ref.shape)
            # print("shape of map_in_sr", map_in_sr.shape)
            # print("shape of map_ref[0]", np.array(map_ref[0]).shape)
            # print("shape of map_ref[1]", np.array(map_ref[1]).shape)
            # print("shape of map_ref[2]", np.array(map_ref[2]).shape)
            # print("shape of map_in_center_sr[0]", np.array(map_in_center_sr[0]).shape)
            # print("shape of map_in_center_sr[1]", np.array(map_in_center_sr[1]).shape)
            # print("shape of map_in_center_sr[2]", np.array(map_in_center_sr[2]).shape)
            # print("shape of map_in_top_sr[0]", np.array(map_in_top_sr[0]).shape)
            # print("shape of map_in_top_sr[1]", np.array(map_in_top_sr[1]).shape)
            # print("shape of map_in_top_sr[2]", np.array(map_in_top_sr[2]).shape)
            # print("shape of map_ref", np.array(map_ref).shape)
            # print("shape of map_ref_sr[0]", np.array(map_ref_sr[0]).shape)
            # print("shape of map_ref_sr[1]", np.array(map_ref_sr[1]).shape)
            # print("shape of map_ref_sr[2]", np.array(map_ref_sr[2]).shape)
            # print("shape of other_condition", len(other_condition))
            # print("shape of other_center", len(other_center))
            # print("shape of other_top", len(other_top))
            # print("shape of other_bottom", len(other_bottom))


            corr = swaper.conditional_swap_multi_layer(
                content=[map_in_center_sr[0], map_in_top_sr[0], map_in_bottom_sr[0]],
                style=[map_ref],
                condition=[map_ref_sr[0]],
                other_conditions=other_condition,
                other_centers=other_center,
                other_tops=other_top,
                other_bottoms=other_bottom
            )
            print("relu3_1和relu1_1的相关性总和",corr)
            corr_list.append(corr)
        i = i + 1
        write_corr_list(corr_list)
        corr_list = []

        # save maps
        # np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)
