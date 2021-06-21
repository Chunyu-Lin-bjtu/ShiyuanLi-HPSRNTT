import tensorflow as tf
from SRNTT.tensorlayer import *
import numpy as np
from glob import glob
from os.path import exists, join, split, realpath, dirname
from os import makedirs
from SRNTT.model import *
from SRNTT.vgg19 import *
from SRNTT.swap import *
from scipy.misc import imread, imresize
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser('offline_patchMatch_textureSwap')
parser.add_argument('--data_folder', type=str, default='data/train/CUFED', help='The dir of dataset: CUFED or DIV2K')
args = parser.parse_args()

data_folder = args.data_folder
if 'CUFED' in data_folder:
    input_size = 40
elif 'DIV2K' in data_folder:
    input_size = 80
# elif '360image' in data_folder:
#     input_size = 60
elif '360image_2k_60' in data_folder:
    input_size = 60
else:
    raise Exception('Unrecognized dataset!')

input_path = join(data_folder, 'input')
# ref_path = join(data_folder, 'ref')
ref_path = join(data_folder, 'ref')
matching_layer = ['relu3_1', 'relu2_1','relu1_1']
# matching_layer = ['relu1_1']
# save_path = join(data_folder, 'map_321')
save_path = join(data_folder, 'map_321_temp')
if not exists(save_path):
    makedirs(save_path)

input_files = sorted(glob(join(input_path, '*.jpg')),reverse=True)
ref_files = sorted(glob(join(ref_path, '*.jpg')),reverse=True)
n_files = len(input_files)
print('len',n_files)
assert n_files == len(ref_files)

vgg19_model_path = 'SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat'
tf_input = tf.placeholder(dtype=tf.float32, shape=[1, input_size, input_size, 3])
srntt = SRNTT(vgg19_model_path=vgg19_model_path)
net_upscale, _ = srntt.model(tf_input / 127.5 - 1, is_train=False)
net_vgg19 = VGG19(model_path=vgg19_model_path)
swaper = Swap()

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
    for i in range(n_files):
        file_name = join(save_path, split(input_files[i])[-1].replace('.png', '.npz'))
        if exists(file_name):
            print("this file already exists")
            continue
        print(print_format % (i + 1, n_files))
        img_in_lr = imresize(imread(input_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
        img_ref = imresize(imread(ref_files[i], mode='RGB'), (input_size * 4, input_size * 4), interp='bicubic')
        img_ref_lr = imresize(img_ref, (input_size, input_size), interp='bicubic')
        img_in_sr = (net_upscale.outputs.eval({tf_input: [img_in_lr]})[0] + 1) * 127.5
        img_ref_sr = (net_upscale.outputs.eval({tf_input: [img_ref_lr]})[0] + 1) * 127.5

        # get feature maps via VGG19
        # [1,40,40,256] [3,40,40,256] [1,40,40,256]
        # matching_layer的取值['relu3_1', 'relu2_1', 'relu1_1']
        # map_ref[0] [1,40,40,256] ; map_ref[1] [1,80,80,128] ; map_ref[2] [1,160,160,64]
        map_in_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_in_sr, layer_name=matching_layer[0])
        map_ref = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref, layer_name=matching_layer)
        map_ref_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref_sr, layer_name=matching_layer[0])

        # patch matching and swapping
        # [2,80,80,128]
        other_style = []
        for m in map_ref[1:]:
            other_style.append([m])

        # print("shape of map_in_sr", len(map_in_sr))
        # print("shape of map_ref", len(map_ref))
        # print("shape of map_ref[0]", len(map_ref[0]))
        # print("shape of map_ref_sr", len(map_ref_sr))
        # print("shape of other_style", len(other_style))
        # print("shape of map_ref[0]", np.array(map_ref[0]).shape)
        # print("shape of map_ref[1]", np.array(map_ref[1]).shape)
        # print("shape of map_ref[2]", np.array(map_ref[2]).shape)
        maps, weights, correspondence = swaper.conditional_swap_multi_layer(
            content=map_in_sr,
            style=[map_ref[0]],
            condition=[map_ref_sr],
            other_styles=other_style
        )

        # save maps
        np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)
