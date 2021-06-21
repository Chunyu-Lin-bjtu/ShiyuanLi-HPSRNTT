import tensorflow as tf
import cv2
import math
from .tensorlayer import *
from .tensorlayer.layers import *
from os.path import join, exists, split, isfile
from os import makedirs, environ
from shutil import rmtree
from .vgg19 import *
from .swap import *
from glob import glob
from scipy.misc import imread, imresize, imsave, imrotate
from .download_vgg19_model import *
from bicubic_kernel import back_projection_loss
import logging
from scipy.io import savemat
from scipy.signal import convolve2d

# set logging level for TensorFlow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
    filename='SRNTT.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# some global variables
MODEL_FOLDER = 'model'
SAMPLE_FOLDER = 'sample'
SRNTT_MODEL_NAMES = {
    'init': 'srntt_init.npz',
    'conditional_texture_transfer': 'srntt.npz',
    'content_extractor': 'upscale.npz',
    'discriminator': 'discrim.npz',
    'weighted': 'srntt_weighted.npz'
}


class SRNTT(object):
    MAX_IMAGE_SIZE = 2046 ** 2

    def __init__(
            self,
            srntt_model_path='models/SRNTT',
            vgg19_model_path='models/VGG19/imagenet-vgg-verydeep-19.mat',
            save_dir=None,
            num_res_blocks=16,
    ):
        self.srntt_model_path = srntt_model_path
        self.vgg19_model_path = vgg19_model_path
        self.save_dir = save_dir
        self.num_res_blocks = int(num_res_blocks)
        self.is_model_built = False
        download_vgg19(self.vgg19_model_path)

    def model(
            self,
            inputs,  # LR images, in range of [-1, 1]
            maps=None,  # texture feature maps after texture swapping
            weights=None,  # weights of each pixel on the maps
            is_train=True,
            reuse=False,
            concat=False  # concatenate weights to feature
    ):

        # ********************************************************************************
        # *** content extractor
        # ********************************************************************************
        # print('\tcontent extractor')
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("content_extractor", reuse=reuse):
            layers.set_name_reuse(reuse)
            net = InputLayer(inputs=inputs, name='input')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='n64s1/c')
            temp = net
            for i in range(16):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='n64s1/b1/%s' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='n64s1/b2/%s' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='b_residual_add/%s' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
            content_feature = ElementwiseLayer(layer=[net, temp], combine_fn=tf.add, name='add3')

            # upscaling (4x) for texture extractor
            net = Conv2d(net=content_feature, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='n256s1/1')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='n256s1/2')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

            # output value range is [-1, 1]
            net_upscale = Conv2d(net=net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                                 padding='SAME', W_init=w_init, name='out')
            if maps is None:
                return net_upscale, None

        # ********************************************************************************
        # *** conditional texture transfer
        # ********************************************************************************
        with tf.variable_scope("texture_transfer", reuse=reuse):
            layers.set_name_reuse(reuse)
            assert isinstance(maps, (list, tuple))
            # fusion content and texture maps at the smallest scale
            # print('\tfusion content and texture maps at SMALL scale')
            map_in = InputLayer(inputs=content_feature.outputs, name='content_feature_maps')
            if weights is not None and concat:
                self.a1 = tf.get_variable(dtype=tf.float32, name='small/a', initializer=1.)
                self.b1 = tf.get_variable(dtype=tf.float32, name='small/b', initializer=0.)
                map_ref = maps[0] * tf.nn.sigmoid(self.a1 * weights + self.b1)
            else:
                map_ref = maps[0]
            map_ref = InputLayer(inputs=map_ref, name='reference_feature_maps1')
            net = ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation1')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='small/conv1')
            for i in range(self.num_res_blocks):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='small/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='small/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='small/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='small/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='small/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='small/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='small/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='small/add2')
            # upscaling (2x)
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='small/conv3')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='small/subpixel')

            # fusion content and texture maps at the medium scale
            # print('\tfusion content and texture maps at MEDIUM scale')
            map_in = net
            if weights is not None and concat:
                self.a2 = tf.get_variable(dtype=tf.float32, name='medium/a', initializer=1.)
                self.b2 = tf.get_variable(dtype=tf.float32, name='medium/b', initializer=0.)
                map_ref = maps[1] * tf.nn.sigmoid(self.a2 * tf.image.resize_bicubic(
                    weights, [weights.get_shape()[1] * 2, weights.get_shape()[2] * 2]) + self.b2)
            else:
                map_ref = maps[1]
            map_ref = InputLayer(inputs=map_ref, name='reference_feature_maps2')
            net = ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation2')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='medium/conv1')
            for i in range(int(self.num_res_blocks / 2)):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='medium/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='medium/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='medium/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='medium/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='medium/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='medium/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='medium/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='medium/add2')
            # upscaling (2x)
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='medium/conv3')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='medium/subpixel')

            # fusion content and texture maps at the large scale
            # print('\tfusion content and texture maps at LARGE scale')
            map_in = net
            if weights is not None and concat:
                self.a3 = tf.get_variable(dtype=tf.float32, name='large/a', initializer=1.)
                self.b3 = tf.get_variable(dtype=tf.float32, name='large/b', initializer=0.)
                map_ref = maps[2] * tf.nn.sigmoid(self.a3 * tf.image.resize_bicubic(
                    weights, [weights.get_shape()[1] * 4, weights.get_shape()[2] * 4]) + self.b3)
            else:
                map_ref = maps[2]
            map_ref = InputLayer(inputs=map_ref, name='reference_feature_maps3')
            net = ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation3')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='large/conv1')
            for i in range(int(self.num_res_blocks / 4)):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='large/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='large/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='large/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='large/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='large/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='large/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='large/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='large/add2')
            net = Conv2d(net=net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='large/conv3')
            # net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='large/bn2')

            # output of SRNTT, range [-1, 1]
            net_srntt = Conv2d(net=net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                               padding='SAME', W_init=w_init, name='out')

            return net_upscale, net_srntt

    # 把这里改成patch-gan的判别器
    def discriminator(self, input_image, is_train=True, reuse=False):
        # wgan的判别器最后一层不经过sigmoid!!!
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        g_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x: act.lrelu(x, 0)
        df_dim = 32
        with tf.variable_scope('discriminator', reuse=reuse):
            layers.set_name_reuse(reuse)
            net = InputLayer(inputs=input_image, name='input')
            # conv1:(N, 320, 320, 3) -> (N, 80, 80, 32)
            n_channels = df_dim
            net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(2, 2), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv1_conv1' )
            net = BatchNormLayer(layer=net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                 name='conv1_batch1')
            net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv1_conv2' )
            net = BatchNormLayer(layer=net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                 name='conv1_batch2')
            net = MaxPool2d(net, (2, 2), padding='SAME', name='conv1_pool1')

            # conv2: (N, 80, 80, 32) -> (N, 40, 40, 64)
            n_channels = df_dim * 2
            net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv2_conv1' )
            net = BatchNormLayer(layer=net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                 name='conv2_batch1')
            net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv2_conv2' )
            net = BatchNormLayer(layer=net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                 name='conv2_batch2')
            net = MaxPool2d(net, (2, 2), padding='SAME', name='conv2_pool1')

            # conv3: (N, 40, 40, 64) -> (N, 20, 20, 128)
            n_channels = df_dim * 4
            net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv3_conv1' )
            net = BatchNormLayer(layer=net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                 name='conv3_batch1')
            net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv3_conv2' )
            net = BatchNormLayer(layer=net, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                 name='conv3_batch2')
            net = MaxPool2d(net, (2, 2), padding='SAME', name='conv3_pool1')

            # output layer: (N, 40, 40, 128) -> (N, 40, 40, 1)
            net = Conv2d(net=net, n_filter=1, filter_size=(1, 1), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='conv_output')

            logits = net.outputs
            net.outputs = tf.nn.sigmoid(net.outputs)
            # print("判别器中logits的维度", logits.get_shape().as_list())

            return net, logits
    # 上面注释掉的这个是patch-gan
    # def discriminator(self, input_image, is_train=True, reuse=False):
    #     w_init = tf.random_normal_initializer(stddev=0.02)
    #     b_init = None
    #     g_init = tf.random_normal_initializer(1., 0.02)
    #     lrelu = lambda x: act.lrelu(x, 0.2)
    #     df_dim = 32
    #     with tf.variable_scope('discriminator', reuse=reuse):
    #         layers.set_name_reuse(reuse)
    #         net = InputLayer(inputs=input_image, name='input')
    #         for i in range(5):
    #             n_channels = df_dim * 2 ** i
    #             net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1), act=None,
    #                          padding='SAME', W_init=w_init, b_init=b_init, name='n%ds1/c' % n_channels)
    #             net = BatchNormLayer(layer=net, act=lrelu, is_train=is_train, gamma_init=g_init,
    #                                  name='n%ds1/b' % n_channels)
    #             net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(2, 2), act=None,
    #                          padding='SAME', W_init=w_init, b_init=b_init, name='n%ds2/c' % n_channels)
    #             net = BatchNormLayer(layer=net, act=lrelu, is_train=is_train, gamma_init=g_init,
    #                                  name='n%ds2/b' % n_channels)
    #         net = FlattenLayer(layer=net, name='flatten')
    #         net = DenseLayer(layer=net, n_units=1024, act=lrelu, name='fc2014')
    #         net = DenseLayer(net, n_units=1, name='output')
    #         logits = net.outputs
    #         net.outputs = tf.nn.sigmoid(net.outputs)
    #
    #         return net, logits

    def tf_gram_matrix(self, x):
        x = tf.reshape(x, tf.stack([-1, tf.reduce_prod(x.get_shape()[1:-1]), x.get_shape()[-1]]))
        return tf.matmul(x, x, transpose_a=True)

    def eta(self, time_per_iter, n_iter_remain, current_eta=None, alpha=.8):
        eta_ = time_per_iter * n_iter_remain
        if current_eta is not None:
            eta_ = (current_eta - time_per_iter) * alpha + eta_ * (1 - alpha)
        new_eta = eta_

        days = eta_ // (3600 * 24)
        eta_ -= days * (3600 * 24)

        hours = eta_ // 3600
        eta_ -= hours * 3600

        minutes = eta_ // 60
        eta_ -= minutes * 60

        seconds = eta_

        if days > 0:
            if days > 1:
                time_str = '%2d days %2d hr' % (days, hours)
            else:
                time_str = '%2d day %2d hr' % (days, hours)
        elif hours > 0 or minutes > 0:
            time_str = '%02d:%02d' % (hours, minutes)
        else:
            time_str = '%02d sec' % seconds

        return time_str, new_eta

    def train(
            self,
            input_dir='data/train/input',  # original images
            ref_dir='data/train/ref',  # reference images
            map_dir='data/train/map_321',  # texture maps after texture swapping
            batch_size=9,
            num_init_epochs=5,
            num_epochs=100,
            learning_rate=1e-4,
            beta1=0.9,
            use_pretrained_model=True,
            use_init_model_only=False,  # the init model is trained only with the reconstruction loss
            weights=(1e-4, 1e-4, 1e-6, 1., 1.),
            # (perceptual loss, texture loss, adversarial loss, back projection loss, reconstruction_loss)
            vgg_perceptual_loss_layer='relu5_1',  # the layer name to compute perceptrual loss
            is_WGAN_GP=True,
            is_L1_loss=True,
            param_WGAN_GP=10,
            input_size=40,
            use_weight_map=False,
            use_lower_layers_in_per_loss=False
    ):
        if np.sqrt(batch_size) != int(np.sqrt(batch_size)):
            logging.error('The batch size must be the power of an integer.')
            exit(0)

        # detect existing model if not use_pretrained_model
        if self.save_dir is None:
            self.save_dir = 'default_save_dir'
        if not use_pretrained_model and exists(join(self.save_dir, MODEL_FOLDER)):
            logging.warning('The existing model dir %s is removed!' % join(self.save_dir, MODEL_FOLDER))
            rmtree(join(self.save_dir, MODEL_FOLDER))

        # create save folders
        for folder in [MODEL_FOLDER, SAMPLE_FOLDER]:
            if not exists(join(self.save_dir, folder)):
                makedirs(join(self.save_dir, folder))

        # check input dir
        files_input = sorted(glob(join(input_dir, '*.jpg')))
        files_map = sorted(glob(join(map_dir, '*.npz')))
        files_ref = sorted(glob(join(ref_dir, '*.jpg')))
        num_files = len(files_input)
        print("len num_files", num_files)
        print("len files_ref", len(files_ref))
        print("len files_map", len(files_map))
        assert num_files == len(files_ref) == len(files_map)

        # ********************************************************************************
        # *** build graph
        # ********************************************************************************
        logging.info('Building graph ...')
        # input LR images, range [-1, 1]
        self.input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size, input_size, 3])

        # original images, range [-1, 1]
        self.ground_truth = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size * 4, input_size * 4, 3])

        # texture feature maps, range [0, ?]
        self.maps = tuple([tf.placeholder(dtype=tf.float32, shape=[batch_size, m.shape[0], m.shape[1], m.shape[2]])
                           for m in np.load(files_map[0])['target_map']])

        # weight maps
        self.weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size, input_size])

        # reference images, ranges[-1, 1]
        self.ref = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size, input_size, 3])

        # SRNTT network
        if use_weight_map:
            self.net_upscale, self.net_srntt = self.model(self.input, self.maps,
                                                          weights=tf.expand_dims(self.weights, axis=-1))
        else:
            self.net_upscale, self.net_srntt = self.model(self.input, self.maps)

        # VGG19 network, input range [0, 255]
        self.net_vgg_sr = VGG19((self.net_srntt.outputs + 1) * 127.5, model_path=self.vgg19_model_path)
        self.net_vgg_hr = VGG19((self.ground_truth + 1) * 127.5, model_path=self.vgg19_model_path)

        # discriminator network
        self.net_d, d_real_logits = self.discriminator(self.ground_truth)
        _, d_fake_logits = self.discriminator(self.net_srntt.outputs, reuse=True)

        # ********************************************************************************
        # *** objectives
        # ********************************************************************************
        # reconstruction loss
        if is_L1_loss:
            loss_reconst = tf.reduce_mean(tf.abs(self.net_srntt.outputs - self.ground_truth))
        else:
            loss_reconst = cost.mean_squared_error(self.net_srntt.outputs, self.ground_truth, is_mean=True)

        # perceptual loss
        loss_percep = cost.mean_squared_error(
            self.net_vgg_sr.layers[vgg_perceptual_loss_layer],
            self.net_vgg_hr.layers[vgg_perceptual_loss_layer],
            is_mean=True)
        try:
            available_layers = ['relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
            available_layers = available_layers[:available_layers.index(vgg_perceptual_loss_layer)]
            loss_percep_lower_layers = [cost.mean_squared_error(
                self.net_vgg_sr.layers[l],
                self.net_vgg_hr.layers[l],
                is_mean=True) for l in available_layers]
            if use_lower_layers_in_per_loss:
                loss_percep = tf.reduce_mean([loss_percep] + loss_percep_lower_layers)
        except Exception:
            logging.warning('Failed to use lower layers in perceptual loss!')

        # texture loss
        if use_weight_map:
            self.a1, self.a2, self.a3 = -20., -20, -20
            self.b1, self.b2, self.b3 = .65, .65, .65
            loss_texture = tf.reduce_mean(tf.squared_difference(
                self.tf_gram_matrix(
                    self.maps[0] * tf.nn.sigmoid(tf.expand_dims(self.weights, axis=-1) * self.a1 + self.b1)),
                self.tf_gram_matrix(self.net_vgg_sr.layers['relu3_1'] * tf.nn.sigmoid(
                    tf.expand_dims(self.weights, axis=-1) * self.a1 + self.b1))
            ) / 4. / (input_size * input_size * 256) ** 2) + tf.reduce_mean(tf.squared_difference(
                self.tf_gram_matrix(
                    self.maps[1] * tf.nn.sigmoid(tf.image.resize_bicubic(tf.expand_dims(self.weights, axis=-1),
                                                                         [input_size * 2] * 2) * self.a2 + self.b2)),
                self.tf_gram_matrix(
                    self.net_vgg_sr.layers['relu2_1'] * tf.nn.sigmoid(
                        tf.image.resize_bicubic(tf.expand_dims(self.weights, axis=-1),
                                                [input_size * 2] * 2) * self.a2 + self.b2))
            ) / 4. / (input_size * input_size * 512) ** 2) + tf.reduce_mean(tf.squared_difference(
                self.tf_gram_matrix(
                    self.maps[2] * tf.nn.sigmoid(tf.image.resize_bicubic(tf.expand_dims(self.weights, axis=-1),
                                                                         [input_size * 4] * 2) * self.a3 + self.b3)),
                self.tf_gram_matrix(self.net_vgg_sr.layers['relu1_1'] * tf.nn.sigmoid(
                    tf.image.resize_bicubic(tf.expand_dims(self.weights, axis=-1),
                                            [input_size * 4] * 2) * self.a3 + self.b3))
            ) / 4. / (input_size * input_size * 1024) ** 2)
            loss_texture /= 3.
        else:
            # 计算gram矩阵时，不计算maps[1]的
            # loss_texture = tf.reduce_mean(tf.squared_difference(
            #     self.tf_gram_matrix(self.maps[0]),
            #     self.tf_gram_matrix(self.net_vgg_sr.layers['relu3_1'])
            # ) / 4. / (input_size * input_size * 256) ** 2) + tf.reduce_mean(tf.squared_difference(
            #     self.tf_gram_matrix(self.maps[1]),
            #     self.tf_gram_matrix(self.net_vgg_sr.layers['relu2_1'])
            # ) / 4. / (input_size * input_size * 512) ** 2) + tf.reduce_mean(tf.squared_difference(
            #     self.tf_gram_matrix(self.maps[2]),
            #     self.tf_gram_matrix(self.net_vgg_sr.layers['relu1_1'])
            # ) / 4. / (input_size * input_size * 1024) ** 2)
            # loss_texture /= 3.
            # print("calculate gram_321")

            # 下面注释掉的是没有计算map1的gram矩阵
            # loss_texture = tf.reduce_mean(tf.squared_difference(
            #     self.tf_gram_matrix(self.maps[0]),
            #     self.tf_gram_matrix(self.net_vgg_sr.layers['relu3_1'])
            # ) / 4. / (input_size * input_size * 256) ** 2) + tf.reduce_mean(tf.squared_difference(
            #     self.tf_gram_matrix(self.maps[2]),
            #     self.tf_gram_matrix(self.net_vgg_sr.layers['relu1_1'])
            # ) / 4. / (input_size * input_size * 1024) ** 2)
            # loss_texture /= 3.
            # print("no map1 in gram")
            # 下面是没有计算map2的
            loss_texture = tf.reduce_mean(tf.squared_difference(
                self.tf_gram_matrix(self.maps[0]),
                self.tf_gram_matrix(self.net_vgg_sr.layers['relu3_1'])
            ) / 4. / (input_size * input_size * 256) ** 2) + tf.reduce_mean(tf.squared_difference(
                self.tf_gram_matrix(self.maps[1]),
                self.tf_gram_matrix(self.net_vgg_sr.layers['relu2_1'])
            ) / 4. / (input_size * input_size * 512) ** 2)
            loss_texture /= 2.
            print("no map2 in gram")

        # adversarial loss
        if is_WGAN_GP:
            # WGAN losses
            loss_d = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits)
            loss_g = -tf.reduce_mean(d_fake_logits)
            # GP: gradient penalty
            alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolates = alpha * self.ground_truth + ((1 - alpha) * self.net_srntt.outputs)
            _, disc_interpolates = self.discriminator(interpolates, reuse=True)
            gradients = tf.gradients(disc_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=-1))
            gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
            loss_d += param_WGAN_GP * gradient_penalty
        else:
            loss_g = cost.sigmoid_cross_entropy(d_fake_logits, tf.ones_like(d_fake_logits))
            loss_d_fake = cost.sigmoid_cross_entropy(d_fake_logits, tf.zeros_like(d_fake_logits))
            loss_d_real = cost.sigmoid_cross_entropy(d_real_logits, tf.ones_like(d_real_logits))
            loss_d = loss_d_fake + loss_d_real

        # back projection loss
        loss_bp = back_projection_loss(tf_input=self.input, tf_output=self.net_srntt.outputs)

        # total loss
        loss_init = weights[4] * loss_reconst + weights[3] * loss_bp
        loss = weights[4] * loss_reconst + weights[3] * loss_bp + \
               weights[2] * loss_g + \
               weights[1] * loss_texture + \
               weights[0] * loss_percep

        # ********************************************************************************
        # *** optimizers
        # ********************************************************************************
        # trainable variables
        trainable_vars = tf.trainable_variables()
        var_g = [v for v in trainable_vars if 'texture_transfer' in v.name]
        var_d = [v for v in trainable_vars if 'discriminator' in v.name]

        # learning rate decay
        global_step = tf.Variable(0, trainable=False, name='global_step')
        num_batches = int(num_files / batch_size)
        decayed_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=max(num_epochs * num_batches / 2, 1),
            decay_rate=.1,
            staircase=True
        )

        # optimizer
        optimizer_init = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1).minimize(loss_init, var_list=var_g)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=decayed_learning_rate, beta1=beta1).minimize(loss, var_list=var_g, global_step=global_step)
        optimizer_d = tf.train.AdamOptimizer(
            learning_rate=decayed_learning_rate, beta1=beta1).minimize(loss_d, var_list=var_d, global_step=global_step)

        # ********************************************************************************
        # *** samples for monitoring the training process
        # ********************************************************************************
        np.random.seed(2019)
        idx = np.random.choice(np.arange(num_files), batch_size, replace=False)
        samples_in = [imread(files_input[i], mode='RGB') for i in idx]
        samples_ref = [imresize(imread(files_ref[i], mode='RGB'), (input_size * 4, input_size * 4), interp='bicubic')
                       for i in idx]
        samples_input = [imresize(img, (input_size, input_size), interp='bicubic').astype(np.float32) / 127.5 - 1
                         for img in samples_in]
        samples_texture_map_tmp = [np.load(files_map[i])['target_map'] for i in idx]
        samples_texture_map = [[] for _ in range(len(samples_texture_map_tmp[0]))]
        for s in samples_texture_map_tmp:
            for i, item in enumerate(samples_texture_map):
                item.append(s[i])
        samples_texture_map = [np.array(b) for b in samples_texture_map]
        if use_weight_map:
            samples_weight_map = [np.pad(np.load(files_map[i])['weights'], ((1, 1), (1, 1)), 'edge') for i in idx]
        else:
            samples_weight_map = np.zeros(shape=(batch_size, input_size, input_size))
        frame_size = int(np.sqrt(batch_size))
        vis.save_images(np.array(samples_in), [frame_size, frame_size], join(self.save_dir, SAMPLE_FOLDER, 'HR.png'))
        vis.save_images(np.round((np.array(samples_input) + 1) * 127.5).astype(np.uint8), [frame_size, frame_size],
                        join(self.save_dir, SAMPLE_FOLDER, 'LR.png'))
        vis.save_images(np.array(samples_ref), [frame_size, frame_size], join(self.save_dir, SAMPLE_FOLDER, 'Ref.png'))

        # ********************************************************************************
        # *** load models and training
        # ********************************************************************************
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            logging.info('Loading models ...')
            tf.global_variables_initializer().run()

            # load pre-trained upscaling.
            model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['content_extractor'])
            if files.load_and_assign_npz(
                    sess=sess,
                    name=model_path,
                    network=self.net_upscale) is False:
                logging.error('FAILED load %s' % model_path)
                exit(0)
            vis.save_images(
                np.round((self.net_upscale.outputs.eval({self.input: samples_input}) + 1) * 127.5).astype(np.uint8),
                [frame_size, frame_size], join(self.save_dir, SAMPLE_FOLDER, 'Upscale.png'))

            # load the specific texture transfer model, specified by save_dir
            is_load_success = False
            if use_init_model_only:
                model_path = join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['init'])
                if files.load_and_assign_npz(
                        sess=sess,
                        name=model_path,
                        network=self.net_srntt):
                    num_init_epochs = 0
                    is_load_success = True
                    logging.info('SUCCESS load %s' % model_path)
                else:
                    logging.warning('FAILED load %s' % model_path)
            elif use_pretrained_model:
                model_path = join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['conditional_texture_transfer'])
                if files.load_and_assign_npz(
                        sess=sess,
                        name=model_path,
                        network=self.net_srntt):
                    num_init_epochs = 0
                    is_load_success = True
                    logging.info('SUCCESS load %s' % model_path)
                else:
                    logging.warning('FAILED load %s' % model_path)

                model_path = join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['discriminator'])
                if files.load_and_assign_npz(
                        sess=sess,
                        name=model_path,
                        network=self.net_d):
                    logging.info('SUCCESS load %s' % model_path)
                else:
                    logging.warning('FAILED load %s' % model_path)

            # load pre-trained conditional texture transfer
            if not is_load_success:
                use_weight_map = False
                if use_init_model_only:
                    model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['init'])
                    if files.load_and_assign_npz(
                            sess=sess,
                            name=model_path,
                            network=self.net_srntt):
                        num_init_epochs = 0
                        logging.info('SUCCESS load %s' % model_path)
                    else:
                        logging.error('FAILED load %s' % model_path)
                        exit(0)
                elif use_pretrained_model:
                    model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['conditional_texture_transfer'])
                    if files.load_and_assign_npz(
                            sess=sess,
                            name=model_path,
                            network=self.net_srntt):
                        num_init_epochs = 0
                        logging.info('SUCCESS load %s' % model_path)
                    else:
                        logging.error('FAILED load %s' % model_path)
                        exit(0)

            logging.info('**********'
                         ' Start training '
                         '**********')
            # pre-train with only reconstruction loss
            current_eta = None
            idx = np.arange(num_files)
            for epoch in xrange(num_init_epochs):
                np.random.shuffle(idx)
                for n_batch in xrange(num_batches):
                    step_time = time.time()
                    sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
                    batch_imgs = [imread(files_input[i], mode='RGB') for i in sub_idx]
                    batch_truth = [img.astype(np.float32) / 127.5 - 1 for img in batch_imgs]
                    batch_input = [imresize(img, .25, interp='bicubic').astype(np.float32) / 127.5 - 1 for img in
                                   batch_imgs]
                    batch_maps_tmp = [np.load(files_map[i])['target_map'] for i in sub_idx]
                    batch_maps = [[] for _ in range(len(batch_maps_tmp[0]))]
                    for s in batch_maps_tmp:
                        for i, item in enumerate(batch_maps):
                            item.append(s[i])
                    batch_maps = [np.array(b) for b in batch_maps]

                    if use_weight_map:
                        batch_weights = [np.pad(np.load(files_map[i])['weights'], ((1, 1), (1, 1)), 'edge')
                                         for i in sub_idx]

                    else:
                        batch_weights = np.zeros(shape=(batch_size, input_size, input_size))
                    # train with reference
                    _, l_reconst, l_bp, map_hr_3, map_hr_2, map_hr_1 = sess.run(
                        fetches=[optimizer_init, loss_reconst, loss_bp,
                                 self.net_vgg_hr.layers['relu3_1'],
                                 self.net_vgg_hr.layers['relu2_1'],
                                 self.net_vgg_hr.layers['relu1_1']],
                        feed_dict={
                            self.input: batch_input,
                            self.maps: batch_maps,
                            self.ground_truth: batch_truth,
                            self.weights: batch_weights
                        }
                    )

                    # train with truth
                    _, l_reconst, l_bp = sess.run(
                        fetches=[optimizer_init, loss_reconst, loss_bp],
                        feed_dict={
                            self.input: batch_input,
                            self.maps: [map_hr_3, map_hr_2, map_hr_1],
                            self.ground_truth: batch_truth,
                            self.weights: np.ones_like(np.array(batch_weights))
                        }
                    )

                    # print
                    time_per_iter = time.time() - step_time
                    n_iter_remain = (num_init_epochs - epoch - 1) * num_batches + num_batches - n_batch
                    eta_str, eta_ = self.eta(time_per_iter, n_iter_remain, current_eta)
                    current_eta = eta_
                    logging.info('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\n'
                                 '\tl_rec = %.4f \t l_bp = %.4f' %
                                 (epoch + 1, num_init_epochs, n_batch + 1, num_batches, eta_str,
                                  weights[4] * l_reconst, weights[3] * l_bp))

                # save intermediate results
                vis.save_images(
                    np.round((self.net_srntt.outputs.eval({
                        self.input: samples_input, self.maps: samples_texture_map,
                        self.weights: samples_weight_map}) + 1) * 127.5).astype(np.uint8),
                    [frame_size, frame_size],
                    join(self.save_dir, SAMPLE_FOLDER, 'init_E%03d.png' % (epoch + 1)))

                # save model for each epoch
                files.save_npz(
                    save_list=self.net_srntt.all_params,
                    name=join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['init']),
                    sess=sess)

            # train with all losses
            current_eta = None
            for epoch in xrange(num_epochs):
                np.random.shuffle(idx)
                for n_batch in xrange(num_batches):
                    step_time = time.time()
                    sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
                    batch_imgs = [imread(files_input[i], mode='RGB') for i in sub_idx]
                    batch_truth = [img.astype(np.float32) / 127.5 - 1 for img in batch_imgs]
                    batch_input = [imresize(img, .25, interp='bicubic').astype(np.float32) / 127.5 - 1 for img in
                                   batch_imgs]
                    batch_maps_tmp = [np.load(files_map[i])['target_map'] for i in sub_idx]
                    batch_maps = [[] for _ in range(len(batch_maps_tmp[0]))]
                    for s in batch_maps_tmp:
                        for i, item in enumerate(batch_maps):
                            item.append(s[i])
                    batch_maps = [np.array(b) for b in batch_maps]
                    if use_weight_map:
                        batch_weights = [np.pad(np.load(files_map[i])['weights'], ((1, 1), (1, 1)), 'edge')
                                         for i in sub_idx]
                    else:
                        batch_weights = np.zeros(shape=(batch_size, input_size, input_size))

                    # train with reference
                    for _ in xrange(2):
                        _ = sess.run(
                            fetches=[optimizer_d],
                            feed_dict={
                                self.input: batch_input,
                                self.maps: batch_maps,
                                self.ground_truth: batch_truth,
                                self.weights: batch_weights
                            }
                        )
                    _, _, l_rec, l_per, l_tex, l_adv, l_dis, l_bp, map_hr_3, map_hr_2, map_hr_1 = sess.run(
                        fetches=[optimizer, optimizer_d, loss_reconst, loss_percep, loss_texture, loss_g, loss_d,
                                 loss_bp,
                                 self.net_vgg_hr.layers['relu3_1'],
                                 self.net_vgg_hr.layers['relu2_1'],
                                 self.net_vgg_hr.layers['relu1_1'],
                                 ],
                        feed_dict={
                            self.input: batch_input,
                            self.maps: batch_maps,
                            self.ground_truth: batch_truth,
                            self.weights: batch_weights
                        }
                    )

                    # train with truth
                    _, _, l_rec, l_per, l_tex, l_adv, l_dis, l_bp = sess.run(
                        fetches=[optimizer, optimizer_d, loss_reconst, loss_percep, loss_texture, loss_g, loss_d,
                                 loss_bp],
                        feed_dict={
                            self.input: batch_input,
                            self.maps: [map_hr_3, map_hr_2, map_hr_1],
                            self.ground_truth: batch_truth,
                            self.weights: np.ones_like(np.array(batch_weights))
                        }
                    )

                    # print
                    time_per_iter = time.time() - step_time
                    n_iter_remain = (num_epochs - epoch - 1) * num_batches + num_batches - n_batch
                    eta_str, eta_ = self.eta(time_per_iter, n_iter_remain, current_eta)
                    current_eta = eta_
                    logging.info('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\n'
                                 '\tl_rec = %.4f\tl_bp  = %.4f\n'
                                 '\tl_per = %.4f\tl_tex = %.4f\n'
                                 '\tl_adv = %.4f\tl_dis = %.4f' %
                                 (epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str,
                                  weights[4] * l_rec, weights[3] * l_bp,
                                  weights[0] * l_per, weights[1] * l_tex,
                                  weights[2] * l_adv, l_dis))
                    logging.info('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\n'
                                 '\ttotal_loss = %.4f' %
                                 (epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str,
                                  weights[4] * l_rec + weights[3] * l_bp + weights[0] * l_per + weights[1] * l_tex +
                                  weights[2] * l_adv))

                # save intermediate results
                vis.save_images(
                    np.round((self.net_srntt.outputs.eval({
                        self.input: samples_input, self.maps: samples_texture_map,
                        self.weights: samples_weight_map}) + 1) * 127.5).astype(np.uint8),
                    [frame_size, frame_size],
                    join(self.save_dir, SAMPLE_FOLDER, 'E%03d.png' % (epoch + 1)))

                # save models for each epoch
                files.save_npz(
                    save_list=self.net_srntt.all_params,
                    name=join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['conditional_texture_transfer']),
                    sess=sess)
                files.save_npz(
                    save_list=self.net_d.all_params,
                    name=join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['discriminator']),
                    sess=sess)

    def cal_psnr(self, img1, img2):
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

    def cal_ssim(self, img1, img2, ssim):
        height = img1.shape[0]
        width = img1.shape[1]
        total_wssim = 0
        total_wij = 0
        for i in range(height):
            wij = math.cos((i + 0.5 - height / 2) * math.pi / height)
            total_wssim += ssim*wij
            total_wij += wij
        print("total_wij",total_wij)
        print("total_wssim",total_wssim)
        ws_ssim = total_wssim / total_wij
        return ws_ssim



    def combine_overlap(self, top_left, top_right, center_left, ref, bot_left, bot_right, processWidth=0):
        img_left = center_left
        img_right = ref
        img_center_w = img_left.shape[1]
        img_center_h = img_left.shape[0]

        img_top_left = top_left
        img_top_right = top_right
        img_top_w = img_top_left.shape[1]
        img_top_h = img_top_left.shape[0]

        img_bot_left = bot_left
        img_bot_right = bot_right
        img_bot_w = img_bot_left.shape[1]
        img_bot_h = img_bot_left.shape[0]

        # print("img_center_w",img_center_w)
        # print("img_center_h", img_center_h)
        # print("img_top_w", img_top_w)
        # print("img_top_h", img_top_h)
        # print("img_bot_w", img_bot_w)
        # print("img_bot_h", img_bot_h)

        processWidth = processWidth
        new_img_center = np.zeros((img_center_h, processWidth, 3))
        new_img_top = np.zeros((img_top_h, processWidth, 3))
        new_img_bot = np.zeros((img_bot_h, processWidth, 3))
        # new_img_top_ver = np.zeros((processWidth, 1920, 3))
        # new_img_bot_ver = np.zeros((processWidth, 1920, 3))
        new_img_top_ver = np.zeros((processWidth, int(2*(img_center_w-processWidth/2)), 3))
        new_img_bot_ver = np.zeros((processWidth, int(2*(img_center_w-processWidth/2)), 3))


        # left = int(960 - processWidth / 2)
        # right = int(960 + processWidth / 2)
        # top1 = int(240 - processWidth / 2)
        # bot1 = int(240 + processWidth / 2)
        # top2 = int(720 - processWidth / 2)
        # bot2 = int(720 + processWidth / 2)


        # left = int(480 - processWidth / 2)
        # right = int(480 + processWidth / 2)
        # top1 = int(120 - processWidth / 2)
        # bot1 = int(120 + processWidth / 2)
        # top2 = int(360 - processWidth / 2)
        # bot2 = int(360 + processWidth / 2)
        #
        # print("left1",left)
        # print("right1",right)
        # print("top11",top1)
        # print("bot11", bot1)
        # print("top21", top2)
        # print("top21", bot2)

        left = int(img_center_w - processWidth)
        right = int(img_center_w)
        top1 = int(img_top_h - processWidth)
        bot1 = int(img_top_h)
        top2 = int(3*int(img_top_h - processWidth/2) - processWidth / 2)
        bot2 = int(3*int(img_top_h - processWidth/2) + processWidth / 2)
        # print("left", left)
        # print("right", right)
        # print("top1", top1)
        # print("bot1", bot1)
        # print("top2", top2)
        # print("top2", bot2)
        for x in range(left, right):
            alpha = (processWidth - (x - left)) / processWidth
            new_img_center[:, x - left, :] = alpha * img_left[:, x, :] + (1 - alpha) * img_right[:, x - left, :]
            new_img_top[:, x - left, :] = alpha * img_top_left[:, x, :] + (1 - alpha) * img_top_right[:, x - left, :]
            new_img_bot[:, x - left, :] = alpha * img_bot_left[:, x, :] + (1 - alpha) * img_bot_right[:, x - left, :]
        # 连接的时候，取左边968-16，融合后的16，右边的968-16到968.即左右各取952，再加上中间融合的16
        img_center_overlap = np.concatenate((img_left[:, 0:img_center_w - processWidth, :], new_img_center[:, :, :],
                                             img_right[:, processWidth:img_center_w, :]), axis=1)
        img_top_overlap = np.concatenate((img_top_left[:, 0:img_top_w - processWidth, :], new_img_top[:, :, :],
                                          img_top_right[:, processWidth:img_top_w, :]), axis=1)
        img_bot_overlap = np.concatenate((img_bot_left[:, 0:img_bot_w - processWidth, :], new_img_bot[:, :, :],
                                          img_bot_right[:, processWidth:img_bot_w, :]), axis=1)


        # print("img_top_overlap[0:img_top_h - processWidth, :, :]", img_top_overlap.shape)
        # print("new_img_top_ver[:, :, :]",new_img_top_ver.shape)
        # print("img_center_overlap[processWidth:img_center_h, :, :]",img_center_overlap.shape)


        for y in range(top1, bot1):
            alpha = (processWidth - (y - top1)) / processWidth
            # print(alpha)
            new_img_top_ver[y - top1, :, :] = alpha * img_top_overlap[y, :, :] + (1 - alpha) * img_center_overlap[
                                                                                               y - top1, :, :]

        img_top_center_overlap = np.concatenate((img_top_overlap[0:img_top_h - processWidth, :, :],
                                                 new_img_top_ver[:, :, :],
                                                 img_center_overlap[processWidth:img_center_h, :, :]), axis=0)
        for y in range(top2, bot2):
            alpha = (processWidth - (y - top2)) / processWidth
            # print(alpha)
            new_img_bot_ver[y - top2, :, :] = alpha * img_top_center_overlap[y, :, :] + (1 - alpha) * img_bot_overlap[
                                                                                                      y - top2, :, :]
        img_top_center_overlap_h = img_top_center_overlap.shape[0]
        img_top_center_bot_overlap = np.concatenate((
            img_top_center_overlap[0:img_top_center_overlap_h - processWidth, :,
            :], new_img_bot_ver[:, :, :],
            img_bot_overlap[processWidth:img_bot_h, :, :]), axis=0)
        img_top_center_bot_overlap = np.array(img_top_center_bot_overlap).squeeze().round().clip(0, 255).astype(np.uint8)
        return img_top_center_bot_overlap

    def test(
            self,
            input_dir,  # original image
            ref_dir=None,  # reference images
            overlap=0,
            hr_full=None,
            use_pretrained_model=True,
            use_init_model_only=False,  # the init model is trained only with the reconstruction loss
            use_weight_map=False,
            result_dir=None,
            ref_scale=1.0,
            is_original_image=True,
            max_batch_size=16,
            save_ref=True
    ):
        logging.info('Testing mode')

        if ref_dir is None:
            return self.test_without_ref(
                input_dir=input_dir,
                use_pretrained_model=use_pretrained_model,
                use_init_model_only=use_init_model_only,
                use_weight_map=use_weight_map,
                result_dir=result_dir,
                ref_scale=ref_scale,
                is_original_image=is_original_image,
                max_batch_size=max_batch_size,
                save_ref=save_ref
            )

        # ********************************************************************************
        # *** check input and reference images
        # ********************************************************************************
        # check input_dir
        img_inputtt, img_hr = None, None
        if isinstance(input_dir, np.ndarray):
            assert len(input_dir.shape) == 3
            img_inputtt = np.copy(input_dir)
        elif isfile(input_dir):
            img_inputtt = imread(input_dir, mode='RGB')
        else:
            logging.error('Unrecognized input_dir %s' % input_dir)
            exit(0)

        input_block = []
        all_upscale = []
        all_SRNTT = []
        all_bicubic = []
        all_hr = []
        lr_all_w = img_inputtt.shape[1]
        lr_all_h = img_inputtt.shape[0]
        # input_block.append(img_inputtt[:, 0:960 + overlap, :])
        # input_block.append(img_inputtt[:, 960 + overlap:1920 + 2 * overlap, :])
        # input_block.append(img_inputtt[:, 1920 + 2 * overlap:2880 + 3 * overlap, :])
        input_block.append(img_inputtt[:, 0:int(lr_all_w/3), :])
        input_block.append(img_inputtt[:, int(lr_all_w/3):int(lr_all_w*2/3), :])
        input_block.append(img_inputtt[:, int(lr_all_w*2/3):lr_all_w, :])
        iii = 0
        for img_input in input_block:
            iii = iii + 1
            # imsave(join(result_dir, 'testtttttttttttt.png'), img_input)
            h, w, _ = img_input.shape
            if is_original_image:
                # ensure that the size of img_input can be divided by 4 with no remainder
                h = int(h // 4 * 4)
                w = int(w // 4 * 4)
                img_hr = img_input[0:h, 0:w, ::]
                # 把4倍改成2倍
                img_input = imresize(img_hr, .25, interp='bicubic')
                # img_input = imresize(img_hr, .5, interp='bicubic')
                h, w, _ = img_input.shape
            img_input_copy = np.copy(img_input)
            if h * w * 16 > SRNTT.MAX_IMAGE_SIZE:  # avoid OOM
                # split img_input into patches
                patches = []
                grids = []
                patch_size = 128
                stride = 100
                for ind_row in range(0, h - (patch_size - stride), stride):
                    for ind_col in range(0, w - (patch_size - stride), stride):
                        patch = img_input[ind_row:ind_row + patch_size, ind_col:ind_col + patch_size, :]
                        if patch.shape != (patch_size, patch_size, 3):
                            patch = np.pad(patch,
                                           ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)),
                                           'reflect')
                        patches.append(patch)
                        grids.append((ind_row * 4, ind_col * 4, patch_size * 4))
                grids = np.stack(grids, axis=0)
                logging.info('grids dementiom', grids.shape)
                # yyyyyyyyyyy
                img_input = np.stack(patches, axis=0)
            else:
                grids = None
                img_input = np.expand_dims(img_input, axis=0)

            # check ref_dir
            img_ref = []
            if not isinstance(ref_dir, (list, tuple)):
                ref_dir = [ref_dir]
            # print(ref_dir)

            for ref in ref_dir:
                if isinstance(ref, np.ndarray):
                    assert len(ref.shape) == 3
                    img_ref.append(np.copy(ref))
                elif isfile(ref):
                    img_ref.append(imread(ref, mode='RGB'))
                else:
                    logging.error('Unrecognized ref_dir type!')
                    exit(0)

            if ref_scale <= 0:  # keep the same scale as HR image
                img_ref = [imresize(img, (h * 4, w * 4), interp='bicubic') for img in img_ref]
            elif ref_scale != 1:
                img_ref = [imresize(img, float(ref_scale), interp='bicubic') for img in img_ref]

            for i in xrange(len(img_ref)):
                h2, w2, _ = img_ref[i].shape
                h2 = int(h2 // 4 * 4)
                w2 = int(w2 // 4 * 4)
                img_ref[i] = img_ref[i][0:h2, 0:w2, ::]

            # create result folder
            if result_dir is None:
                result_dir = join(self.save_dir, 'test')
            if not exists(result_dir):
                makedirs(result_dir)
            if not exists(join(result_dir, 'tmp')):
                makedirs(join(result_dir, 'tmp'))

            # ********************************************************************************
            # *** build graph
            # ********************************************************************************
            if not self.is_model_built:
                self.is_model_built = True
                logging.info('Building graph ...')
                # input image, range [-1, 1]
                self.input_srntt = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)

                # reference images, range [0, 255]
                self.input_vgg19 = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)

                # swapped feature map and weights
                self.maps = (
                    tf.placeholder(
                        dtype=tf.float32,
                        shape=(1, None, None, 256)),
                    tf.placeholder(
                        dtype=tf.float32,
                        shape=(1, None, None, 128)),
                    tf.placeholder(
                        dtype=tf.float32,
                        shape=(1, None, None, 64))
                )

                self.weights = tf.placeholder(
                    dtype=tf.float32,
                    shape=(1, None, None))

                # SRNTT network
                logging.info('Build SRNTT model')
                if use_weight_map:
                    self.net_upscale, self.net_srntt = self.model(
                        self.input_srntt, self.maps, weights=tf.expand_dims(self.weights, axis=-1), is_train=False)
                else:
                    self.net_upscale, self.net_srntt = self.model(self.input_srntt, self.maps, is_train=False)

                # VGG19 network, input range [0, 255]
                logging.info('Build VGG19 model')
                self.net_vgg19 = VGG19(
                    input_image=self.input_vgg19,
                    model_path=self.vgg19_model_path,
                    final_layer='relu3_1'
                )

                # ********************************************************************************
                # *** load models
                # ********************************************************************************
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = False
                self.sess = tf.Session(config=config)

                # instant of Swap()
                logging.info('Initialize the swapper')
                self.swaper = Swap(sess=self.sess)

                logging.info('Loading models ...')
                self.sess.run(tf.global_variables_initializer())

                # load pre-trained content extractor, including upscaling.
                model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['content_extractor'])
                if files.load_and_assign_npz(
                        sess=self.sess,
                        name=model_path,
                        network=self.net_upscale) is False:
                    logging.error('FAILED load %s' % model_path)
                    exit(0)

                # load the specific conditional texture transfer model, specified by save_dir
                if self.save_dir is None:
                    if use_init_model_only:
                        model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['init'])
                        if files.load_and_assign_npz(
                                sess=self.sess,
                                name=model_path,
                                network=self.net_srntt):
                            logging.info('SUCCESS load %s' % model_path)
                        else:
                            logging.error('FAILED load %s' % model_path)
                            exit(0)
                    else:
                        model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['conditional_texture_transfer'])
                        if files.load_and_assign_npz(
                                sess=self.sess,
                                name=model_path,
                                network=self.net_srntt):
                            logging.info('SUCCESS load %s' % model_path)
                        else:
                            logging.error('FAILED load %s' % model_path)
                            exit(0)
                else:
                    if use_init_model_only:
                        model_path = join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['init'])
                        if files.load_and_assign_npz(
                                sess=self.sess,
                                name=model_path,
                                network=self.net_srntt):
                            logging.info('SUCCESS load %s' % model_path)
                        else:
                            logging.error('FAILED load %s' % model_path)
                            exit(0)
                    else:
                        model_path = join(self.save_dir, MODEL_FOLDER,
                                          SRNTT_MODEL_NAMES['conditional_texture_transfer'])
                        if files.load_and_assign_npz(
                                sess=self.sess,
                                name=model_path,
                                network=self.net_srntt):
                            logging.info('SUCCESS load %s' % model_path)
                        else:
                            logging.error('FAILED load %s' % model_path)
                            exit(0)

            logging.info('**********'
                         ' Start testing '
                         '**********')

            matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']

            logging.info('Get VGG19 Feature Maps')

            logging.info('\t[1/2] Getting feature map of Ref image ...')
            t_start = time.time()
            map_ref = []
            for i in img_ref:
                map_ref.append(
                    self.net_vgg19.get_layer_output(
                        sess=self.sess, layer_name=matching_layer,
                        feed_image=i)
                )
            styles = [[] for _ in xrange(len(matching_layer))]
            for i in map_ref:
                for j in xrange(len(styles)):
                    styles[j].append(i[j])

            logging.info('\t[2/2] Getting feature map of LR->SR Ref image ...')
            map_ref_sr = []
            for i in img_ref:
                # 下采样为原来的一半，然后再上采样两倍---begin
                img_ref_downscale = imresize(i, .25, interp='bicubic')
                img_ref_upscale = imresize(img_ref_downscale, 4., interp='bicubic')
                # 下采样为原来的一半，然后再上采样两倍---end
                # img_ref_downscale = imresize(i, .5, interp='bicubic')
                # img_ref_upscale = imresize(img_ref_downscale, 2., interp='bicubic')
                map_ref_sr.append(
                    self.net_vgg19.get_layer_output(
                        sess=self.sess, layer_name=matching_layer[0],
                        feed_image=img_ref_upscale)
                )

            # swap ref to in
            logging.info('Patch-Wise Matching and Swapping')
            for idx, patch in enumerate(img_input):
                # logging.info('iii变了吗 %03d' % (iii))
                logging.info('\tPatch %03d/%03d' % (idx + 1, img_input.shape[0]))
                ##########第一次sr后，tmp中就有文件了。所以这句话会导致之后的sr被跳过
                # skip if the results exists
                # if exists(join(result_dir, 'tmp', 'srntt_%05d.png' % idx)):
                #     continue

                ##########让所有块都用SRNTT，加上if语句块，就是上下两极用upscale
                # if(iii>1):
                #     # 如果是输入的2 3块，就只需要得到他们的upscale
                #     out_srntt, out_upscale = self.sess.run(
                #         fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
                #         feed_dict={
                #             self.input_srntt: [patch / 127.5 - 1],
                #             self.maps: [np.expand_dims(m, axis=0) for m in map_target]
                #         }
                #     )
                #     imsave(join(result_dir, 'tmp', 'upscale_%05d.png' % idx),
                #            np.round((out_upscale.squeeze() + 1) * 127.5).astype(np.uint8))
                #     continue

                logging.info('\tGetting feature map of input LR image ...')
                # 将输入上采样为原来的两倍---begin
                img_input_upscale = imresize(patch, 4., interp='bicubic')
                # img_input_upscale = imresize(patch, 2., interp='bicubic')
                # 将输入上采样为原来的两倍---end
                map_sr = self.net_vgg19.get_layer_output(
                    sess=self.sess, layer_name=matching_layer[0], feed_image=img_input_upscale)

                logging.info('\tMatching and swapping features ...')
                map_target, weight, _ = self.swaper.conditional_swap_multi_layer(
                    content=map_sr,
                    style=styles[0],
                    condition=map_ref_sr,
                    other_styles=styles[1:],
                    is_weight=use_weight_map
                )

                # 这样是不行的，会报维度错误，只能在swap函数的最后强制把map设为全0的矩阵
                # map_target, weight, _ = self.swaper.conditional_swap_multi_layer(
                #     content=map_sr,
                #     style=styles[0],
                #     condition=map_ref_sr,
                #     other_styles=None,
                #     is_weight=use_weight_map
                # )

                logging.info('Obtain SR patches')
                if use_weight_map:
                    weight = np.pad(weight, ((1, 1), (1, 1)), 'edge')
                    out_srntt, out_upscale = self.sess.run(
                        fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
                        feed_dict={
                            self.input_srntt: [patch / 127.5 - 1],
                            self.maps: [np.expand_dims(m, axis=0) for m in map_target],
                            self.weights: [weight]
                        }
                    )
                else:
                    time_step_1 = time.time()
                    out_srntt, out_upscale = self.sess.run(
                        fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
                        feed_dict={
                            self.input_srntt: [patch / 127.5 - 1],
                            self.maps: [np.expand_dims(m, axis=0) for m in map_target],
                        }
                    )
                    time_step_2 = time.time()

                    logging.info('Time elapsed: PM: %.3f sec, SR: %.3f sec' %
                                 ((time_step_1 - t_start), (time_step_2 - time_step_1)))

                # if(iii == 2):
                #     imsave(join(result_dir, 'midea.png'),np.round((out_srntt.squeeze() + 1) * 127.5).astype(np.uint8))
                imsave(join(result_dir, 'tmp', 'srntt_%05d.png' % idx),
                       np.round((out_srntt.squeeze() + 1) * 127.5).astype(np.uint8))
                imsave(join(result_dir, 'tmp', 'upscale_%05d.png' % idx),
                       np.round((out_upscale.squeeze() + 1) * 127.5).astype(np.uint8))
                logging.info('Saved to %s' % join(result_dir, 'tmp', 'srntt_%05d.png' % idx))

            t_end = time.time()
            logging.info('Reconstruct SR image')
            out_srntt_files = sorted(glob(join(result_dir, 'tmp', 'srntt_*.png')))
            out_upscale_files = sorted(glob(join(result_dir, 'tmp', 'upscale_*.png')))

            if grids is not None:
                patch_size = grids[0, 2]
                h_l, w_l = grids[-1, 0] + patch_size, grids[-1, 1] + patch_size
                out_upscale_large = np.zeros((h_l, w_l, 3), dtype=np.float32)
                out_srntt_large = np.copy(out_upscale_large)
                counter = np.zeros_like(out_srntt_large, dtype=np.float32)
                for idx in xrange(len(grids)):
                    out_upscale_large[
                    grids[idx, 0]:grids[idx, 0] + patch_size,
                    grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(out_upscale_files[idx], mode='RGB').astype(
                        np.float32)

                    out_srntt_large[
                    grids[idx, 0]:grids[idx, 0] + patch_size,
                    grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(out_srntt_files[idx], mode='RGB').astype(
                        np.float32)

                    counter[
                    grids[idx, 0]:grids[idx, 0] + patch_size,
                    grids[idx, 1]:grids[idx, 1] + patch_size, :] += 1

                out_upscale_large /= counter
                out_srntt_large /= counter
                out_upscale = out_upscale_large[:h * 4, :w * 4, :]
                out_srntt = out_srntt_large[:h * 4, :w * 4, :]
            else:
                out_upscale = imread(out_upscale_files[0], mode='RGB')
                out_srntt = imread(out_srntt_files[0], mode='RGB')

            # log run time
            with open(join(result_dir, 'run_time.txt'), 'w') as f:
                line = '%02d min %02d sec\n' % ((t_end - t_start) // 60, (t_end - t_start) % 60)
                f.write(line)
                f.close()
            # imsave(join(result_dir,'up11.png'), np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8))
            all_upscale.append(np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8))
            all_SRNTT.append(np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8))
            all_bicubic.append(imresize(img_input_copy, 4., interp='bicubic'))
            all_hr.append(img_hr)
        #
        # imsave(join(result_dir, 'up11.png'), all_upscale[0])
        # imsave(join(result_dir, 'up22.png'), all_upscale[1])
        # imsave(join(result_dir, 'up33.png'), all_bupscale[2])

        # print("srntt dimension", all_SRNTT[0].shape)
        # center_left_overlap = all_SRNTT[0]
        # top_left_overlap = all_SRNTT[1][0:240 + overlap, :, :]
        # top_right_overlap = all_SRNTT[1][240 + overlap:480 + 2 * overlap, :, :]
        # bot_left_overlap = all_SRNTT[2][0:240 + overlap, :, :]
        # bot_right_overlap = all_SRNTT[2][240 + overlap:480 + 2 * overlap, :, :]
        # ref_overlap = img_ref[0]
        print("srntt dimension", all_SRNTT[0].shape)
        center_left_overlap = all_SRNTT[0]
        top_left_overlap = all_SRNTT[1][0:int(lr_all_h/2), :, :]
        top_right_overlap = all_SRNTT[1][int(lr_all_h/2):lr_all_h, :, :]
        bot_left_overlap = all_SRNTT[2][0:int(lr_all_h/2), :, :]
        bot_right_overlap = all_SRNTT[2][int(lr_all_h/2):lr_all_h, :, :]
        ref_overlap = img_ref[0]


        # print("center_left_overlap", center_left_overlap.shape)
        # print("top_left_overlap", top_left_overlap.shape)
        # print("bot_left_overlap", bot_left_overlap.shape)
        # print("ref_overlap", ref_overlap.shape)

        full_SRNTT = self.combine_overlap(top_left_overlap, top_right_overlap, center_left_overlap, ref_overlap
                                          , bot_left_overlap, bot_right_overlap, processWidth=overlap*2)

        center_left_upscale = all_upscale[0]
        top_left_upscale = all_upscale[1][0:int(lr_all_h/2), :, :]
        top_right_upscale = all_upscale[1][int(lr_all_h/2):lr_all_h, :, :]
        bot_left_upscale = all_upscale[2][0:int(lr_all_h/2), :, :]
        bot_right_upscale = all_upscale[2][int(lr_all_h/2):lr_all_h, :, :]
        ref_upscale = img_ref[0]
        full_upscale = self.combine_overlap(top_left_upscale, top_right_upscale, center_left_upscale, ref_upscale
                                          , bot_left_upscale, bot_right_upscale, processWidth=overlap*2)

        center_left_bicubic = all_bicubic[0]
        top_left_bicubic = all_bicubic[1][0:int(lr_all_h/2), :, :]
        top_right_bicubic = all_bicubic[1][int(lr_all_h/2):lr_all_h, :, :]
        bot_left_bicubic = all_bicubic[2][0:int(lr_all_h/2), :, :]
        bot_right_bicubic = all_bicubic[2][int(lr_all_h/2):lr_all_h, :, :]
        ref_bicubic = img_ref[0]
        full_bicubic = self.combine_overlap(top_left_bicubic, top_right_bicubic, center_left_bicubic, ref_bicubic
                                          , bot_left_bicubic, bot_right_bicubic, processWidth=overlap*2)
        full_hr = hr_full


        # imsave(join(result_dir, 'center.png'), center_left_overlap)
        # imsave(join(result_dir, 'top_left.png'), top_left_overlap)
        # imsave(join(result_dir, 'top_right.png'), top_right_overlap)
        # imsave(join(result_dir, 'bot_left.png'), bot_left_overlap)
        # imsave(join(result_dir, 'bot_right.png'), bot_right_overlap)
        # imsave(join(result_dir, 'ref.png'), ref_overlap)

        # lr_upscale = all_upscale[0]
        # top_upscale = np.concatenate((all_upscale[1][0:240,:,:],all_upscale[1][240:480,:,:]),1)
        # bottom_upscale = np.concatenate((all_upscale[2][0:240,:,:],all_upscale[2][240:480,:,:]),1)
        #
        # lr_SRNTT = all_SRNTT[0]
        # top_SRNTT = np.concatenate((all_SRNTT[1][0:240,:,:],all_SRNTT[1][240:480,:,:]),1)
        # bottom_SRNTT = np.concatenate((all_SRNTT[2][0:240,:,:],all_SRNTT[2][240:480,:,:]),1)
        #
        # lr_bicubic = all_bicubic[0]
        # top_bicubic = np.concatenate((all_bicubic[1][0:240,:,:],all_bicubic[1][240:480,:,:]),1)
        # bottom_bicubic = np.concatenate((all_bicubic[2][0:240,:,:],all_bicubic[2][240:480,:,:]),1)
        #
        # lr_hr = all_hr[0]
        # top_hr = np.concatenate((all_hr[1][0:240, :, :], all_hr[1][240:480, :, :]), 1)
        # bottom_hr = np.concatenate((all_hr[2][0:240, :, :], all_hr[2][240:480, :, :]), 1)
        #
        # print(top_upscale.shape)
        # print(lr_upscale.shape)
        # print(img_ref[0].shape)
        # print(bottom_upscale.shape)
        # reff = img_ref[0]
        #
        # full_bicubic = np.concatenate((top_bicubic,(np.concatenate((np.concatenate((lr_bicubic,reff),1),bottom_bicubic),0))),0)
        # full_SRNTT = np.concatenate((top_SRNTT,(np.concatenate((np.concatenate((lr_SRNTT,reff),1),bottom_SRNTT),0))),0)
        # # full_SRNTT = np.concatenate((top_upscale, (np.concatenate((np.concatenate((lr_SRNTT, reff), 1), bottom_upscale), 0))), 0)
        # full_upscale = np.concatenate((top_upscale,(np.concatenate((np.concatenate((lr_upscale,reff),1),bottom_upscale),0))),0)
        # full_hr = np.concatenate((top_hr,(np.concatenate((np.concatenate((lr_hr, reff), 1), bottom_hr), 0))),0)

        imsave(join(result_dir, 'Bicubic.png'), full_bicubic)
        imsave(join(result_dir, 'SRNTT.png'), full_SRNTT)
        imsave(join(result_dir, 'upscale.png'), full_upscale)
        imsave(join(result_dir, 'hr.png'), full_hr)

        # print("full_bicubic",full_bicubic.shape)
        # print("full_upscale", full_upscale.shape)
        # print("full_hr", full_hr.shape)
        # print("full_srntt", full_SRNTT.shape)

        sess = tf.InteractiveSession()
        psnr1 = sess.run(tf.image.psnr(full_hr, full_bicubic, max_val=255))
        psnr2 = sess.run(tf.image.psnr(full_hr, full_upscale, max_val=255))
        psnr3 = sess.run(tf.image.psnr(full_hr, full_SRNTT, max_val=255))
        ws_psnr1 = self.cal_psnr(full_hr, full_bicubic)
        ws_psnr2 = self.cal_psnr(full_hr, full_upscale)
        ws_psnr3 = self.cal_psnr(full_hr, full_SRNTT)

        full_hr_tensor = tf.convert_to_tensor(full_hr)
        full_bicubic_tensor = tf.convert_to_tensor(full_bicubic)
        full_upscale_tensor = tf.convert_to_tensor(full_upscale)
        full_SRNTT_tensor = tf.convert_to_tensor(full_SRNTT)

        ssim1 = sess.run(tf.image.ssim(full_hr_tensor, full_bicubic_tensor, max_val=255))
        ssim2 = sess.run(tf.image.ssim(full_hr_tensor, full_upscale_tensor, max_val=255))
        ssim3 = sess.run(tf.image.ssim(full_hr_tensor, full_SRNTT_tensor, max_val=255))

        ws_ssim1 = self.cal_ssim(full_hr, full_bicubic, ssim1)
        ws_ssim2 = self.cal_ssim(full_hr, full_upscale, ssim2)
        ws_ssim3 = self.cal_ssim(full_hr, full_SRNTT, ssim3)

        logging.info('psnr:bic and hr %.3f' % psnr1)
        logging.info('psnr:upscale and hr %.3f' % psnr2)
        logging.info('psnr:srntt and hr %.3f' % psnr3)

        logging.info('wspsnr:bic and hr %.3f' % ws_psnr1)
        logging.info('wspsnr:upscale and hr %.3f' % ws_psnr2)
        logging.info('wspsnr:srntt and hr %.3f' % ws_psnr3)

        logging.info('ssim:bic and hr %.3f' % ssim1)
        logging.info('ssim:upscale and hr %.3f' % ssim2)
        logging.info('ssim:srntt and hr %.3f' % ssim3)

        logging.info('wsssim:bic and hr %.3f' % ws_ssim1)
        logging.info('wsssim:upscale and hr %.3f' % ws_ssim2)
        logging.info('wsssim:srntt and hr %.3f' % ws_ssim3)
        sess.close()
        return psnr1, psnr2, psnr3, ws_psnr1, ws_psnr2, ws_psnr3, ssim1, ssim2, ssim3, ws_ssim1, ws_ssim2, ws_ssim3
        # save results
        # save HR image if it exists
        # if img_hr is not None:
        #     imsave(join(result_dir, 'HR.png'), img_hr)
        # # save LR (input) image
        # imsave(join(result_dir, 'LR.png'), img_input_copy)
        # # save reference image(s)
        # if save_ref:
        #     for idx, ref in enumerate(img_ref):
        #         imsave(join(result_dir, 'Ref_%02d.png' % idx), ref)
        # save bicubic
        # imsave(join(result_dir, 'Bicubic.png'), imresize(img_input_copy, 4., interp='bicubic'))
        # # imsave(join(result_dir, 'Bicubic.png'), imresize(img_input_copy, 2., interp='bicubic'))
        # # save SR images
        # imsave(join(result_dir, 'Upscale.png'), np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8))
        # imsave(join(result_dir, 'SRNTT.png'), np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8))
        # srntt = np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
        # bicubic = imresize(img_input_copy, 4., interp='bicubic')
        # # bicubic = imresize(img_input_copy, 2., interp='bicubic')
        # upscale = np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8)
        #
        # full_hr = np.concatenate((img_hr, ref), axis=1)
        # full = np.concatenate((srntt, ref), axis=1)
        # full_lr = np.concatenate((bicubic, ref), axis=1)
        # full_upscale = np.concatenate((upscale, ref), axis=1)
        #
        # imsave(join(result_dir, 'full.png'), full)
        # imsave(join(result_dir, 'full_lr.png'), full_lr)
        # imsave(join(result_dir, 'full_hr.png'), full_hr)
        #
        # logging.info('Saved results to folder %s' % result_dir)
        # img_srntt = np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
        # img_upscale = np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8)
        # img_bic = imresize(img_input_copy, 4., interp='bicubic')
        # sess = tf.InteractiveSession()
        # psnr_full = tf.image.psnr(full_hr, full, max_val=255)
        # psnr_full_lr = tf.image.psnr(full_hr, full_lr, max_val=255)
        # psnr_full_upscale = tf.image.psnr(full_hr, full_upscale, max_val=255)
        # psnr1 = tf.image.psnr(img_hr, img_bic, max_val=255)
        # psnr2 = tf.image.psnr(img_hr, img_upscale, max_val=255)
        # psnr3 = tf.image.psnr(img_hr, img_srntt, max_val=255)
        # logging.info('psnr:bic and hr %.3f' % sess.run(psnr1))
        # logging.info('psnr:upscale and hr %.3f' % sess.run(psnr2))
        # logging.info('psnr:srntt and hr %.3f' % sess.run(psnr3))
        #
        # logging.info('psnr:full_bic and hr %.3f' % sess.run(psnr_full_lr))
        # logging.info('psnr:full_upscale and hr %.3f' % sess.run(psnr_full_upscale))
        # logging.info('psnr:full_srntt and hr %.3f' % sess.run(psnr_full))
        #
        # logging.info('Saved results to folder %s' % result_dir)
        # psnr1 = sess.run(psnr1)
        # psnr2 = sess.run(psnr2)
        # psnr3 = sess.run(psnr3)
        # psnr_full_lr = sess.run(psnr_full_lr)
        # psnr_full_upscale = sess.run(psnr_full_upscale)
        # psnr_full_srntt = sess.run(psnr_full)
        # # logging.info('返回的类型是', type(psnr1))
        # sess.close()
        #
        # # return np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
        # return psnr1, psnr2, psnr3, psnr_full_lr, psnr_full_upscale, psnr_full_srntt

    def model_get_rid_of_line(
            self,
            inputs,  # LR images, in range of [-1, 1]
            maps=None,  # texture feature maps after texture swapping
            weights=None,  # weights of each pixel on the maps
            is_train=True,
            reuse=False,
            concat=False  # concatenate weights to feature
    ):

        # ********************************************************************************
        # *** content extractor
        # ********************************************************************************
        # print('\tcontent extractor')

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("content_extractor", reuse=reuse):
            layers.set_name_reuse(reuse)
            net = InputLayer(inputs=inputs, name='input')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='n64s1/c')
            temp = net
            for i in range(16):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='n64s1/b1/%s' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='n64s1/b2/%s' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='b_residual_add/%s' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
            content_feature = ElementwiseLayer(layer=[net, temp], combine_fn=tf.add, name='add3')

            # upscaling (4x) for texture extractor
            net = Conv2d(net=content_feature, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='n256s1/1')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='n256s1/2')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

            # output value range is [-1, 1]
            net_upscale = Conv2d(net=net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                                 padding='SAME', W_init=w_init, name='out')

        # *** conditional texture transfer
        with tf.variable_scope("texture_transfer", reuse=reuse):
            layers.set_name_reuse(reuse)
            map_in = InputLayer(inputs=content_feature.outputs, name='content_feature_maps')
            net = Conv2d(net=map_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='small/conv1')
            for i in range(self.num_res_blocks):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='small/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='small/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='small/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='small/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='small/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='small/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='small/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='small/add2')
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='small/conv3')
            net_srntt = Conv2d(net=net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                               padding='SAME', W_init=w_init, name='out')
            return net_upscale, net_srntt

    def train_get_rid_of_line(
            self,
            input_dir='data/train/input',  # original images
            input_dir_truth='data/train/input_truth',
            batch_size=9,
            num_epochs=100,
            learning_rate=1e-4,
            beta1=0.9,
            use_pretrained_model=True,
            is_L1_loss=True,
            input_size=40,
    ):
        if np.sqrt(batch_size) != int(np.sqrt(batch_size)):
            logging.error('The batch size must be the power of an integer.')
            exit(0)
        input_size_w = 480
        input_size_h = 240
        # detect existing model if not use_pretrained_model
        # save_dir 存放模型和sample的地方
        if self.save_dir is None:
            self.save_dir = 'default_save_dir'
        if not use_pretrained_model and exists(join(self.save_dir, MODEL_FOLDER)):
            logging.warning('The existing model dir %s is removed!' % join(self.save_dir, MODEL_FOLDER))
            rmtree(join(self.save_dir, MODEL_FOLDER))


        # create save folders
        for folder in [MODEL_FOLDER, SAMPLE_FOLDER]:
            if not exists(join(self.save_dir, folder)):
                makedirs(join(self.save_dir, folder))

        # check input dir
        # files_input = sorted(glob(join(input_dir, '*.jpg')))
        # files_input_truth = sorted(glob(join(input_dir_truth, '*.jpg')))

        files_input = sorted(glob(join(input_dir, '*.png')))
        files_input_truth = sorted(glob(join(input_dir_truth, '*.png')))

        num_files = len(files_input)
        print("len num_files", num_files)
        print("len num_truth_files", len(files_input_truth))
        assert num_files == len(files_input_truth)

        # ********************************************************************************
        # *** build graph
        # ********************************************************************************
        logging.info('Building graph ...')
        # input LR images, range [-1, 1]
        self.input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size_h, input_size_w, 3])

        # original images, range [-1, 1]
        self.ground_truth = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size_h, input_size_w, 3])

        self.weights = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size_h, input_size_w])
        # SRNTT network
        self.net_upscale, self.net_srntt = self.model_get_rid_of_line(self.input)

        # reconstruction loss
        if is_L1_loss:
            loss_reconst = tf.reduce_mean(tf.abs(self.net_srntt.outputs - self.ground_truth))
        else:
            loss_reconst = cost.mean_squared_error(self.net_srntt.outputs, self.ground_truth, is_mean=True)

        # total loss
        loss = loss_reconst

        # ********************************************************************************
        # *** optimizers
        # ********************************************************************************
        # trainable variables
        trainable_vars = tf.trainable_variables()
        var_g = [v for v in trainable_vars if 'texture_transfer' in v.name]

        # learning rate decay
        global_step = tf.Variable(0, trainable=False, name='global_step')
        num_batches = int(num_files / batch_size)
        decayed_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=max(num_epochs * num_batches / 2, 1),
            decay_rate=.1,
            staircase=True
        )

        # optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=decayed_learning_rate, beta1=beta1).minimize(loss, var_list=var_g, global_step=global_step)

        # ********************************************************************************
        # *** samples for monitoring the training process
        # ********************************************************************************
        np.random.seed(2019)
        idx = np.random.choice(np.arange(num_files), batch_size, replace=False)
        # # sample_in是没有拼接缝的图
        # samples_in = [imread(files_input_truth[i], mode='RGB') for i in idx]
        # # samples_input是有拼缝的图
        # samples_input = [imread(files_input[i], mode='RGB') for i in idx]

        samples_in = [imread(files_input[i], mode='RGB') for i in idx]
        samples_input = [imresize(img, (input_size_h, input_size_w), interp='bicubic').astype(np.float32) / 127.5 - 1
                         for img in samples_in]

        samples_in_truth = [imread(files_input_truth[i], mode='RGB') for i in idx]
        samples_input_truth = [
            imresize(img_truth, (input_size_h, input_size_w), interp='bicubic').astype(np.float32) / 127.5 - 1
            for img_truth in samples_in_truth]

        samples_weight_map = np.zeros(shape=(batch_size, input_size_h, input_size_w))
        frame_size = int(np.sqrt(batch_size))
        # vis.save_images(np.array(samples_input_truth), [frame_size, frame_size], join(self.save_dir, SAMPLE_FOLDER, 'HR.png'))
        vis.save_images(np.round((np.array(samples_input) + 1) * 127.5).astype(np.uint8), [frame_size, frame_size],
                        join(self.save_dir, SAMPLE_FOLDER, 'LR.png'))
        vis.save_images(np.round((np.array(samples_input_truth) + 1) * 127.5).astype(np.uint8),
                        [frame_size, frame_size],
                        join(self.save_dir, SAMPLE_FOLDER, 'HR.png'))

        # ********************************************************************************
        # *** load models and training
        # ********************************************************************************
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            logging.info('Loading models ...')
            tf.global_variables_initializer().run()

            # load pre-trained upscaling.
            model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['content_extractor'])
            if files.load_and_assign_npz(
                    sess=sess,
                    name=model_path,
                    network=self.net_upscale) is False:
                logging.error('FAILED load %s' % model_path)
                exit(0)
            logging.info('**********'
                         ' Start training '
                         '**********')
            # pre-train with only reconstruction loss
            current_eta = None
            idx = np.arange(num_files)
            for epoch in xrange(20):
                np.random.shuffle(idx)
                for n_batch in xrange(num_batches):
                    step_time = time.time()
                    sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]

                    batch_imgs = [imread(files_input[i], mode='RGB') for i in sub_idx]
                    batch_imgs_truth = [imread(files_input_truth[i], mode='RGB') for i in sub_idx]
                    batch_truth = [imresize(img, .25, interp='bicubic').astype(np.float32) / 127.5 - 1 for img in
                                   batch_imgs_truth]
                    batch_input = [imresize(img, .25, interp='bicubic').astype(np.float32) / 127.5 - 1 for img in
                                   batch_imgs]
                    batch_weights = np.zeros(shape=(batch_size, input_size_h, input_size_w))

                    # train with truth
                    _, l_reconst = sess.run(
                        fetches=[optimizer, loss_reconst],
                        feed_dict={
                            self.input: batch_input,
                            self.ground_truth: batch_truth,
                            self.weights: np.ones_like(np.array(batch_weights))
                        }
                    )

                    # print
                    time_per_iter = time.time() - step_time
                    n_iter_remain = (20 - epoch - 1) * num_batches + num_batches - n_batch
                    eta_str, eta_ = self.eta(time_per_iter, n_iter_remain, current_eta)
                    current_eta = eta_
                    logging.info('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\n'
                                 '\tl_rec = %.4f' %
                                 (epoch + 1, 20, n_batch + 1, num_batches, eta_str,
                                  l_reconst))
                # save intermediate results
                vis.save_images(
                    np.round((self.net_srntt.outputs.eval({
                        self.input: samples_input,
                        self.weights: samples_weight_map}) + 1) * 127.5).astype(np.uint8),
                    [frame_size, frame_size],
                    join(self.save_dir, SAMPLE_FOLDER, 'init_E%03d.png' % (epoch + 1)))

                # save models for each epoch
                files.save_npz(
                    save_list=self.net_srntt.all_params,
                    name=join(self.save_dir, MODEL_FOLDER, 'srntt%03d.npz' % (epoch + 1)),
                    sess=sess)
    # def test_without_ref(
    #         self,
    #         input_dir,  # original image
    #         ref_dir=None,  # reference images
    #         use_pretrained_model=True,
    #         use_init_model_only=False,  # the init model is trained only with the reconstruction loss
    #         use_weight_map=False,
    #         result_dir=None,
    #         ref_scale=1.0,
    #         is_original_image=True,
    #         max_batch_size=16,
    #         save_ref=True
    # ):
    #     logging.info('Testing without references')
    #
    #     # ********************************************************************************
    #     # *** check input and reference images
    #     # ********************************************************************************
    #     # check input_dir
    #     img_input, img_hr = None, None
    #     if isinstance(input_dir, np.ndarray):
    #         assert len(input_dir.shape) == 3
    #         img_input = np.copy(input_dir)
    #     elif isfile(input_dir):
    #         img_input = imread(input_dir, mode='RGB')
    #     else:
    #         logging.info('Unrecognized input_dir %s' % input_dir)
    #         exit(0)
    #     input_block = []
    #     input_block.append(img_input[:,0:960,:])
    #     input_block.append(img_input[:,960:1920,:])
    #     input_block.append(img_input[:,1920:2880,:])
    #     for item in input_block:
    #         img_input = item
    #         h, w, _ = img_input.shape
    #         if is_original_image:
    #             # ensure that the size of img_input can be divided by 4 with no remainder
    #             h = int(h // 4 * 4)
    #             w = int(w // 4 * 4)
    #             img_hr = img_input[0:h, 0:w, ::]
    #             img_input = imresize(img_hr, .25, interp='bicubic')
    #             h, w, _ = img_input.shape
    #         img_input_copy = np.copy(img_input)
    #
    #         if h * w * 16 > SRNTT.MAX_IMAGE_SIZE:  # avoid OOM
    #             # split img_input into patches
    #             patches = []
    #             grids = []
    #             patch_size = 128
    #             stride = 100
    #             for ind_row in range(0, h - (patch_size - stride), stride):
    #                 for ind_col in range(0, w - (patch_size - stride), stride):
    #                     patch = img_input[ind_row:ind_row + patch_size, ind_col:ind_col + patch_size, :]
    #                     if patch.shape != (patch_size, patch_size, 3):
    #                         patch = np.pad(patch,
    #                                        ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)),
    #                                        'reflect')
    #                     patches.append(patch)
    #                     grids.append((ind_row * 4, ind_col * 4, patch_size * 4))
    #             grids = np.stack(grids, axis=0)
    #             img_input = np.stack(patches, axis=0)
    #         else:
    #             grids = None
    #             img_input = np.expand_dims(img_input, axis=0)
    #
    #         # check ref_dir
    #         is_ref = True
    #         if ref_dir is None:
    #             is_ref = False
    #             ref_dir = input_dir
    #
    #         img_ref = []
    #
    #         if not isinstance(ref_dir, (list, tuple)):
    #             ref_dir = [ref_dir]
    #
    #         for ref in ref_dir:
    #             if isinstance(ref, np.ndarray):
    #                 assert len(ref.shape) == 3
    #                 img_ref.append(np.copy(ref))
    #             elif isfile(ref):
    #                 img_ref.append(imread(ref, mode='RGB'))
    #             else:
    #                 logging.info('Unrecognized ref_dir type!')
    #                 exit(0)
    #
    #         if ref_scale <= 0:  # keep the same scale as HR image
    #             img_ref = [imresize(img, (h * 4, w * 4), interp='bicubic') for img in img_ref]
    #         elif ref_scale != 1:
    #             img_ref = [imresize(img, float(ref_scale), interp='bicubic') for img in img_ref]
    #
    #         for i in xrange(len(img_ref)):
    #             h2, w2, _ = img_ref[i].shape
    #             h2 = int(h2 // 4 * 4)
    #             w2 = int(w2 // 4 * 4)
    #             img_ref[i] = img_ref[i][0:h2, 0:w2, ::]
    #             if not is_ref and is_original_image:
    #                 img_ref[i] = imresize(img_ref[i], .25, interp='bicubic')
    #
    #         # create result folder
    #         if result_dir is None:
    #             result_dir = join(self.save_dir, 'test')
    #         if not exists(result_dir):
    #             makedirs(result_dir)
    #         if not exists(join(result_dir, 'tmp')):
    #             makedirs(join(result_dir, 'tmp'))
    #
    #         # ********************************************************************************
    #         # *** build graph
    #         # ********************************************************************************
    #         if not self.is_model_built:
    #             self.is_model_built = True
    #             logging.info('Building graph ...')
    #             # input image, range [-1, 1]
    #             self.input_srntt = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)
    #
    #             # reference images, range [0, 255]
    #             self.input_vgg19 = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)
    #
    #             # swapped feature map and weights
    #             self.maps = (
    #                 tf.placeholder(
    #                     dtype=tf.float32,
    #                     shape=(1, None, None, 256)),
    #                 tf.placeholder(
    #                     dtype=tf.float32,
    #                     shape=(1, None, None, 128)),
    #                 tf.placeholder(
    #                     dtype=tf.float32,
    #                     shape=(1, None, None, 64))
    #             )
    #
    #             self.weights = tf.placeholder(
    #                 dtype=tf.float32,
    #                 shape=(1, None, None))
    #
    #             # SRNTT network
    #             logging.info('Build SRNTT model')
    #             if use_weight_map:
    #                 self.net_upscale, self.net_srntt = self.model(
    #                     self.input_srntt, self.maps, weights=tf.expand_dims(self.weights, axis=-1), is_train=False)
    #             else:
    #                 self.net_upscale, self.net_srntt = self.model(self.input_srntt, self.maps, is_train=False)
    #
    #             # VGG19 network, input range [0, 255]
    #             logging.info('Build VGG19 model')
    #             self.net_vgg19 = VGG19(
    #                 input_image=self.input_vgg19,
    #                 model_path=self.vgg19_model_path,
    #                 final_layer='relu3_1'
    #             )
    #
    #             # ********************************************************************************
    #             # *** load models
    #             # ********************************************************************************
    #             config = tf.ConfigProto()
    #             config.gpu_options.allow_growth = False
    #             self.sess = tf.Session(config=config)
    #
    #             # instant of Swap()
    #             logging.info('Initialize the swapper')
    #             self.swaper = Swap(sess=self.sess)
    #
    #             logging.info('Loading models ...')
    #             self.sess.run(tf.global_variables_initializer())
    #
    #             # load pre-trained content extractor, including upscaling.
    #             model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['content_extractor'])
    #             if files.load_and_assign_npz(
    #                     sess=self.sess,
    #                     name=model_path,
    #                     network=self.net_upscale) is False:
    #                 logging.error('FAILED load %s' % model_path)
    #                 exit(0)
    #
    #             # load the specific conditional texture transfer model, specified by save_dir
    #             if self.save_dir is None:
    #                 if use_init_model_only:
    #                     model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['init'])
    #                     if files.load_and_assign_npz(
    #                             sess=self.sess,
    #                             name=model_path,
    #                             network=self.net_srntt):
    #                         logging.info('SUCCESS load %s' % model_path)
    #                     else:
    #                         logging.error('FAILED load %s' % model_path)
    #                         exit(0)
    #                 else:
    #                     model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['conditional_texture_transfer'])
    #                     if files.load_and_assign_npz(
    #                             sess=self.sess,
    #                             name=model_path,
    #                             network=self.net_srntt):
    #                         logging.info('SUCCESS load %s' % model_path)
    #                     else:
    #                         logging.error('FAILED load %s' % model_path)
    #                         exit(0)
    #             else:
    #                 if use_init_model_only:
    #                     model_path = join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['init'])
    #                     if files.load_and_assign_npz(
    #                             sess=self.sess,
    #                             name=model_path,
    #                             network=self.net_srntt):
    #                         logging.info('SUCCESS load %s' % model_path)
    #                     else:
    #                         logging.error('FAILED load %s' % model_path)
    #                         exit(0)
    #                 else:
    #                     model_path = join(self.save_dir, MODEL_FOLDER,
    #                                       SRNTT_MODEL_NAMES['conditional_texture_transfer'])
    #                     if files.load_and_assign_npz(
    #                             sess=self.sess,
    #                             name=model_path,
    #                             network=self.net_srntt):
    #                         logging.info('SUCCESS load %s' % model_path)
    #                     else:
    #                         logging.error('FAILED load %s' % model_path)
    #                         exit(0)
    #
    #         logging.info('**********'
    #                      ' Start testing '
    #                      '**********')
    #
    #         matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
    #
    #         logging.info('Get VGG19 Feature Maps')
    #
    #         logging.info('\t[1/2] Getting feature map of Ref image ...')
    #         t_start = time.time()
    #         map_ref = []
    #         for i in img_ref:
    #             map_ref.append(
    #                 self.net_vgg19.get_layer_output(
    #                     sess=self.sess, layer_name=matching_layer,
    #                     feed_image=i)
    #             )
    #         styles = [[] for _ in xrange(len(matching_layer))]
    #         for i in map_ref:
    #             for j in xrange(len(styles)):
    #                 styles[j].append(i[j])
    #
    #         logging.info('\t[2/2] Getting feature map of LR->SR Ref image ...')
    #         map_ref_sr = []
    #         if is_ref:
    #             for i in img_ref:
    #                 img_ref_downscale = imresize(i, .25, interp='bicubic')
    #                 img_ref_upscale = self.net_upscale.outputs.eval({self.input_srntt: [img_ref_downscale / 127.5 - 1]},
    #                                                                 session=self.sess)
    #                 img_ref_upscale = (img_ref_upscale + 1) * 127.5
    #                 map_ref_sr.append(
    #                     self.net_vgg19.get_layer_output(
    #                         sess=self.sess, layer_name=matching_layer[0],
    #                         feed_image=img_ref_upscale)
    #                 )
    #         else:
    #             map_ref_sr = styles
    #
    #         # swap ref to in
    #         logging.info('Patch-Wise Matching and Swapping')
    #         for idx, patch in enumerate(img_input):
    #             logging.info('\tPatch %03d/%03d' % (idx + 1, img_input.shape[0]))
    #
    #             # skip if the results exists
    #             if exists(join(result_dir, 'tmp', 'srntt_%05d.png' % idx)):
    #                 continue
    #
    #             logging.info('\tGetting feature map of input LR image ...')
    #
    #             if 'Urban' in input_dir:
    #                 img_input_upscale = imread(
    #                     join('../EDSR-PyTorch/test_Urban100_MDSR', split(input_dir)[-1], 'SRNTT.png'),
    #                     mode='RGB').astype(np.float32)
    #             elif 'CUFED5' in input_dir and False:
    #                 img_input_upscale = imread(
    #                     join('../EDSR-PyTorch/test_CUFED5_MDSR', split(input_dir)[-1], 'SRNTT.png'),
    #                     mode='RGB').astype(np.float32)
    #             elif 'Sun80' in input_dir or 'sun80' in input_dir:
    #                 img_input_upscale = imread(
    #                     join('../EDSR-PyTorch/test_Sun100_MDSR', split(input_dir)[-1].split('.')[0], 'SRNTT.png'),
    #                     mode='RGB').astype(np.float32)
    #             else:
    #                 img_input_upscale = self.net_upscale.outputs.eval({self.input_srntt: [patch / 127.5 - 1]},
    #                                                                   session=self.sess)
    #                 img_input_upscale = (img_input_upscale + 1) * 127.5
    #
    #             if is_ref:
    #                 map_sr = self.net_vgg19.get_layer_output(
    #                     sess=self.sess, layer_name=matching_layer[0], feed_image=img_input_upscale)
    #             else:
    #                 map_sr = self.net_vgg19.get_layer_output(
    #                     sess=self.sess, layer_name=matching_layer, feed_image=img_input_upscale)
    #
    #             logging.info('\tMatching and swapping features ...')
    #             if is_ref:
    #                 map_target, weight, _ = self.swaper.conditional_swap_multi_layer(
    #                     content=map_sr,
    #                     style=styles[0],
    #                     condition=map_ref_sr,
    #                     other_styles=styles[1:],
    #                     is_weight=use_weight_map
    #                 )
    #             else:
    #                 map_target, weight = [], []
    #                 for i in xrange(len(matching_layer)):
    #                     m_target, w, _ = self.swaper.conditional_swap_multi_layer(
    #                         content=map_sr[i],
    #                         style=styles[i],
    #                         condition=map_ref_sr[i],
    #                         is_weight=use_weight_map
    #                     )
    #                     map_target.append(np.squeeze(m_target))
    #                     weight.append(w)
    #
    #             logging.info('Obtain SR patches')
    #             if use_weight_map:
    #                 weight = np.pad(weight, ((1, 1), (1, 1)), 'edge')
    #                 out_srntt, out_upscale = self.sess.run(
    #                     fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
    #                     feed_dict={
    #                         self.input_srntt: [patch / 127.5 - 1],
    #                         self.maps: [np.expand_dims(m, axis=0) for m in map_target],
    #                         self.weights: [weight]
    #                     }
    #                 )
    #             else:
    #                 time_step_1 = time.time()
    #                 out_srntt, out_upscale = self.sess.run(
    #                     fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
    #                     feed_dict={
    #                         self.input_srntt: [patch / 127.5 - 1],
    #                         self.maps: [np.expand_dims(m, axis=0) for m in map_target],
    #                     }
    #                 )
    #                 time_step_2 = time.time()
    #
    #                 logging.info('Time elapsed: PM: %.3f sec, SR: %.3f sec' %
    #                              ((time_step_1 - t_start), (time_step_2 - time_step_1)))
    #
    #             imsave(join(result_dir, 'tmp', 'srntt_%05d.png' % idx),
    #                    np.round((out_srntt.squeeze() + 1) * 127.5).astype(np.uint8))
    #             imsave(join(result_dir, 'tmp', 'upscale_%05d.png' % idx),
    #                    np.round((out_upscale.squeeze() + 1) * 127.5).astype(np.uint8))
    #             logging.info('Saved to %s' % join(result_dir, 'tmp', 'srntt_%05d.png' % idx))
    #         logging.info('Reconstruct SR image')
    #
    #         ########################################
    #         out_srntt_files = sorted(glob(join(result_dir, 'tmp', 'srntt_*.png')))
    #         out_upscale_files = sorted(glob(join(result_dir, 'tmp', 'upscale_*.png')))
    #
    #         if grids is not None:
    #             patch_size = grids[0, 2]
    #             h_l, w_l = grids[-1, 0] + patch_size, grids[-1, 1] + patch_size
    #             out_upscale_large = np.zeros((h_l, w_l, 3), dtype=np.float32)
    #             out_srntt_large = np.copy(out_upscale_large)
    #             counter = np.zeros_like(out_srntt_large, dtype=np.float32)
    #             for idx in xrange(len(grids)):
    #                 out_upscale_large[
    #                 grids[idx, 0]:grids[idx, 0] + patch_size,
    #                 grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(out_upscale_files[idx], mode='RGB').astype(
    #                     np.float32)
    #
    #                 out_srntt_large[
    #                 grids[idx, 0]:grids[idx, 0] + patch_size,
    #                 grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(out_srntt_files[idx], mode='RGB').astype(
    #                     np.float32)
    #
    #                 counter[
    #                 grids[idx, 0]:grids[idx, 0] + patch_size,
    #                 grids[idx, 1]:grids[idx, 1] + patch_size, :] += 1
    #
    #             out_upscale_large /= counter
    #             out_srntt_large /= counter
    #             out_upscale = out_upscale_large[:(h * 4), :(w * 4), :]
    #             out_srntt = out_srntt_large[:(h * 4), :(w * 4), :]
    #         else:
    #             out_upscale = imread(out_upscale_files[0], mode='RGB')
    #             out_srntt = imread(out_srntt_files[0], mode='RGB')
    #
    #         t_end = time.time()
    #
    #         # log run time
    #         with open(join(result_dir, 'run_time.txt'), 'w') as f:
    #             line = '%02d min %02d sec\n' % ((t_end - t_start) // 60, (t_end - t_start) % 60)
    #             f.write(line)
    #             f.close()
    #
    #     # save results
    #     # save HR image if it exists
    #     if img_hr is not None:
    #         imsave(join(result_dir, 'HR.png'), img_hr)
    #     # save LR (input) image
    #     imsave(join(result_dir, 'LR.png'), img_input_copy)
    #     # save reference image(s)
    #     if save_ref:
    #         for idx, ref in enumerate(img_ref):
    #             imsave(join(result_dir, 'Ref_%02d.png' % idx), ref)
    #     # save bicubic
    #     imsave(join(result_dir, 'Bicubic.png'), imresize(img_input_copy, 4., interp='bicubic'))
    #     # save SR images
    #     imsave(join(result_dir, 'Upscale.png'), np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8))
    #     imsave(join(result_dir, 'SRNTT.png'), np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8))
    #     srntt = np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
    #     return np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8)
