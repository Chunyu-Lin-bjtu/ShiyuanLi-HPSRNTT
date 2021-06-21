import tensorflow as tf
import numpy as np
import os


class Swap(object):
    def __init__(self, patch_size=3, stride=1, sess=None):
        self.patch_size = patch_size
        self.stride = stride
        self.sess = sess
        self.content = None
        self.style = None
        self.condition = None
        self.conv_input = tf.placeholder(dtype=tf.float32, shape=[1, None, None, None], name='swap_input')
        self.conv_filter = tf.placeholder(dtype=tf.float32, shape=[self.patch_size, self.patch_size, None, None],
                                          name='swap_filter')
        self.conv = tf.nn.conv2d(
            input=self.conv_input,
            filter=self.conv_filter,
            strides=(1, self.stride, self.stride, 1),
            padding='VALID',
            name='feature_swap'
        )

    def style2patches(self, feature_map=None):
        """
        sample patches from the style (reference) map
        :param feature_map: array, [H, W, C]
        :return: array (conv kernel), [H, W, C_in, C_out]
        """
        if feature_map is None:
            feature_map = self.style
        h, w, c = feature_map.shape
        patches = []
        for ind_row in range(0, h - self.patch_size + 1, self.stride):
            for ind_col in range(0, w - self.patch_size + 1, self.stride):
                patches.append(feature_map[ind_row:ind_row + self.patch_size, ind_col:ind_col + self.patch_size, :])
        return np.stack(patches, axis=-1)

    def conditional_swap_multi_layer(self, content, style, condition, patch_size=3, stride=1, other_styles=None,
                                     is_weight=False):
        """
        feature swapping with multiple references on multiple feature layers
        :param content: array (h, w, c), feature map of content
        :param style: list of array [(h, w, c)], feature map of each reference
        :param condition: list of array [(h, w, c)], augmented feature map of each reference for matching with content map
        :param patch_size: int, size of matching patch
        :param stride: int, stride of sliding the patch
        :param other_styles: list (different layers) of lists (different references) of array (feature map),
                [[(h_, w_, c_)]], feature map of each reference from other layers
        :param is_weight, bool, whether compute weights
        :return: swapped feature maps - [3D array, ...], matching weights - 2D array, matching idx - 2D array
        """
        assert isinstance(content, np.ndarray)
        self.content = np.squeeze(content)
        # [40,40,256]
        assert len(self.content.shape) == 3
        # print("在swap函数中——————————————————————")
        # print("shape of content", self.content.shape)

        assert isinstance(style, list)
        self.style = [np.squeeze(s) for s in style]
        assert all([len(self.style[i].shape) == 3 for i in range(len(self.style))])
        # print("shape of style", np.array(self.style).shape)

        assert isinstance(condition, list)
        self.condition = [np.squeeze(c) for c in condition]
        assert all([len(self.condition[i].shape) == 3 for i in range(len(self.condition))])
        assert len(self.condition) == len(self.style)

        num_channels = self.content.shape[-1]
        assert all([self.style[i].shape[-1] == num_channels for i in range(len(self.style))])
        # assert all([self.condition[i].shape[-1] == num_channels for i in range(len(self.condition))])
        assert all([self.style[i].shape == self.condition[i].shape for i in range(len(self.style))])

        if other_styles is not None:
            assert isinstance(other_styles, list)
            assert all([isinstance(s, list) for s in other_styles])
            other_styles = [[np.squeeze(s) for s in styles] for styles in other_styles]
            # print("shape of other_styles", np.array(other_styles).shape)
            assert all([all([len(s.shape) == 3 for s in styles]) for styles in other_styles])

        self.patch_size = patch_size
        self.stride = stride

        # split content, style, and condition into patches
        # content是待超分辨率的原图。style是参考图，用于替代。condition是模糊的参考图，用于匹配
        # 把content切成3*3的patch（stride=1），然后沿着-1轴连接到一起(因为style和condition都是列表，所以不是直接调用函数的，而是用的map)
        patches_content = self.style2patches(self.content)
        # print("shape of patches_content", patches_content.shape)
        # 把style切成3*3的patch（stride=1），然后沿着-1轴连接到一起
        patches_style = np.concatenate(list(map(self.style2patches, self.style)), axis=-1)
        # 把condition切成3*3的patch（stride=1），然后沿着-1轴连接到一起
        patches = np.concatenate(list(map(self.style2patches, self.condition)), axis=-1)
        # print("shape of patches", patches.shape)

        # normalize content and condition patches
        norm = np.sqrt(np.sum(np.square(patches), axis=(0, 1, 2)))
        patches_style_normed = patches / norm
        norm = np.sqrt(np.sum(np.square(patches_content), axis=(0, 1, 2)))
        patches_content_normed = patches_content / norm
        # 2 3应该比1多一个维度（因为2 3 是list拼接起来的），4 5 应该是一样的
        # print("shape of patches_content_normed", patches_content_normed.shape)
        # print("shape of patches_style_normed", patches_style_normed.shape)
        # print("shape of patches_style", patches_style.shape)
        # print("shape of content", content.shape)
        # print("shape of style[0]", style[0].shape)
        # print("shape of style", np.array(style).shape)
        # print("shape of other_style", np.array(other_styles).shape)
        # print("shape of other styles[0][0]", other_styles[0][0].shape)
        # print("shape of other styles[1][0]", other_styles[1][0].shape)
        del norm, patches, patches_content
        # 最后留下：
        # patches_style:没有归一化的，高分辨率的切块的ref的vgg特征
        # patches_content_normed:归一化后的(归一化前对应patches_content,也就是content,也就是bicubic之后的LR)，LR的切块后的vgg特征
        # patches_style_normed:归一化后的(归一化前对应patches,也就是condition,也就是用于特征匹配的ref)，ref降再升之后的切块后的vgg特征
        # match content and condition patches (batch-wise matching because of memory limitation)
        # the size of a batch is 512MB
        batch_size = int(1024. ** 2 * 512 / (self.content.shape[0] * self.content.shape[1]))
        num_out_channels = patches_style_normed.shape[-1]
        print('\tMatching ...')
        max_idx, max_val = None, None
        for idx in range(0, num_out_channels, batch_size):
            print('\t  Batch %02d/%02d' % (idx / batch_size + 1, np.ceil(1. * num_out_channels / batch_size)))
            batch = patches_style_normed[..., idx:idx + batch_size]
            # print("the shape of batch", np.array(batch).shape)
            # print("the shape of content", self.content.shape)
            if self.sess:
                corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch}, session=self.sess)
            else:
                corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch})
            # print("the shape of corr", np.array(corr).shape)
            corr = np.squeeze(corr)
            # print("the shape of corr(squeeze)", np.array(corr).shape)
            max_idx_tmp = np.argmax(corr, axis=-1) + idx
            max_val_tmp = np.max(corr, axis=-1)
            # print("the shape of max_idx_tmp", max_idx_tmp.shape)
            # print("the shape of max_val_tmp", max_val_tmp.shape)
            del corr, batch
            if max_idx is None:
                max_idx, max_val = max_idx_tmp, max_val_tmp
            else:
                indices = max_val_tmp > max_val
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices]

        # compute matching similarity (inner product)
        if is_weight:
            print('\tWeighting ...')
            corr2 = np.matmul(
                np.transpose(np.reshape(patches_content_normed, (-1, patches_content_normed.shape[-1]))),
                np.reshape(patches_style_normed, (-1, patches_style_normed.shape[-1]))
            )
            weights = np.reshape(np.max(corr2, axis=-1), max_idx.shape)
            del patches_content_normed, patches_style_normed, corr2
        else:
            weights = None
            del patches_content_normed, patches_style_normed

        # stitch matches style patches according to content spacial structure
        print('\tSwapping ...')
        maps = []
        target_map = np.zeros_like(self.content)
        count_map = np.zeros(shape=target_map.shape[:2])
        for i in range(max_idx.shape[0]):
            for j in range(max_idx.shape[1]):
                target_map[i:i + self.patch_size, j:j + self.patch_size, :] += patches_style[:, :, :, max_idx[i, j]]
                count_map[i:i + self.patch_size, j:j + self.patch_size] += 1.0
        target_map = np.transpose(target_map, axes=(2, 0, 1)) / count_map
        target_map = np.transpose(target_map, axes=(1, 2, 0))
        maps.append(target_map)

        # stitch other styles
        patch_size, stride = self.patch_size, self.stride
        if other_styles:
            for style in other_styles:
                ratio = float(style[0].shape[0]) / self.style[0].shape[0]
                # print("ratio:", ratio)
                # print("shape of style[0].shape[0]", np.array(self.style[0]).shape[0])
                # print("ratio is", ratio)
                assert int(ratio) == ratio
                ratio = int(ratio)
                self.patch_size = patch_size * ratio
                self.stride = stride * ratio
                patches_style = np.concatenate(list(map(self.style2patches, style)), axis=-1)
                target_map = np.zeros((self.content.shape[0] * ratio, self.content.shape[1] * ratio, style[0].shape[2]))
                count_map = np.zeros(shape=target_map.shape[:2])
                # print("stride:", stride)
                # print("patch_size:",patch_size)
                # print("self.stride:", self.stride)
                # print("self.patch_size:",self.patch_size)
                # print("the shape of patches_style", patches_style.shape)
                # print("the shape of style", np.array(style).shape)
                # print("the shape of target_map", target_map.shape)
                # print("the shape of count_map", count_map.shape)
                for i in range(max_idx.shape[0]):
                    for j in range(max_idx.shape[1]):
                        target_map[i * ratio:i * ratio + self.patch_size, j * ratio:j * ratio + self.patch_size,
                        :] += patches_style[:, :, :, max_idx[i, j]]
                        count_map[i * ratio:i * ratio + self.patch_size, j * ratio:j * ratio + self.patch_size] += 1.0
                target_map = np.transpose(target_map, axes=(2, 0, 1)) / count_map
                target_map = np.transpose(target_map, axes=(1, 2, 0))
                # print("the shape of target_map", target_map.shape)
                maps.append(target_map)
            # maps[0] = np.zeros(maps[0].shape)
            # maps[2] = np.zeros(maps[2].shape)
            print("maps dimension", np.array(maps).shape)
            print("maps[0] dimension", maps[0].shape)
            print("maps[1] dimension", maps[1].shape)
            print("maps[2] dimension", maps[2].shape)


        return maps, weights, max_idx
