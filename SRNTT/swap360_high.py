import tensorflow as tf
import numpy as np
import os
import math


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

    def get_weight_360(self, H, W):
        # H W是中间块的一半的大小
        w_center = np.zeros((H, W, 1))
        w_top = np.zeros((int(H / 2), 2 * W, 1))
        w_bottom = np.zeros((int(H / 2), 2 * W, 1))
        for i in range(0, H):
            # 整幅图的时候是从0开始，中间部分的话要从120开始
            for j in range(0, W):
                wij = math.cos((i + H / 2 + 0.5 - H) * math.pi / (2 * H))
                w_center[i, j, :] = wij

        for i in range(0, int(H / 2)):
            for j in range(0, 2 * W):
                wij = math.cos((i + 0.5 - H) * math.pi / (2 * H))
                w_top[i, j, :] = wij
        for i in range(0, int(H / 2)):
            for j in range(0, 2 * W):
                wij = math.cos((i + 3 * H / 2 + 0.5 - H) * math.pi / (2 * H))
                w_bottom[i, j, :] = wij

        w_center = w_center.astype(np.float32)
        w_top = w_top.astype(np.float32)
        w_bottom = w_bottom.astype(np.float32)
        w_top = np.concatenate((w_top[:, 0:W, :], w_top[:, W:2 * W, :]), axis=0)
        w_bottom = np.concatenate((w_bottom[:, 0:W, :], w_bottom[:, W:2 * W, :]), axis=0)
        # print("center", w_center.shape)
        # print("top", w_top.shape)
        # print("bot", w_bottom.shape)
        return w_center, w_top, w_bottom

    def write_relu(row):
        path = "corr_list_relu13.csv"
        with open(path, 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(row)

    def conditional_swap_multi_layer(self, content, style, condition, patch_size=3, stride=1, other_styles=None,
                                     other_centers=None, other_tops=None, other_bottoms=None, is_weight=False):
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
        # assert isinstance(content, np.ndarray)
        # self.content = content
        self.content = [np.squeeze(q) for q in content]
        # self.content[0] = np.squeeze(content[0])
        # self.content[1] = np.squeeze(content[1])
        # self.content[2] = np.squeeze(content[2])

        assert isinstance(style, list)
        self.style = [np.squeeze(s) for s in style]
        assert all([len(self.style[i].shape) == 3 for i in range(len(self.style))])

        assert isinstance(condition, list)
        self.condition = [np.squeeze(c) for c in condition]
        assert all([len(self.condition[i].shape) == 3 for i in range(len(self.condition))])
        assert len(self.condition) == len(self.style)

        num_channels = self.content[0].shape[-1]
        assert all([self.style[i].shape[-1] == num_channels for i in range(len(self.style))])
        assert all([self.style[i].shape == self.condition[i].shape for i in range(len(self.style))])

        # other_conditions = [[np.squeeze(s) for s in conditions] for conditions in other_conditions]
        other_styles = [[np.squeeze(s) for s in styles] for styles in other_styles]
        other_centers = [[np.squeeze(s) for s in centers] for centers in other_centers]
        other_tops = [[np.squeeze(s) for s in tops] for tops in other_tops]
        other_bottoms = [[np.squeeze(s) for s in bottoms] for bottoms in other_bottoms]

        self.patch_size = patch_size
        self.stride = stride
        # final_corr = 0
        # print("condition",self.condition[0].shape)
        patches = np.concatenate(list(map(self.style2patches, self.condition)), axis=-1)
        norm = np.sqrt(np.sum(np.square(patches), axis=(0, 1, 2)))
        patches_style_normed = patches / norm
        del norm, patches
        batch_size = int(1024. ** 2 * 512 / (self.content[0].shape[0] * self.content[0].shape[1]))
        num_out_channels = patches_style_normed.shape[-1]
        # print("num_out_channels",num_out_channels)
        print('\tMatching ...')
        final_corr=0
        final_var=0
        max_idx1, max_val1, max_idx2, max_val2, max_idx3, max_val3 = None, None, None, None, None, None
        for idx in range(0, num_out_channels, batch_size):
            print('\t  Batch %02d/%02d' % (idx / batch_size + 1, np.ceil(1. * num_out_channels / batch_size)))
            batch = patches_style_normed[..., idx:idx + batch_size]
            if self.sess:
                corr_center = self.conv.eval({self.conv_input: [self.content[0]], self.conv_filter: batch},
                                             session=self.sess)
                corr_top = self.conv.eval({self.conv_input: [self.content[1]], self.conv_filter: batch},
                                          session=self.sess)
                corr_bottom = self.conv.eval({self.conv_input: [self.content[2]], self.conv_filter: batch},
                                             session=self.sess)
            else:
                corr_center = self.conv.eval({self.conv_input: [self.content[0]], self.conv_filter: batch})
                corr_top = self.conv.eval({self.conv_input: [self.content[1]], self.conv_filter: batch})
                corr_bottom = self.conv.eval({self.conv_input: [self.content[2]], self.conv_filter: batch})
            corr_center = np.squeeze(corr_center)
            corr_top = np.squeeze(corr_top)
            corr_bottom = np.squeeze(corr_bottom)
            H = corr_center.shape[0]
            W = corr_center.shape[1]
            weight_center, weight_top, weight_bottom = self.get_weight_360(H, W)

            w_corr_center = weight_center * corr_center
            w_corr_top = weight_top * corr_top
            w_corr_bottom = weight_bottom * corr_bottom
            max_idx_tmp1 = np.argmax(w_corr_center, axis=-1) + idx
            max_val_tmp1 = np.max(w_corr_center, axis=-1)
            del corr_center, corr_top, corr_bottom, batch, w_corr_center
            if max_idx1 is None:
                max_idx1, max_val1 = max_idx_tmp1, max_val_tmp1
            else:
                indices1 = max_val_tmp1 > max_val1
                max_val1[indices1] = max_val_tmp1[indices1]
                max_idx1[indices1] = max_idx_tmp1[indices1]

            max_idx_tmp2 = np.argmax(w_corr_top, axis=-1) + idx
            max_val_tmp2 = np.max(w_corr_top, axis=-1)
            del w_corr_top
            if max_idx2 is None:
                max_idx2, max_val2 = max_idx_tmp2, max_val_tmp2
            else:
                indices2 = max_val_tmp2 > max_val2
                max_val2[indices2] = max_val_tmp2[indices2]
                max_idx2[indices2] = max_idx_tmp2[indices2]

            max_idx_tmp3 = np.argmax(w_corr_bottom, axis=-1) + idx
            max_val_tmp3 = np.max(w_corr_bottom, axis=-1)
            del w_corr_bottom
            if max_idx3 is None:
                max_idx3, max_val3 = max_idx_tmp3, max_val_tmp3
            else:
                indices3 = max_val_tmp3 > max_val3
                max_val3[indices3] = max_val_tmp3[indices3]
                max_idx3[indices3] = max_idx_tmp3[indices3]

        # mean_center = np.mean(max_val1)
        # mean_top = np.mean(max_val2)
        # mean_bottom = np.mean(max_val3)
        # final_corr_3_1 = (mean_center + mean_top + mean_bottom) / 3
        # final_corr += final_corr_3_1
        # final_corr = mean_center
        # print("center top bottom的最大值的平均分别是：", mean_center, mean_top, mean_bottom)
        # print("平均relu3_1", final_corr_3_1)
        # stitch other styles
        # content_index = 0
        del patches_style_normed
        patch_size, stride = self.patch_size, self.stride
        content_index = 0
        if other_styles:
            for style in other_styles:
                ratio = float(style[0].shape[0]) / self.style[0].shape[0]
                # print("condition[0].shape",condition[0].shape)
                # print("self.condition[0].shape",self.condition[0].shape)
                print("ratio:", ratio)
                assert int(ratio) == ratio
                ratio = int(ratio)
                self.patch_size = patch_size * ratio
                self.stride = stride * ratio
                patches_style = np.concatenate(list(map(self.style2patches, style)), axis=-1)
                norm = np.sqrt(np.sum(np.square(patches_style), axis=(0, 1, 2)))
                patches_style_normed = patches_style / norm
                # patches_center = np.concatenate(list(map(self.style2patches, other_centers[0])), axis=-1)
                corr_map_center = np.zeros((max_idx1.shape[0], max_idx1.shape[1]))
                corr_map_top = np.zeros((max_idx1.shape[0], max_idx1.shape[1]))
                corr_map_bottom = np.zeros((max_idx1.shape[0], max_idx1.shape[1]))

                for i in range(max_idx1.shape[0]):
                    for j in range(max_idx1.shape[1]):
                        corr_map_center[i, j] += np.sum(other_centers[content_index][0][i * ratio:i * ratio + self.patch_size, j * ratio:j * ratio + self.patch_size, :]
                                                        * patches_style_normed[:, :, :, max_idx1[i, j]])
                        corr_map_top[i, j] += np.sum(other_tops[content_index][0][i * ratio:i * ratio + self.patch_size,j * ratio:j * ratio + self.patch_size, :]
                                                        * patches_style_normed[:, :, :, max_idx2[i, j]])
                        corr_map_bottom[i, j] += np.sum(other_bottoms[content_index][0][i * ratio:i * ratio + self.patch_size,j * ratio:j * ratio + self.patch_size, :]
                                                        * patches_style_normed[:, :, :, max_idx3[i, j]])
                del norm, patches_style,patches_style_normed
                        # print("other_centers",other_centers[0][0][i * ratio:i * ratio + self.patch_size, j * ratio:j * ratio + self.patch_size,:])
                # print("other_centers shape", other_centers[content_index][0][i * ratio:i * ratio + self.patch_size, j * ratio:j * ratio + self.patch_size, :].shape)
                        # print("patches_condition",patches_condition[:, :, :, max_idx1[i, j]])
                # print("patches_condition shape", patches_style_normed[:, :, :, max_idx3[i, j]].shape)

                        # print(max_idx1[i, j])
                        # print(max_idx2[i, j])
                        # print(max_idx3[i, j])
                        # count_map是用来平均重叠像素的，相关性应该在平均之前算
                    # print("count_map",count_map[i * ratio:i * ratio + self.patch_size, j * ratio:j * ratio + self.patch_size])
                # print("count_map",count_map)
                # print("patches_condition",patches_condition.shape)
                # print("max_idx1", max_idx1.shape)
                # print("target_map",target_map.shape)
                # print("count_map", count_map)
                corr_center_weight = np.squeeze(weight_center) * corr_map_center
                corr_top_weight = np.squeeze(weight_top) * corr_map_top
                corr_bottom_weight = np.squeeze(weight_bottom) * corr_map_bottom
                mean_center_1_1 = np.mean(corr_center_weight)
                mean_top_1_1 = np.mean(corr_top_weight)
                mean_bottom_1_1 = np.mean(corr_bottom_weight)
                var_center_1_1 = np.var(corr_center_weight)
                # var_top_1_1 = np.var(corr_top_weight)
                # var_bottom_1_1 = np.var(corr_bottom_weight)
                # if(content_index==0):
                #     final_corr_1_1 = 0.0001 * (mean_center_1_1 + mean_top_1_1 + mean_bottom_1_1) / 3
                # if (content_index == 1):
                #     final_corr_1_1 = 0.001 * (mean_center_1_1 + mean_top_1_1 + mean_bottom_1_1) / 3
                final_corr_1_1 = (mean_center_1_1 + mean_top_1_1 + mean_bottom_1_1) / 3
                # for_write.append((mean_center_1_1 + mean_top_1_1 + mean_bottom_1_1) / 3)
                print("center top bottom的最大值的平均(1_1)分别是：", mean_center_1_1, mean_top_1_1, mean_bottom_1_1)
                print("平均relu_1_1:", final_corr_1_1)
                content_index += 1
                final_corr += final_corr_1_1
                final_var += var_center_1_1
                del corr_center_weight, corr_top_weight, corr_bottom_weight
                # content_index = content_index+1
                # final_corr += final_corr_1_1
            # self.write_relu(for_write)

        return final_corr,final_var
