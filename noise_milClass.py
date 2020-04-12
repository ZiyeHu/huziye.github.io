""" This file defines Meta Imitation Learning (MIL). """
from __future__ import division
import numpy as np
import random
import tensorflow as tf
import logging
import imageio
import gym
import math

from data_generator import DataGenerator
#
# from evaluation.eval_reach import evaluate_vision_reach
# from evaluation.eval_push import evaluate_push
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
LOGGER = logging.getLogger(__name__)
from tf_utils import *
from utils import Timer
from natsort import natsorted
import os
import time
import imageio
from utils import mkdir_p, load_scale_and_bias

## Dataset/method options
flags.DEFINE_string('experiment', 'sim_vision_reach', 'sim_vision_reach or sim_push')
flags.DEFINE_string('demo_file', 'mil_data/data/sim_vision_reach/',
                    'path to the directory where demo files that containing robot states and actions are stored')
flags.DEFINE_string('demo_gif_dir', 'mil_data/data/sim_vision_reach/', 'path to the videos of demonstrations')
flags.DEFINE_string('gif_prefix', 'color', 'prefix of the video directory for each task, e.g. object_0 for task 0')
flags.DEFINE_integer('im_width', 80,
                     'width of the images in the demo videos,  125 for sim_push, and 80 for sim_vision_reach')
flags.DEFINE_integer('im_height', 64,
                     'height of the images in the demo videos, 125 for sim_push, and 64 for sim_vision_reach')
flags.DEFINE_integer('num_channels', 3, 'number of channels of the images in the demo videos')
flags.DEFINE_integer('T', 50, 'time horizon of the demo videos, 50 for reach, 100 for push')
flags.DEFINE_bool('hsv', False, 'convert the image to HSV format')
flags.DEFINE_bool('use_noisy_demos', False, 'use noisy demonstrations or not (for domain shift)')
flags.DEFINE_string('noisy_demo_gif_dir', None, 'path to the videos of noisy demonstrations')
flags.DEFINE_string('noisy_demo_file', None,
                    'path to the directory where noisy demo files that containing robot states and actions are stored')
flags.DEFINE_bool('no_action', True, 'do not include actions in the demonstrations for inner update')
flags.DEFINE_bool('no_state', False, 'do not include states in the demonstrations during training')
flags.DEFINE_bool('no_final_eept', False, 'do not include final ee pos in the demonstrations for inner update')
flags.DEFINE_bool('zero_state', True,
                  'zero-out states (meta-learn state) in the demonstrations for inner update (used in the paper with video-only demos)')
flags.DEFINE_bool('two_arms', False, 'use two-arm structure when state is zeroed-out')
flags.DEFINE_integer('training_set_size', 750, 'size of the training set, 1500 for sim_reach, 693 for sim push, and \
                                                -1 for all data except those in validation set')
flags.DEFINE_integer('val_set_size', 150, 'size of the training set, 150 for sim_reach and 76 for sim push')

## Training options
flags.DEFINE_integer('metatrain_iterations', 30000,'number of metatraining iterations.')  # 30k for pushing, 50k for reaching and placing
flags.DEFINE_integer('meta_batch_size', 25,'number of tasks sampled per meta-update')  # 25 for reaching, 15 for pushing, 12 for placing
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('train_update_lr', 1e-3,
                   'step size alpha for inner gradient update.')  # 0.001 for reaching, 0.01 for pushing and placing
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')  # 5 for placing
flags.DEFINE_bool('clip', True, 'use gradient clipping for fast gradient')
flags.DEFINE_float('clip_max', 20.0, 'maximum clipping value for fast gradient')
flags.DEFINE_float('clip_min', -20.0, 'minimum clipping value for fast gradient')
flags.DEFINE_bool('fc_bt', True, 'use bias transformation for the first fc layer')
flags.DEFINE_bool('all_fc_bt', False, 'use bias transformation for all fc layers')
flags.DEFINE_bool('conv_bt', False, 'use bias transformation for the first conv layer, N/A for using pretraining')
flags.DEFINE_integer('bt_dim', 10, 'the dimension of bias transformation for FC layers')
flags.DEFINE_string('pretrain_weight_path', 'N/A', 'path to pretrained weights')
flags.DEFINE_bool('train_pretrain_conv1', False, 'whether to finetune the pretrained weights')
flags.DEFINE_bool('two_head', True, 'use two-head architecture')
flags.DEFINE_bool('learn_final_eept', False, 'learn an auxiliary loss for predicting final end-effector pose')
flags.DEFINE_bool('learn_final_eept_whole_traj', False, 'learn an auxiliary loss for predicting final end-effector pose \
                                                         by passing the whole trajectory of eepts (used for video-only models)')
flags.DEFINE_bool('stopgrad_final_eept', True,
                  'stop the gradient when concatenate the predicted final eept with the feature points')
flags.DEFINE_integer('final_eept_min', 6, 'first index of the final eept in the action array')
flags.DEFINE_integer('final_eept_max', 8, 'last index of the final eept in the action array')
flags.DEFINE_float('final_eept_loss_eps', 0.1, 'the coefficient of the auxiliary loss')
flags.DEFINE_float('act_loss_eps', 1.0, 'the coefficient of the action loss')
flags.DEFINE_float('loss_multiplier', 100.0,
                   'the constant multiplied with the loss value, 100 for reach and 50 for push')
flags.DEFINE_bool('use_l1_l2_loss', False, 'use a loss with combination of l1 and l2')
flags.DEFINE_float('l2_eps', 0.01, 'coeffcient of l2 loss')
flags.DEFINE_bool('shuffle_val', False, 'whether to choose the validation set via shuffling or not')

## Model options
flags.DEFINE_integer('random_seed', 0, 'random seed for training')
flags.DEFINE_bool('fp', True, 'use spatial soft-argmax or not')
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('dropout', False, 'use dropout for fc layers or not')
flags.DEFINE_float('keep_prob', 0.5, 'keep probability for dropout')
flags.DEFINE_integer('num_filters', 30,
                     'number of filters for conv nets -- 64 for placing, 16 for pushing, 40 for reaching.')
flags.DEFINE_integer('filter_size', 3, 'filter size for conv nets -- 3 for placing, 5 for pushing, 3 for reaching.')
flags.DEFINE_integer('num_conv_layers', 5, 'number of conv layers -- 5 for placing, 4 for pushing, 3 for reaching.')
flags.DEFINE_integer('num_strides', 3,
                     'number of conv layers with strided filters -- 3 for placing, 4 for pushing, 3 for reaching.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_integer('num_fc_layers', 3, 'number of fully-connected layers')
flags.DEFINE_integer('layer_size', 200, 'hidden dimension of fully-connected layers')
flags.DEFINE_bool('temporal_conv_2_head', True,
                  'whether or not to use temporal convolutions for the two-head architecture in video-only setting.')
flags.DEFINE_bool('temporal_conv_2_head_ee', False, 'whether or not to use temporal convolutions for the two-head architecture in video-only setting \
                for predicting the ee pose.')
flags.DEFINE_integer('temporal_filter_size', 10, 'filter size for temporal convolution')
flags.DEFINE_integer('temporal_num_filters', 32, 'number of filters for temporal convolution')
flags.DEFINE_integer('temporal_num_filters_ee', 64, 'number of filters for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temporal_num_layers', 3, 'number of layers for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temporal_num_layers_ee', 3, 'number of layers for temporal convolution for ee pose prediction')
flags.DEFINE_string('init', 'xavier', 'initializer for conv weights. Choose among random, xavier, and he')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
# flags.DEFINE_string('log_dirs', 'logs/sim_reach_temporal_conv_with_bicycle', 'directory for summaries and checkpoints.')
flags.DEFINE_string('log_dirs', 'logs/sim_reach_temporal_conv', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train',True , 'True to train, False to test.')
flags.DEFINE_integer('restore_iter', 0, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training \
                    (use if you want to test with a different number).')
flags.DEFINE_integer('test_update_batch_size', 1, 'number of demos used during test time')
flags.DEFINE_float('gpu_memory_fraction', 0.9, 'fraction of memory used in gpu')
flags.DEFINE_bool('record_gifs', True, 'record gifs during evaluation')
flags.DEFINE_bool('rl_update_batch_size', 1, 'number of demos used during rl time')
flags.DEFINE_bool('learn_bicycle', False, 'learning strategy')
flags.DEFINE_bool('compare_learn', True, 'learning strategy')
flags.DEFINE_integer('begin_restore_iter', 29100, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('end_restore_iter', 29999, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('gradients_num', 1, 'number of gradient update during training')




def generate_test_demos(data_generator):
    if not FLAGS.use_noisy_demos:
        n_folders = len(data_generator.demos.keys())
        demos = data_generator.demos
        # print('demos',demos)
    else:
        n_folders = len(data_generator.noisy_demos.keys())
        demos = data_generator.noisy_demos
    policy_demo_idx = [np.random.choice(n_demo, replace=False, size=FLAGS.test_update_batch_size) \
                       for n_demo in [demos[i]['demoX'].shape[0] for i in xrange(n_folders)]]

    compare_policy_demo_idx = [np.random.choice(n_demo, replace=False, size=FLAGS.test_update_batch_size) \
                       for n_demo in [demos[i]['demoX'].shape[0] for i in xrange(n_folders)]]

    print('policy_demo_idx',policy_demo_idx)
    print('compare_policy_demo_idx',compare_policy_demo_idx)

    selected_demoO, selected_demoX, selected_demoU = [], [], []
    goal_demoO, goal_demoX, goal_demoU = [], [], []
    compare_selected_demoO, compare_selected_demoX, compare_selected_demoU = [], [], []
    for i in xrange(n_folders):
        selected_cond = np.array(demos[i]['demoConditions'])[
            np.arange(len(demos[i]['demoConditions'])) == policy_demo_idx[i]]
        compare_selected_cond = np.array(demos[i]['demoConditions'])[
            np.arange(len(demos[i]['demoConditions'])) == compare_policy_demo_idx[i]]

        Xs, Us, Os = [], [], []
        goal_Xs, goal_Us, goal_Os = [], [], []
        compare_Xs, compare_Us, compare_Os = [], [], []
        for idx in selected_cond:
            if FLAGS.use_noisy_demos:
                demo_gif_dir = data_generator.noisy_demo_gif_dir
            else:
                demo_gif_dir = data_generator.demo_gif_dir
                # print('demo_gif_dir',demo_gif_dir)
            O = np.array(imageio.mimread(demo_gif_dir + data_generator.gif_prefix + '_%d/cond%d.samp0.gif' % (i, idx)))[
                :, :, :, :3]
            O = np.transpose(O, [0, 3, 2, 1])  # transpose to mujoco setting for images

            g=O[-1,:,:,:]
            g=np.tile(g,[O.shape[0],1,1,1])
            # print('g.shape',g.shape)
            g = g.reshape(FLAGS.T, -1) / 255.0  # normalize
            O = O.reshape(FLAGS.T, -1) / 255.0  # normalize
            Os.append(O)
            goal_Os.append(g)

        for idx in compare_selected_cond:
            if FLAGS.use_noisy_demos:
                demo_gif_dir = data_generator.noisy_demo_gif_dir
            else:
                demo_gif_dir = data_generator.demo_gif_dir
                # print('demo_gif_dir',demo_gif_dir)
            O = np.array(imageio.mimread(demo_gif_dir + data_generator.gif_prefix + '_%d/cond%d.samp0.gif' % (i, idx)))[
                :, :, :, :3]
            O = np.transpose(O, [0, 3, 2, 1])  # transpose to mujoco setting for images
            O = O.reshape(FLAGS.T, -1) / 255.0  # normalize
            compare_Os.append(O)


        Xs.append(demos[i]['demoX'][np.arange(demos[i]['demoX'].shape[0]) == policy_demo_idx[i]].squeeze())
        Us.append(demos[i]['demoU'][np.arange(demos[i]['demoU'].shape[0]) == policy_demo_idx[i]].squeeze())
        selected_demoO.append(np.array(Os))
        selected_demoX.append(np.array(Xs))
        selected_demoU.append(np.array(Us))
        goal_demoO.append(np.array(goal_Os))
        goal_demoX.append(np.array(Xs))
        goal_demoU.append(np.array(Us))

        compare_Xs.append(demos[i]['demoX'][np.arange(demos[i]['demoX'].shape[0]) == compare_policy_demo_idx[i]].squeeze())
        compare_Us.append(demos[i]['demoU'][np.arange(demos[i]['demoU'].shape[0]) == compare_policy_demo_idx[i]].squeeze())
        compare_selected_demoO.append(np.array(compare_Os))
        compare_selected_demoX.append(np.array(compare_Xs))
        compare_selected_demoU.append(np.array(compare_Us))

    print "Finished collecting demos for testing"
    selected_demo = dict(selected_demoX=selected_demoX, selected_demoU=selected_demoU, selected_demoO=selected_demoO)
    goal_demo = dict(selected_demoX=goal_demoX, selected_demoU=goal_demoU, selected_demoO=goal_demoO)
    data_generator.selected_demo = selected_demo
    data_generator.goal_demo = goal_demo

    compare_selected_demo = dict(selected_demoX=compare_selected_demoX, selected_demoU=compare_selected_demoU, selected_demoO=compare_selected_demoO)
    data_generator.compare_selected_demo = compare_selected_demo



def construct_fc_weights(dU, img_idx, state_idx, norm_type='layer_norm', dim_input=27, dim_output=7, network_config=None):
    """ Construct fc_weights for the network. """

    n_layers = network_config.get('n_layers', 4)
    dim_hidden = network_config.get('layer_size', [100] * (n_layers - 1))
    if type(dim_hidden) is not list:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    dim_hidden.append(dim_output)
    weights = {}
    in_shape = dim_input
    for i in xrange(n_layers):
        if FLAGS.two_arms and i == 0:
            if norm_type == 'selu':
                weights['w_%d_img' % i] = init_fc_weights_snn([in_shape - len(state_idx), dim_hidden[i]],
                                                              name='w_%d_img' % i)
                weights['w_%d_state' % i] = init_fc_weights_snn([len(state_idx), dim_hidden[i]], name='w_%d_state' % i)
            else:
                weights['w_%d_img' % i] = init_weights([in_shape - len(state_idx), dim_hidden[i]], name='w_%d_img' % i)
                weights['w_%d_state' % i] = init_weights([len(state_idx), dim_hidden[i]], name='w_%d_state' % i)
                weights['b_%d_state_two_arms' % i] = init_bias([dim_hidden[i]], name='b_%d_state_two_arms' % i)
            weights['b_%d_img' % i] = init_bias([dim_hidden[i]], name='b_%d_img' % i)
            weights['b_%d_state' % i] = init_bias([dim_hidden[i]], name='b_%d_state' % i)
            in_shape = dim_hidden[i]
            continue
        if i > 0 and FLAGS.all_fc_bt:
            in_shape += FLAGS.bt_dim
            weights['context_%d' % i] = init_bias([FLAGS.bt_dim], name='context_%d' % i)
        if norm_type == 'selu':
            weights['w_%d' % i] = init_fc_weights_snn([in_shape, dim_hidden[i]], name='w_%d' % i)
        else:
            weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
        weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
        if (i == n_layers - 1 or (i == 0 and FLAGS.zero_state and not FLAGS.two_arms)) and FLAGS.two_head:
            if i == n_layers - 1 and FLAGS.temporal_conv_2_head:
                temporal_kernel_size = FLAGS.temporal_filter_size
                temporal_num_filters = [FLAGS.temporal_num_filters] * FLAGS.temporal_num_layers
                temporal_num_filters[-1] = dim_output
                for j in xrange(len(temporal_num_filters)):
                    if j != len(temporal_num_filters) - 1:
                        weights['w_1d_conv_2_head_%d' % j] = init_weights(
                            [temporal_kernel_size, in_shape, temporal_num_filters[j]],
                            name='w_1d_conv_2_head_%d' % j)
                        weights['b_1d_conv_2_head_%d' % j] = init_bias([temporal_num_filters[j]],
                                                                       name='b_1d_conv_2_head_%d' % j)
                        in_shape = temporal_num_filters[j]
                    else:
                        weights['w_1d_conv_2_head_%d' % j] = init_weights([1, in_shape, temporal_num_filters[j]],
                                                                          name='w_1d_conv_2_head_%d' % j)
                        weights['b_1d_conv_2_head_%d' % j] = init_bias([temporal_num_filters[j]],
                                                                       name='b_1d_conv_2_head_%d' % j)
            else:
                print('in_shape, dim_hidden[i]', in_shape, dim_hidden[i])
                weights['w_%d_two_heads' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d_two_heads' % i)
                weights['b_%d_two_heads' % i] = init_bias([dim_hidden[i]], name='b_%d_two_heads' % i)
        in_shape = dim_hidden[i]
    return weights


# def construct_weights(self, dim_input=27, dim_output=7, network_config=None):
def construct_weights(dU, img_idx, state_idx, norm_type, network_config=None):
    """ Construct weights for the network. """
    dim_input = len(img_idx) + len(state_idx)
    dim_output = dU
    weights = {}
    num_filters = network_config['num_filters']
    strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
    filter_sizes = network_config.get('filter_size', [3] * len(strides))  # used to be 2
    if type(filter_sizes) is not list:
        filter_sizes = len(strides) * [filter_sizes]
    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    is_dilated = network_config.get('is_dilated', False)
    use_fp = FLAGS.fp
    pretrain = FLAGS.pretrain_weight_path != 'N/A'
    train_pretrain_conv1 = FLAGS.train_pretrain_conv1
    initialization = network_config.get('initialization', 'random')
    if pretrain:
        num_filters[0] = 64
    pretrain_weight_path = FLAGS.pretrain_weight_path
    n_conv_layers = len(num_filters)
    downsample_factor = 1
    for stride in strides:
        downsample_factor *= stride[1]
    if use_fp:
        conv_out_size = int(num_filters[-1] * 2)
    else:
        conv_out_size = int(np.ceil(im_width / (downsample_factor))) * int(np.ceil(im_height / (downsample_factor))) * \
                        num_filters[-1]

    # conv weights
    fan_in = num_channels
    if FLAGS.conv_bt:
        fan_in += num_channels
    if FLAGS.conv_bt:
        weights['img_context'] = safe_get('img_context',
                                          initializer=tf.zeros([im_height, im_width, num_channels], dtype=tf.float32))
        weights['img_context'] = tf.clip_by_value(weights['img_context'], 0., 1.)
    for i in xrange(n_conv_layers):
        if not pretrain or i != 0:
            if norm_type == 'selu':
                weights['wc%d' % (i + 1)] = init_conv_weights_snn(
                    [filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]],
                    name='wc%d' % (i + 1))  # 5x5 conv, 1 input, 32 outputs
            elif initialization == 'xavier':
                weights['wc%d' % (i + 1)] = init_conv_weights_xavier(
                    [filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]],
                    name='wc%d' % (i + 1))  # 5x5 conv, 1 input, 32 outputs
            elif initialization == 'random':
                weights['wc%d' % (i + 1)] = init_weights([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]],
                                                         name='wc%d' % (i + 1))  # 5x5 conv, 1 input, 32 outputs
            else:
                raise NotImplementedError
            weights['bc%d' % (i + 1)] = init_bias([num_filters[i]], name='bc%d' % (i + 1))
            fan_in = num_filters[i]
        else:
            import h5py

            assert num_filters[i] == 64
            vgg_filter_size = 3
            weights['wc%d' % (i + 1)] = safe_get('wc%d' % (i + 1),
                                                 [vgg_filter_size, vgg_filter_size, fan_in, num_filters[i]],
                                                 dtype=tf.float32, trainable=train_pretrain_conv1)
            weights['bc%d' % (i + 1)] = safe_get('bc%d' % (i + 1), [num_filters[i]], dtype=tf.float32,
                                                 trainable=train_pretrain_conv1)
            pretrain_weight = h5py.File(pretrain_weight_path, 'r')
            conv_weight = pretrain_weight['block1_conv%d' % (i + 1)]['block1_conv%d_W_1:0' % (i + 1)][...]
            conv_bias = pretrain_weight['block1_conv%d' % (i + 1)]['block1_conv%d_b_1:0' % (i + 1)][...]
            weights['wc%d' % (i + 1)].assign(conv_weight)
            weights['bc%d' % (i + 1)].assign(conv_bias)
            fan_in = conv_weight.shape[-1]

    # fc weights
    in_shape = conv_out_size
    if not FLAGS.no_state:
        in_shape += len(state_idx)
    if FLAGS.learn_final_eept:
        final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
        final_eept_in_shape = conv_out_size
        if FLAGS.fc_bt:
            weights['context_final_eept'] = safe_get('context_final_eept',
                                                     initializer=tf.zeros([FLAGS.bt_dim], dtype=tf.float32))
            final_eept_in_shape += FLAGS.bt_dim
        weights['w_ee'] = init_weights([final_eept_in_shape, len(final_eept_range)], name='w_ee')
        weights['b_ee'] = init_bias([len(final_eept_range)], name='b_ee')
        if FLAGS.two_head and FLAGS.no_final_eept:
            two_head_in_shape = final_eept_in_shape
            if FLAGS.temporal_conv_2_head_ee:
                temporal_kernel_size = FLAGS.temporal_filter_size
                temporal_num_filters = [FLAGS.temporal_num_filters_ee] * FLAGS.temporal_num_layers_ee
                temporal_num_filters[-1] = len(final_eept_range)
                for j in xrange(len(temporal_num_filters)):
                    if j != len(temporal_num_filters) - 1:
                        weights['w_1d_conv_2_head_ee_%d' % j] = init_weights(
                            [temporal_kernel_size, two_head_in_shape, temporal_num_filters[j]],
                            name='w_1d_conv_2_head_ee_%d' % j)
                        weights['b_1d_conv_2_head_ee_%d' % j] = init_bias([temporal_num_filters[j]],
                                                                          name='b_1d_conv_2_head_ee_%d' % j)
                        two_head_in_shape = temporal_num_filters[j]
                    else:
                        weights['w_1d_conv_2_head_ee_%d' % j] = init_weights(
                            [1, two_head_in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_ee_%d' % j)
                        weights['b_1d_conv_2_head_ee_%d' % j] = init_bias([temporal_num_filters[j]],
                                                                          name='b_1d_conv_2_head_ee_%d' % j)
            else:
                weights['w_ee_two_heads'] = init_weights([two_head_in_shape, len(final_eept_range)],
                                                         name='w_ee_two_heads')
                weights['b_ee_two_heads'] = init_bias([len(final_eept_range)], name='b_ee_two_heads')
        in_shape += (len(final_eept_range))
    if FLAGS.fc_bt:
        in_shape += FLAGS.bt_dim
    if FLAGS.fc_bt:
        weights['context'] = safe_get('context', initializer=tf.zeros([FLAGS.bt_dim], dtype=tf.float32))
    fc_weights = construct_fc_weights(dU, img_idx, state_idx, norm_type, in_shape, dim_output,
                                      network_config=network_config)
    conv_out_size_final = in_shape
    weights.update(fc_weights)
    # print('local_weights',weights)
    return weights, conv_out_size


def construct_image_input(nn_input, state_idx, img_idx, network_config=None):
    """ Preprocess images. """
    state_input = nn_input[:, 0:state_idx[-1] + 1]
    flat_image_input = nn_input[:, state_idx[-1] + 1:img_idx[-1] + 1]

    # image goes through 3 convnet layers
    num_filters = network_config['num_filters']

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(flat_image_input, [-1, num_channels, im_width, im_height])
    image_input = tf.transpose(image_input, perm=[0, 3, 2, 1])
    if FLAGS.pretrain_weight_path != 'N/A':
        image_input = image_input * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
        # 'RGB'->'BGR'
        image_input = image_input[:, :, :, ::-1]
    return image_input, flat_image_input, state_input


def fc_forward(fc_input, weights, state_input=None, meta_testing=False, is_training=True, testing=False, norm_type='layer_norm',
               network_config=None, activation_fn=tf.nn.relu):
    n_layers = network_config.get('n_layers', 4)
    use_dropout = FLAGS.dropout
    prob = FLAGS.keep_prob
    fc_output = tf.add(fc_input, 0)
    use_selu = (norm_type == 'selu')
    if state_input is not None and not FLAGS.two_arms:
        fc_output = tf.concat(axis=1, values=[fc_output, state_input])
    for i in xrange(n_layers):
        if i > 0 and FLAGS.all_fc_bt:
            context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(fc_output)), range(FLAGS.bt_dim)))
            context += weights['context_%d' % i]
            fc_output = tf.concat(axis=1, values=[fc_output, context])
        if (i == n_layers - 1 or (
                i == 0 and FLAGS.zero_state and not FLAGS.two_arms)) and FLAGS.two_head and not meta_testing:
            if i == n_layers - 1 and FLAGS.temporal_conv_2_head:
                fc_output = tf.reshape(fc_output, [-1, FLAGS.T, fc_output.get_shape().dims[-1].value])
                strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
                temporal_num_layers = FLAGS.temporal_num_layers
                n_conv_layers = len(strides)
                if FLAGS.temporal_conv_2_head_ee:
                    n_conv_layers += FLAGS.temporal_num_layers_ee
                for j in xrange(temporal_num_layers):
                    if j != temporal_num_layers - 1:
                        fc_output = norm(conv1d(img=fc_output, w=weights['w_1d_conv_2_head_%d' % j],
                                                b=weights['b_1d_conv_2_head_%d' % j]), \
                                         norm_type=norm_type, id=n_conv_layers + j, is_training=is_training,
                                         activation_fn=activation_fn)
                    else:
                        fc_output = conv1d(img=fc_output, w=weights['w_1d_conv_2_head_%d' % j],
                                           b=weights['b_1d_conv_2_head_%d' % j])
            else:
                # print('weights are',weights['w_%d_two_heads' % i],weights['b_%d_two_heads' % i])
                fc_output = tf.matmul(fc_output, weights['w_%d_two_heads' % i]) + weights['b_%d_two_heads' % i]
        elif i == 0 and FLAGS.two_arms:
            assert state_input is not None
            if FLAGS.two_arms:
                state_part = weights['b_%d_state_two_arms' % i]
            else:
                state_part = tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
            if not meta_testing:
                fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i] + state_part
            else:
                fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i] + \
                            tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
        else:
            fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
        if i != n_layers - 1:
            fc_output = activation_fn(fc_output)
            # only use dropout for post-update
            if use_dropout:
                fc_output = dropout(fc_output, keep_prob=prob, is_training=is_training, name='dropout_fc_%d' % i,
                                    selu=use_selu)
    return fc_output


def forward(conv_out_size, image_input, state_input, weights, meta_testing=False, is_training=True, testing=False,
            norm_type='layer_norm', network_config=None, activation_fn=tf.nn.relu):
    """ Perform the forward pass. """
    if FLAGS.fc_bt:
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        flatten_image = tf.reshape(image_input, [-1, im_height * im_width * num_channels])
        context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(FLAGS.bt_dim)))
        context += weights['context']
        if FLAGS.learn_final_eept:
            context_final_eept = tf.transpose(
                tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(FLAGS.bt_dim)))
            context_final_eept += weights['context_final_eept']

    decay = network_config.get('decay', 0.9)
    strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
    downsample_factor = strides[0][1]
    n_strides = len(strides)
    n_conv_layers = len(strides)
    use_dropout = FLAGS.dropout
    prob = FLAGS.keep_prob
    is_dilated = network_config.get('is_dilated', False)
    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    conv_layer = image_input
    if FLAGS.conv_bt:
        img_context = tf.zeros_like(conv_layer)
        img_context += weights['img_context']
        conv_layer = tf.concat(axis=3, values=[conv_layer, img_context])
    for i in xrange(n_conv_layers):
        if not use_dropout:
            conv_layer = norm(
                conv2d(img=conv_layer, w=weights['wc%d' % (i + 1)], b=weights['bc%d' % (i + 1)], strides=strides[i],
                       is_dilated=is_dilated), \
                norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=activation_fn)
        else:
            conv_layer = dropout(norm(
                conv2d(img=conv_layer, w=weights['wc%d' % (i + 1)], b=weights['bc%d' % (i + 1)], strides=strides[i],
                       is_dilated=is_dilated), \
                norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=activation_fn),
                                 keep_prob=prob, is_training=is_training, name='dropout_%d' % (i + 1))
    if FLAGS.fp:
        _, num_rows, num_cols, num_fp = conv_layer.get_shape()
        if is_dilated:
            num_rows = int(np.ceil(im_width / (downsample_factor ** n_strides)))
            num_cols = int(np.ceil(im_height / (downsample_factor ** n_strides)))
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = tf.convert_to_tensor(x_map)
        y_map = tf.convert_to_tensor(y_map)

        x_map = tf.reshape(x_map, [num_rows * num_cols])
        y_map = tf.reshape(y_map, [num_rows * num_cols])

        # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
        features = tf.reshape(tf.transpose(conv_layer, [0, 3, 1, 2]),
                              [-1, num_rows * num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

        conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp * 2])
    else:
        conv_out_flat = tf.reshape(conv_layer, [-1, conv_out_size])
    fc_input = tf.add(conv_out_flat, 0)
    if FLAGS.learn_final_eept:
        final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
        if testing:
            T = 1
        else:
            T = FLAGS.T
        conv_out_flat = tf.reshape(conv_out_flat, [-1, T, conv_out_size])
        conv_size = conv_out_size
        if FLAGS.fc_bt:
            context_dim = FLAGS.bt_dim
            conv_out_flat = tf.concat(axis=2,
                                      values=[conv_out_flat, tf.reshape(context_final_eept, [-1, T, context_dim])])
            conv_size += context_dim
        # only predict the final eept using the initial image
        final_ee_inp = tf.reshape(conv_out_flat, [-1, conv_size])
        # use video for preupdate only if no_final_eept
        if (not FLAGS.learn_final_eept_whole_traj) or meta_testing:
            final_ee_inp = conv_out_flat[:, 0, :]
        if FLAGS.two_head and not meta_testing and FLAGS.no_final_eept:
            if network_config.get('temporal_conv_2_head_ee', False):
                final_eept_pred = final_ee_inp
                final_eept_pred = tf.reshape(final_eept_pred, [-1, T, final_eept_pred.get_shape().dims[-1].value])
                task_label_pred = None
                temporal_num_layers = FLAGS.temporal_num_layers_ee
                for j in xrange(temporal_num_layers):
                    if j != temporal_num_layers - 1:
                        final_eept_pred = norm(conv1d(img=final_eept_pred, w=weights['w_1d_conv_2_head_ee_%d' % j],
                                                      b=weights['b_1d_conv_2_head_ee_%d' % j]), \
                                               norm_type=norm_type, id=n_conv_layers + j, is_training=is_training,
                                               activation_fn=activation_fn)
                    else:
                        final_eept_pred = conv1d(img=final_eept_pred, w=weights['w_1d_conv_2_head_ee_%d' % j],
                                                 b=weights['b_1d_conv_2_head_ee_%d' % j])
                final_eept_pred = tf.reshape(final_eept_pred, [-1, len(final_eept_range)])
            else:
                final_eept_pred = tf.matmul(final_ee_inp, weights['w_ee_two_heads']) + weights['b_ee_two_heads']
        else:
            final_eept_pred = tf.matmul(final_ee_inp, weights['w_ee']) + weights['b_ee']
        if (not FLAGS.learn_final_eept_whole_traj) or meta_testing:
            final_eept_pred = tf.reshape(tf.tile(tf.reshape(final_eept_pred, [-1]), [T]), [-1, len(final_eept_range)])
            final_eept_concat = tf.identity(final_eept_pred)
        else:
            # Assume tbs == 1
            # Only provide the FC layers with final_eept_pred at first time step
            final_eept_concat = final_eept_pred[0]
            final_eept_concat = tf.reshape(tf.tile(tf.reshape(final_eept_concat, [-1]), [T]),
                                           [-1, len(final_eept_range)])
        if FLAGS.stopgrad_final_eept:
            final_eept_concat = tf.stop_gradient(final_eept_concat)
        fc_input = tf.concat(axis=1, values=[fc_input, final_eept_concat])
    else:
        final_eept_pred = None
    if FLAGS.fc_bt:
        fc_input = tf.concat(axis=1, values=[fc_input, context])
    return fc_forward(fc_input, weights, state_input=state_input, meta_testing=meta_testing, is_training=is_training,
                      testing=testing, norm_type=norm_type, network_config=network_config), final_eept_pred


def main():
    tf.set_random_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)

    graph = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=graph, config=tf_config)
    network_config = {
        'num_filters': [FLAGS.num_filters] * FLAGS.num_conv_layers,
        'strides': [[1, 2, 2, 1]] * FLAGS.num_strides + [[1, 1, 1, 1]] * (FLAGS.num_conv_layers - FLAGS.num_strides),
        'filter_size': FLAGS.filter_size,
        'image_width': FLAGS.im_width,
        'image_height': FLAGS.im_height,
        'image_channels': FLAGS.num_channels,
        'n_layers': FLAGS.num_fc_layers,
        'layer_size': FLAGS.layer_size,
        'initialization': FLAGS.init,
    }
    data_generator = DataGenerator()
    state_idx = data_generator.state_idx
    img_idx = range(len(state_idx), len(state_idx) + FLAGS.im_height * FLAGS.im_width * FLAGS.num_channels)
    with graph.as_default():
        # TODO: figure out how to save summaries and checkpoints
        exp_string = FLAGS.experiment + '.' + FLAGS.init + '_init.' + str(FLAGS.num_conv_layers) + '_conv' + '.' + str(
            FLAGS.num_strides) + '_strides' + '.' + str(FLAGS.num_filters) + '_filters' + \
                     '.' + str(FLAGS.num_fc_layers) + '_fc' + '.' + str(FLAGS.layer_size) + '_dim' + '.bt_dim_' + str(
            FLAGS.bt_dim) + '.mbs_' + str(FLAGS.meta_batch_size) + \
                     '.ubs_' + str(FLAGS.update_batch_size) + '.numstep_' + str(FLAGS.num_updates) + '.updatelr_' + str(
            FLAGS.train_update_lr)

        if FLAGS.clip:
            exp_string += '.clip_' + str(int(FLAGS.clip_max))
        if FLAGS.conv_bt:
            exp_string += '.conv_bt'
        if FLAGS.all_fc_bt:
            exp_string += '.all_fc_bt'
        if FLAGS.fp:
            exp_string += '.fp'
        if FLAGS.learn_final_eept:
            exp_string += '.learn_ee_pos'
        if FLAGS.no_action:
            exp_string += '.no_action'
        if FLAGS.zero_state:
            exp_string += '.zero_state'
        if FLAGS.two_head:
            exp_string += '.two_heads'
        if FLAGS.two_arms:
            exp_string += '.two_arms'
        if FLAGS.temporal_conv_2_head:
            exp_string += '.1d_conv_act_' + str(FLAGS.temporal_num_layers) + '_' + str(FLAGS.temporal_num_filters)
            if FLAGS.temporal_conv_2_head_ee:
                exp_string += '_ee_' + str(FLAGS.temporal_num_layers_ee) + '_' + str(FLAGS.temporal_num_filters_ee)
            exp_string += '_' + str(FLAGS.temporal_filter_size) + 'x1_filters'
        if FLAGS.training_set_size != -1:
            exp_string += '.' + str(FLAGS.training_set_size) + '_trials'

        log_dirs = FLAGS.log_dirs + '/' + exp_string

        # construct network
        # train_image_tensors = data_generator.make_batch_tensor(network_config, restore_iter=FLAGS.restore_iter)
        # inputa = train_image_tensors[:, :FLAGS.update_batch_size * FLAGS.T, :]
        # inputb = train_image_tensors[:, FLAGS.update_batch_size * FLAGS.T:, :]
        # train_input_tensors = {'inputa': inputa, 'inputb': inputb}
        # print('prepare inputa and inputb',inputa , inputb)

        # if train_input_tensors is None:
        #     print('train_input_tensors are None')
        #     obsa = tf.placeholder(tf.float32, name='obsa')  # meta_batch_size x update_batch_size x dim_input
        #     obsb = tf.placeholder(tf.float32, name='obsb')
        # else:
        #     print('train_input_tensors are not none')
        #     obsa = train_input_tensors['inputa']  # meta_batch_size x update_batch_size x dim_input
        #     obsb = train_input_tensors['inputb']

        global_obsa = tf.placeholder(tf.float32, name='obsa')  # meta_batch_size x update_batch_size x dim_input
        global_obsb = tf.placeholder(tf.float32, name='obsb')
        global_compare_obs = tf.placeholder(tf.float32, name='compare_obs')

        global_statea = tf.placeholder(tf.float32, name='statea')
        global_stateb = tf.placeholder(tf.float32, name='stateb')
        global_compare_state = tf.placeholder(tf.float32, name='compare_state')

        global_actiona = tf.placeholder(tf.float32, name='actiona')
        global_actionb = tf.placeholder(tf.float32, name='actionb')
        global_compare_action = tf.placeholder(tf.float32, name='compare_action')

        global_inputa = tf.concat(axis=2, values=[global_statea, global_obsa])
        global_inputb = tf.concat(axis=2, values=[global_stateb, global_obsb])
        global_compare_input = tf.concat(axis=2, values=[global_compare_state, global_compare_obs])

        global_disturb = tf.placeholder(shape=[1], dtype=tf.float32)
        global_precision = 0.00001

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            weights, conv_out_size = construct_weights(data_generator._dU, img_idx, state_idx, FLAGS.norm,network_config)
            # if FLAGS.compare_learn:
            #     with tf.variable_scope('compare_learn_model', reuse=tf.AUTO_REUSE) as compare_scope:
            #         compare_learn_weights, conv_out_size = construct_weights(data_generator._dU, img_idx, state_idx, FLAGS.norm,network_config)
            print('conv_out_size', conv_out_size)
            print('weights', weights)
            num_updates = FLAGS.num_updates
            dim_input = len(img_idx) + len(state_idx)
            dim_output = data_generator._dU
            prefix = 'Training'
            sorted_weight_keys = natsorted(weights.keys())
            step_size = FLAGS.train_update_lr
            loss_multiplier = FLAGS.loss_multiplier
            final_eept_loss_eps = FLAGS.final_eept_loss_eps
            act_loss_eps = FLAGS.act_loss_eps
            use_whole_traj = FLAGS.learn_final_eept_whole_traj
            num_updates = FLAGS.num_updates
            norm_type = FLAGS.norm

            def batch_metalearn(inp):
                print('batch_metalearn:weights',weights)
                # print('batch_metalearn:compare_learn_weights',compare_learn_weights)
                inputa, inputb,compare_input, actiona, actionb,compare_action, disturb = inp
                inputa = tf.reshape(inputa, [-1, dim_input])
                inputb = tf.reshape(inputb, [-1, dim_input])
                compare_input = tf.reshape(compare_input, [-1, dim_input])
                actiona = tf.reshape(actiona, [-1, dim_output])
                actionb = tf.reshape(actionb, [-1, dim_output])
                compare_action = tf.reshape(compare_action, [-1, dim_output])
                # gradients_summ = []
                testing = 'Testing' in prefix

                final_eepta, final_eeptb = None, None
                if FLAGS.learn_final_eept:
                    final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
                    final_eepta = actiona[:, final_eept_range[0]:final_eept_range[-1] + 1]
                    final_eeptb = actionb[:, final_eept_range[0]:final_eept_range[-1] + 1]
                    compare_final_eept = compare_action[:, final_eept_range[0]:final_eept_range[-1] + 1]
                    actiona = actiona[:, :final_eept_range[0]]
                    actionb = actionb[:, :final_eept_range[0]]
                    compare_action = compare_action[:, :final_eept_range[0]]
                    if FLAGS.no_final_eept:
                        final_eepta = tf.zeros_like(final_eepta)
                        compare_final_eept = tf.zeros_like(compare_final_eept)

                # if FLAGS.no_action:
                #     actiona = tf.zeros_like(actiona)
                #     compare_action = tf.zeros_like(compare_action)

                local_outputbs, local_lossesb, final_eept_lossesb = [], [], []
                # Assume fixed data for each update
                actionas = [actiona] * num_updates

                # Convert to image dims
                compare_input, _, compare_state_input = construct_image_input(compare_input, state_idx, img_idx,network_config=network_config)
                inputa, _, state_inputa = construct_image_input(inputa, state_idx, img_idx,network_config=network_config)
                inputb, flat_img_inputb, state_inputb = construct_image_input(inputb, state_idx, img_idx,network_config=network_config)
                inputas = [inputa] * num_updates
                inputbs = [inputb] * num_updates

                # if FLAGS.zero_state:
                #     state_inputa = tf.zeros_like(state_inputa)
                #     compare_state_input = tf.zeros_like(compare_state_input)

                state_inputas = [state_inputa] * num_updates
                if FLAGS.no_state:
                    state_inputa = None
                    compare_state_input = None

                if FLAGS.learn_final_eept:
                    final_eeptas = [final_eepta] * num_updates

                # Noise Pre-update
                disturb_weights = dict(zip(weights.keys(), [weights[key] - global_precision * disturb for key in weights.keys()]))
                if 'Training' in prefix:
                    disturb_local_outputa, _ = forward(conv_out_size, inputa, tf.zeros_like(state_inputa), disturb_weights, network_config=network_config)
                else:
                    disturb_local_outputa, _ = forward(conv_out_size, inputa, tf.zeros_like(state_inputa), disturb_weights,  is_training=False, network_config=network_config)
                disturb_lossa = act_loss_eps * euclidean_loss_layer(disturb_local_outputa, tf.zeros_like(disturb_local_outputa),multiplier=loss_multiplier,  use_l1=FLAGS.use_l1_l2_loss)
                disturb_grad = tf.gradients(disturb_lossa, disturb_weights.values())
                disturb_gradients = dict(zip(disturb_weights.keys(), disturb_grad))
                for key in disturb_gradients.keys():
                    if disturb_gradients[key] is None:
                        disturb_gradients[key] = tf.zeros_like(disturb_weights[key])
                if FLAGS.stop_grad:
                    disturb_gradients = {key: tf.stop_gradient(disturb_gradients[key]) for key in disturb_gradients.keys()}
                if FLAGS.clip:
                    clip_min = FLAGS.clip_min
                    clip_max = FLAGS.clip_max
                    for key in disturb_gradients.keys():
                        disturb_gradients[key] = tf.clip_by_value(disturb_gradients[key], clip_min, clip_max)
                if FLAGS.pretrain_weight_path != 'N/A':
                    disturb_gradients['wc1'] = tf.zeros_like(disturb_gradients['wc1'])
                    disturb_gradients['bc1'] = tf.zeros_like(disturb_gradients['bc1'])
                print('disturb_gradients',disturb_gradients)
                fast_weights = dict(zip(weights.keys(), [weights[key] - step_size * disturb_gradients[key] for key in weights.keys()]))


                # Pre-update
                if 'Training' in prefix:
                    local_outputa, final_eept_preda = forward(conv_out_size, inputa, tf.zeros_like(state_inputa),fast_weights, network_config=network_config)
                else:
                    local_outputa, final_eept_preda = forward(conv_out_size, inputa, tf.zeros_like(state_inputa), fast_weights, is_training=False, network_config=network_config)
                local_lossa = act_loss_eps * euclidean_loss_layer(local_outputa, tf.zeros_like(local_outputa),multiplier=loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
                # Compute fast gradients
                grads = tf.gradients(local_lossa, fast_weights.values())
                # print('grads',grads)
                gradients = dict(zip(fast_weights.keys(), grads))
                # print('gradients',gradients)
                # make fast gradient zero for fast_weights with gradient None
                for key in gradients.keys():
                    if gradients[key] is None:
                        gradients[key] = tf.zeros_like(fast_weights[key])
                if FLAGS.stop_grad:
                    gradients = {key: tf.stop_gradient(gradients[key]) for key in gradients.keys()}
                if FLAGS.clip:
                    clip_min = FLAGS.clip_min
                    clip_max = FLAGS.clip_max
                    for key in gradients.keys():
                        gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                if FLAGS.pretrain_weight_path != 'N/A':
                    gradients['wc1'] = tf.zeros_like(gradients['wc1'])
                    gradients['bc1'] = tf.zeros_like(gradients['bc1'])
                # gradients_summ.append([gradients[key] for key in sorted_weight_keys])
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - step_size * gradients[key] for key in fast_weights.keys()]))
                print('FLAGS.gradients_num',FLAGS.gradients_num)
                # for gradients_num in range(1,FLAGS.gradients_num):
                #     grads = tf.gradients(local_lossa, gradients.values())
                #     # print('grads', grads)
                #     gradients = dict(zip(fast_weights.keys(), grads))
                #     # make fast gradient zero for weights with gradient None
                #     for key in gradients.keys():
                #         if gradients[key] is None:
                #             gradients[key] = tf.zeros_like(fast_weights[key])
                #     if FLAGS.stop_grad:
                #         gradients = {key: tf.stop_gradient(gradients[key]) for key in gradients.keys()}
                #     if FLAGS.clip:
                #         clip_min = FLAGS.clip_min
                #         clip_max = FLAGS.clip_max
                #         for key in gradients.keys():
                #             gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                #     if FLAGS.pretrain_weight_path != 'N/A':
                #         gradients['wc1'] = tf.zeros_like(gradients['wc1'])
                #         gradients['bc1'] = tf.zeros_like(gradients['bc1'])
                #     # gradients_summ.append([gradients[key] for key in sorted_weight_keys])
                #     gradients_step_size=step_size/math.factorial(gradients_num+1)
                #     print('gradients_num',gradients_num,'gradients_step_size',gradients_step_size)
                #     fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - gradients_step_size *  gradients[key] for key in fast_weights.keys()]))


                # Post-update
                if 'Training' in prefix:
                    outputb, final_eept_predb  = forward(conv_out_size, compare_input, compare_state_input, fast_weights,meta_testing=True, network_config=network_config)
                else:
                    outputb, final_eept_predb  = forward(conv_out_size, compare_input, compare_state_input, fast_weights,meta_testing=True, is_training=False, testing=testing,network_config=network_config)



                right_lossb = left_lossb = act_loss_eps * euclidean_loss_layer(outputb, tf.zeros_like(outputb), multiplier=loss_multiplier,use_l1=FLAGS.use_l1_l2_loss)

                local_lossb = act_loss_eps * euclidean_loss_layer(outputb, compare_action, multiplier=loss_multiplier,use_l1=FLAGS.use_l1_l2_loss)
                local_outputbs.append(outputb)
                if FLAGS.learn_final_eept:
                    final_eept_lossb = euclidean_loss_layer(final_eept_predb, final_eeptb, multiplier=loss_multiplier,use_l1=FLAGS.use_l1_l2_loss)
                else:
                    final_eept_lossb = tf.constant(0.0)


                if FLAGS.learn_final_eept:
                    local_lossb += final_eept_loss_eps * final_eept_lossb
                if use_whole_traj:
                    # assume tbs == 1
                    final_eept_lossb = euclidean_loss_layer(final_eept_predb[0], final_eeptb[0],
                                                            multiplier=loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
                final_eept_lossesb.append(final_eept_lossb)
                local_lossesb.append(local_lossb)


                local_fn_output = [local_outputa, local_outputbs, local_outputbs[-1], local_lossa, local_lossesb,
                                   final_eept_lossesb, flat_img_inputb,left_lossb,right_lossb]
                return local_fn_output


        # if norm_type:
        #     # initialize batch norm vars.
        #     unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0]))

        # out_dtype = [tf.float32, [tf.float32] * FLAGS.num_updates, tf.float32, tf.float32, [tf.float32] * FLAGS.num_updates,
        #              [tf.float32] * FLAGS.num_updates, tf.float32,
        #              [[tf.float32] * len(weights.keys())] * FLAGS.num_updates]
        out_dtype = [tf.float32, [tf.float32] * FLAGS.num_updates, tf.float32, tf.float32,[tf.float32] * FLAGS.num_updates,
                     [tf.float32] * FLAGS.num_updates, tf.float32,tf.float32,tf.float32]
        result = tf.map_fn(batch_metalearn, elems=(global_inputa, global_inputb,global_compare_input, global_actiona, global_actionb,global_compare_action,global_disturb), dtype=out_dtype)

        # outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, flat_img_inputb, gradients = result
        outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, flat_img_inputb,left_lossb,right_lossb = result
        # trainable_vars = tf.trainable_variables()
        total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        total_left_lossb = tf.reduce_sum(left_lossb) / tf.to_float(FLAGS.meta_batch_size)
        total_right_lossb = tf.reduce_sum(right_lossb) / tf.to_float(FLAGS.meta_batch_size)

        total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(FLAGS.num_updates)]
        total_final_eept_losses2 = [tf.reduce_sum(final_eept_lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(FLAGS.num_updates)]

        select_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='select_model')
        train_left_op = tf.train.AdamOptimizer(FLAGS.meta_lr).minimize(total_left_lossb)
        train_right_op = tf.train.AdamOptimizer(FLAGS.meta_lr).minimize(total_right_lossb)
        # self_learn_train_op = tf.train.AdamOptimizer(1e-6).minimize(total_right_lossb)
        # train_op = tf.train.AdamOptimizer(FLAGS.meta_lr).minimize(total_losses2[FLAGS.num_updates - 1],var_list=select_scope)
        self_learn_train_op = tf.train.AdamOptimizer(1e-6).minimize(total_losses2[FLAGS.num_updates - 1])
        train_op = tf.train.AdamOptimizer(FLAGS.meta_lr).minimize(total_losses2[FLAGS.num_updates - 1])


        # compare_learn_train_op = tf.train.AdamOptimizer(1e-6).minimize(total_losses2[FLAGS.num_updates - 1])

        print('select_scope',select_scope)

        test_act_op = test_output
        image_op = flat_img_inputb
        # saver = tf.train.Saver(max_to_keep=10)
        # Initialize variables.
        init_op = tf.global_variables_initializer()
        sess.run(init_op, feed_dict=None)

        saver = tf.train.Saver(max_to_keep=10)

        # Start queue runners (used for loading videos on the fly)
        tf.train.start_queue_runners(sess=sess)

        def train(graph, saver, sess, data_generator, log_dirs, restore_itr=0):
            """
            Train the
            """
            PRINT_INTERVAL = 100
            TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
            SUMMARY_INTERVAL = 100
            SAVE_INTERVAL = 100
            TOTAL_ITERS = FLAGS.metatrain_iterations
            prelosses, postlosses = [], []
            save_dir = log_dirs + '/model'
            train_writer = tf.summary.FileWriter(log_dirs, graph)
            # batch_zero_reach_demos_states = np.zeros([int(FLAGS.meta_batch_size), FLAGS.T, 10])
            # batch_zero_reach_demos_actions = np.zeros([int(FLAGS.meta_batch_size), FLAGS.T, 2])

            # actual training.
            if restore_itr == 0:
                training_range = range(TOTAL_ITERS)
            else:
                training_range = range(restore_itr + 1, TOTAL_ITERS)
            for itr in training_range:
                # train_image_datas = data_generator.make_batch_data(network_config, restore_iter=itr)
                train_image_datas,train_compare_image_datas = data_generator.make_compare_batch_data(network_config, restore_iter=itr)
                # train_image_datas, train_compare_image_datas = data_generator.make_goal_batch_data(network_config,restore_iter=itr)
                obsa_input = train_image_datas[:, :FLAGS.update_batch_size * FLAGS.T, :]
                obsb_input = train_image_datas[:, FLAGS.update_batch_size * FLAGS.T:, :]
                compare_obsa_input = train_compare_image_datas[:, :FLAGS.update_batch_size * FLAGS.T, :]
                compare_obsb_input = train_compare_image_datas[:, FLAGS.update_batch_size * FLAGS.T:, :]

                state, tgt_mu = data_generator.generate_data_batch(itr)
                # state, tgt_mu,compare_state, compare_tgt_mu = data_generator.generate_compare_data_batch(itr)
                statea_input = state[:, :FLAGS.update_batch_size * FLAGS.T, :]
                stateb_input = state[:, FLAGS.update_batch_size * FLAGS.T:, :]
                actiona_input = tgt_mu[:, :FLAGS.update_batch_size * FLAGS.T, :]
                actionb_input = tgt_mu[:, FLAGS.update_batch_size * FLAGS.T:, :]

                # compare_statea_input = compare_state[:, :FLAGS.update_batch_size * FLAGS.T, :]
                # compare_stateb_input = compare_state[:, FLAGS.update_batch_size * FLAGS.T:, :]
                # compare_actiona_input = compare_tgt_mu[:, :FLAGS.update_batch_size * FLAGS.T, :]
                # compare_actionb_input = compare_tgt_mu[:, FLAGS.update_batch_size * FLAGS.T:, :]

                compare_statea_input = statea_input
                compare_stateb_input = stateb_input
                compare_actiona_input = actiona_input
                compare_actionb_input = actionb_input

                # print('obsa_input',obsa_input.shape)

                # feed_dict = {global_obsa: obsa_input,
                #              global_obsb: compare_obsa_input,
                #              global_compare_obs: obsb_input,
                #              global_statea: statea_input,
                #              global_stateb: compare_statea_input,
                #              global_compare_state: stateb_input,
                #              global_actiona: actiona_input,
                #              global_actionb: compare_actiona_input,
                #              global_compare_action: actionb_input}

                # feed_dict = {global_obsa: obsa_input,
                #              global_obsb: obsa_input,
                #              global_compare_obs: obsa_input,
                #              global_statea: statea_input,
                #              global_stateb: statea_input,
                #              global_compare_state: statea_input,
                #              global_actiona: actiona_input,
                #              global_actionb: actiona_input,
                #              global_compare_action: actiona_input}

                feed_dict = {global_obsa: obsa_input,
                             global_obsb: obsa_input,
                             global_compare_obs: obsb_input,
                             global_statea: statea_input,
                             global_stateb: statea_input,
                             global_compare_state: stateb_input,
                             global_actiona: actiona_input,
                             global_actionb: actiona_input,
                             global_compare_action: actionb_input,
                             global_disturb: np.random.normal(0,1.0,1)}


                input_tensors = [train_op,
                                 total_losses2]
                results = sess.run(input_tensors, feed_dict=feed_dict)
                print '(Phase 1)Iteration %d:  average right_postloss is %.2f '  % \
                      (itr, np.mean(results[-1]))

                with open('logs/sim_reach_temporal_conv/traing_loss.txt', 'a') as f:
                    f.write("%d %f \n"%(itr,np.mean(results[-1])))

                # feed_dict = {global_obsa: compare_obsa_input,
                #              global_obsb: obsa_input,
                #              global_compare_obs: obsb_input,
                #              global_statea: compare_statea_input,
                #              global_stateb: statea_input,
                #              global_compare_state: stateb_input,
                #              global_actiona: compare_actiona_input,
                #              global_actionb: actiona_input,
                #              global_compare_action: actionb_input}

                # input_tensors = [train_op,
                #                  total_losses2]

                # feed_dict = {global_obsa: obsa_input,
                #              global_obsb: obsa_input,
                #              global_compare_obs: obsb_input,
                #              global_statea: statea_input,
                #              global_stateb: statea_input,
                #              global_compare_state: stateb_input,
                #              global_actiona: actiona_input,
                #              global_actionb: actiona_input,
                #              global_compare_action: actionb_input}
                # results2 = sess.run(input_tensors, feed_dict=feed_dict)
                # print '(Phase 2)Iteration %d:  average right_postloss is %.2f' % \
                #       (itr, np.mean(results2[-1]))
                #
                # with open('logs/sim_reach_temporal_conv/traing_loss.txt', 'a') as f:
                #     f.write("%d %f %f\n"%(itr,np.mean(results[-1]),np.mean(results2[-1])))



                # input_tensors = [train_op,
                #                  total_losses2[num_updates - 1]]
                # results = sess.run(input_tensors, feed_dict=feed_dict)
                # print '(Phase 3)Iteration %d: total_weighted_lossb is %2f ' % \
                #       (itr, np.mean(results[-1]))




                # input_tensors = [total_left_lossb,total_right_lossb,total_losses2[num_updates - 1]]
                # results = sess.run(input_tensors, feed_dict=feed_dict)
                # print '(Phase 3)Iteration %d: average left_postloss is %.2f, average right_postloss is %.2f, ' \
                #       'total_weighted_lossb is %2f' % (itr,  np.mean(results[-3]),np.mean(results[-2]), np.mean(results[-1]))





                # feed_dict = {global_obsa: compare_obsa_input,
                #              global_obsb: obsb_input,
                #              global_compare_obs:compare_obsa_input,
                #              global_statea: compare_statea_input,
                #              global_stateb: stateb_input,
                #              global_compare_state:compare_statea_input,
                #              global_actiona: compare_actiona_input,
                #              global_actionb: actionb_input,
                #              global_compare_action:compare_actiona_input}
                #
                # input_tensors = [train_op, total_loss1, total_losses2[num_updates - 1]]
                # results = sess.run(input_tensors, feed_dict=feed_dict)
                # print '(Phase 2)Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(results[-2]), np.mean(results[-1]))
                #
                # feed_dict = {global_obsa: obsa_input,
                #              global_obsb: obsb_input,
                #              global_compare_obs:obsa_input,
                #              global_statea: statea_input,
                #              global_stateb: stateb_input,
                #              global_compare_state:statea_input,
                #              global_actiona: actiona_input,
                #              global_actionb: actionb_input,
                #              global_compare_action:actiona_input}
                #
                # input_tensors = [train_op, total_loss1, total_losses2[num_updates - 1]]
                # results = sess.run(input_tensors, feed_dict=feed_dict)
                # print '(Phase 3)Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(results[-2]), np.mean(results[-1]))

                # if FLAGS.learn_bicycle:
                #     feed_dict = {global_obsa: obsb_input,
                #                  global_obsb: obsa_input,
                #                  global_statea: stateb_input,
                #                  global_stateb: statea_input,
                #                  global_actiona: actionb_input,
                #                  global_actionb: actiona_input}
                #     input_tensors = [train_op, total_loss1, total_losses2[num_updates - 1]]
                #     results = sess.run(input_tensors, feed_dict=feed_dict)
                #     print '(Phase 2)Iteration %d: average preloss is %.2f, average postloss is %.2f' % (
                #         itr, np.mean(results[-2]), np.mean(results[-1]))



                if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
                    print 'Saving model to: %s' % (save_dir + '_%d' % itr)
                    saver.save(sess, save_dir + '_%d' % itr)

        def evaluate_vision_reach(env, graph, data_generator, sess, exp_string, record_gifs, log_dirs):
            T = FLAGS.T
            scale, bias = load_scale_and_bias('mil_data/data/scale_and_bias_sim_vision_reach.pkl')
            successes = []
            selected_demo = data_generator.selected_demo
            # compare_selected_demo = data_generator.compare_selected_demo
            # compare_selected_demo = data_generator.goal_demo
            compare_selected_demo = data_generator.selected_demo
            if record_gifs:
                record_gifs_dir = os.path.join(log_dirs, 'evaluated_gifs')
                mkdir_p(record_gifs_dir)
            for i in xrange(len(selected_demo['selected_demoX'])):
                selected_demoO = selected_demo['selected_demoO'][i]
                selected_demoX = selected_demo['selected_demoX'][i]
                selected_demoU = selected_demo['selected_demoU'][i]

                # print('selected_demoO', selected_demoO.shape)

                compare_selected_demoO = compare_selected_demo['selected_demoO'][i]
                compare_selected_demoX = compare_selected_demo['selected_demoX'][i]
                compare_selected_demoU = compare_selected_demo['selected_demoU'][i]
                if record_gifs:
                    gifs_dir = os.path.join(record_gifs_dir, 'color_%d' % i)
                    mkdir_p(gifs_dir)
                for j in xrange(REACH_DEMO_CONDITIONS):
                    if j in data_generator.demos[i]['demoConditions']:
                        dists = []
                        # ob = env.reset()
                        # use env.set_state here to arrange blocks
                        Os = []
                        for t in range(T):
                            # import pdb; pdb.set_trace()
                            env.render()
                            # time.sleep(0.05)
                            obs, state = env.env.get_current_image_obs()
                            sum_obs = 0.0
                            sum_state = 0.0
                            sum_obs += np.sum(obs)
                            sum_state += np.sum(sum_state)
                            # print('i', i, 'j', j, 't',t,'sum_obs', sum_obs, 'sum_state', sum_state,'obs.shape',obs.shape)
                            while sum_obs == 0:
                                env.render()
                                obs, state = env.env.get_current_image_obs()
                                sum_obs = 0.0
                                sum_state = 0.0
                                sum_obs += np.sum(obs)
                                sum_state += np.sum(sum_state)
                            Os.append(obs)
                            obs = np.transpose(obs, [2, 1, 0]) / 255.0
                            obs = obs.reshape(1, 1, -1)
                            state = state.reshape(1, 1, -1)

                            # feed_dict = {
                            #     global_obsa: selected_demoO,
                            #     global_obsb: compare_selected_demoO,
                            #     global_compare_obs: obs,
                            #     global_statea: selected_demoX.dot(scale) + bias,
                            #     global_stateb: compare_selected_demoX.dot(scale) + bias,
                            #     global_compare_state: state.dot(scale) + bias,
                            #     # global_actionb: selected_demoU,
                            #     # global_compare_action: selected_demoU,
                            # }


                            # feed_dict = {
                            #     global_obsa: selected_demoO,
                            #     global_obsb: compare_selected_demoO,
                            #     global_compare_obs: obs,
                            #     global_statea: selected_demoX.dot(scale) + bias,
                            #     global_stateb: compare_selected_demoX.dot(scale) + bias,
                            #     global_compare_state: state.dot(scale) + bias,
                            #     # global_actionb: selected_demoU,
                            #     # global_compare_action: selected_demoU,
                            # }
                            feed_dict = {
                                global_obsa: selected_demoO,
                                global_obsb: selected_demoO,
                                global_compare_obs: obs,
                                global_statea: selected_demoX.dot(scale) + bias,
                                global_stateb: selected_demoX.dot(scale) + bias,
                                global_compare_state: state.dot(scale) + bias,
                                # global_actionb: selected_demoU,
                                # global_compare_action: selected_demoU,
                            }
                            with graph.as_default():
                                action = sess.run(test_act_op, feed_dict=feed_dict)
                            ob, reward, done, reward_dict = env.step(np.squeeze(action))
                            dist = -reward_dict['reward_dist']
                            if t >= T - REACH_SUCCESS_TIME_RANGE:
                                dists.append(dist)
                        if np.amin(dists) <= REACH_SUCCESS_THRESH:
                            successes.append(1.)
                        else:
                            successes.append(0.)
                        if record_gifs:
                            video = np.array(Os)
                            record_gif_path = os.path.join(gifs_dir, 'cond%d.samp0.gif' % j)
                            # print 'Saving gif sample to :%s' % record_gif_path
                            imageio.mimwrite(record_gif_path, video)
                    env.render(close=True)
                    if j != REACH_DEMO_CONDITIONS - 1 or i != len(selected_demo['selected_demoX']) - 1:
                        env.env.next()
                        env.render()
                        # time.sleep(0.5)
                        time.sleep(0.1)
                if i % 5  == 0:
                    print "Task %d: current success rate is %.5f" % (i, np.mean(successes))
            success_rate_msg = "Final success rate is %.5f" % (np.mean(successes))
            print success_rate_msg
            with open('logs/sim_reach_temporal_conv/log_sim_vision_reach.txt', 'a') as f:
                f.write(exp_string + ':\n')
                f.write(success_rate_msg + '\n')



        def reinforce_actions(actions,rewards,T):

            temp_actions=np.copy(actions)
            # print('actions',actions)
            abs_actions=np.abs(actions)
            value_range=5*np.min(abs_actions)
            if value_range<0.005:
                value_range=0.005

            # print('value_range',value_range)
            for i in range(1,T):
                # if rewards[i]>rewards[i-1]:
                #     temp_actions[i]=actions[i]+(rewards[i]-rewards[i-1])*temp_actions[i]*0.1+np.random.uniform(-0.001,0.001,actions[i].shape)
                temp_actions[i] = actions[i] + np.random.uniform(-value_range, value_range, actions[i].shape)

            return temp_actions


        def evaluate_rl_vision_reach(graph, data_generator, sess, exp_string, record_gifs, log_dirs,iterations=10,learn_examples=150):

            T = FLAGS.T
            scale, bias = load_scale_and_bias('mil_data/data/scale_and_bias_sim_vision_reach.pkl')

            selected_demo = data_generator.selected_demo
            if record_gifs:
                record_gifs_dir = os.path.join(log_dirs, 'evaluated_gifs')
                mkdir_p(record_gifs_dir)
            for itr in range(iterations):
              successes = []
              env = gym.make('ReacherMILTest-v1')
              env.reset()
              rl_env = gym.make('ReacherMILTest-v1')
              rl_env.reset()
              print('iter',itr)
              for i in xrange(learn_examples):
                print('data_generator.demos[i][\'demoConditions\']',data_generator.demos[i]['demoConditions'])
                selected_demoO = selected_demo['selected_demoO'][i]
                selected_demoX = selected_demo['selected_demoX'][i]
                selected_demoU = selected_demo['selected_demoU'][i]
                if record_gifs:
                    gifs_dir = os.path.join(record_gifs_dir, 'color_%d' % i)
                    mkdir_p(gifs_dir)
                for j in xrange(REACH_DEMO_CONDITIONS):
                    if j in data_generator.demos[i]['demoConditions']:
                        dists = []
                        # ob = env.reset()
                        # use env.set_state here to arrange blocks22
                        Os = []
                        record_actions=[]
                        record_rewards=[]
                        record_obs=[]
                        record_states=[]
                        # sum_obs=0.0
                        # sum_state=0.0
                        for t in range(T):
                            # import pdb; pdb.set_trace()

                            env.render()
                            # time.sleep(0.05)
                            obs, state = env.env.get_current_image_obs()
                            sum_obs = 0.0
                            sum_state = 0.0
                            sum_obs += np.sum(obs)
                            sum_state+=np.sum(sum_state)
                            # print('i', i, 'j', j, 't',t,'sum_obs', sum_obs, 'sum_state', sum_state,'obs.shape',obs.shape)
                            while sum_obs==0:

                                env.render()
                                obs, state = env.env.get_current_image_obs()
                                sum_obs = 0.0
                                sum_state = 0.0
                                sum_obs += np.sum(obs)
                                sum_state += np.sum(sum_state)
                                # print('i', i, 'j', j, 't', t, 'sum_obs', sum_obs, 'sum_state', sum_state, 'obs.shape',obs.shape)


                            Os.append(obs)
                            obs = np.transpose(obs, [2, 1, 0]) / 255.0
                            obs = obs.reshape(1, 1, -1)
                            record_obs.append(obs)
                            state = state.reshape(1, 1, -1)
                            state=state.dot(scale) + bias
                            record_states.append(state)
                            feed_dict = {
                                global_obsa: selected_demoO,
                                global_statea: selected_demoX.dot(scale) + bias,
                                global_obsb: obs,
                                global_stateb: state,
                                global_compare_obs: obs,
                                global_compare_state: state,
                            }



                            with graph.as_default():
                                action = sess.run(test_act_op, feed_dict=feed_dict)
                                record_actions.append(action)
                            ob, reward, done, reward_dict = env.step(np.squeeze(action))
                            record_rewards.append(reward)

                            dist = -reward_dict['reward_dist']
                            if t >= T - REACH_SUCCESS_TIME_RANGE:
                                dists.append(dist)

                        env.render(close=True)
                        record_actions=np.array(record_actions)
                        record_actions =np.squeeze(record_actions)
                        rl_actions = reinforce_actions(record_actions, record_rewards, FLAGS.T)
                        record_actions = np.expand_dims(record_actions,axis=0)


                        record_obs=np.array(record_obs)
                        record_obs = np.squeeze(record_obs)
                        record_obs = np.expand_dims(record_obs,axis=0)

                        record_states=np.array(record_states)
                        record_states = np.squeeze(record_states)
                        record_states = np.expand_dims(record_states,axis=0)


                        record_rewards = np.array(record_rewards)
                        total_reward = np.sum(record_rewards)
                        rl_rewards = 0.0

                        rl_obs = []
                        rl_states=[]
                        for t in range(T):
                            rl_env.render()
                            obs, state = rl_env.env.get_current_image_obs()
                            sum_obs = 0.0
                            sum_state = 0.0
                            sum_obs += np.sum(obs)
                            sum_state += np.sum(sum_state)
                            # print('i', i, 'j', j, 't',t,'sum_obs', sum_obs, 'sum_state', sum_state,'obs.shape',obs.shape)
                            while sum_obs == 0:
                                rl_env.render()
                                obs, state = rl_env.env.get_current_image_obs()
                                sum_obs = 0.0
                                sum_state = 0.0
                                sum_obs += np.sum(obs)
                                sum_state += np.sum(sum_state)
                                # print('i', i, 'j', j, 't', t, 'sum_obs', sum_obs, 'sum_state', sum_state, 'obs.shape',obs.shape)


                            obs = np.transpose(obs, [2, 1, 0]) / 255.0
                            obs = obs.reshape(1, 1, -1)
                            rl_obs.append(obs)
                            state = state.reshape(1, 1, -1)
                            state = state.dot(scale) + bias
                            rl_states.append(state)
                            # time.sleep(0.1)
                            _, rl_reward, rl_done, _ = rl_env.step(np.squeeze(rl_actions[t]))
                            rl_rewards += rl_reward
                        rl_env.render(close=True)

                        rl_obs = np.array(rl_obs)
                        rl_obs = np.squeeze(rl_obs)
                        rl_obs = np.expand_dims(rl_obs, axis=0)

                        rl_states = np.array(rl_states)
                        rl_states = np.squeeze(rl_states)
                        rl_states = np.expand_dims(rl_states, axis=0)

                        rl_actions = np.expand_dims(rl_actions, axis=0)

                        print('iter',itr,'i', i, 'j', j, 'total_reward', total_reward, 'rl_rewards', rl_rewards)

                        # print('selected_demoO',selected_demoO.shape,'record_obs',record_obs.shape,'selected_demoU',selected_demoU.shape,'record_actions',record_actions.shape,
                        #       'selected_demoX',selected_demoX.shape,'record_states',record_states.shape)
                        if (rl_rewards>-1.5 and rl_rewards>total_reward) or (rl_rewards>-1.0):
                            feed_dict = {
                                global_obsa: selected_demoO,
                                global_statea: selected_demoX.dot(scale) + bias,
                                global_obsb: rl_obs,
                                global_stateb: rl_states,
                                global_compare_obs: rl_obs,
                                global_compare_state: rl_states,
                                global_compare_action: rl_actions
                            }


                            input_tensors = [self_learn_train_op, total_losses2]
                            results = sess.run(input_tensors, feed_dict=feed_dict)
                            print 'Self-learnling:Iteration %d_%d_%d: average postloss is %.6f' % (itr,i,j, np.mean(results[-1]))

                        # print('total_reward',total_reward,'record_actions',record_actions.shape,'record_rewards',record_rewards.shape)
                        # print('record_rewards',record_rewards)
                        if np.amin(dists) <= REACH_SUCCESS_THRESH:
                            successes.append(1.)
                        else:
                            successes.append(0.)

                        # if record_gifs:
                        #     video = np.array(Os)
                        #     record_gif_path = os.path.join(gifs_dir, 'cond%d.samp0.gif' % j)
                        #     print 'Saving gif sample to :%s' % record_gif_path
                        #     imageio.mimwrite(record_gif_path, video)
                    rl_env.render(close=True)
                    env.render(close=True)


                    if j != REACH_DEMO_CONDITIONS - 1 or i != len(selected_demo['selected_demoX']) - 1:

                        rl_env.env.next()
                        time.sleep(0.1)
                        env.env.next()
                        time.sleep(0.1)
                # if i % 5  == 0:
                print "Iter:%d,Task %d: current success rate is %.5f" % (itr,i, np.mean(successes))
              rl_env.render(close=True)
              env.render(close=True)
              success_rate_msg = "Iter:%d,Final success rate is %.5f" % (itr,np.mean(successes))
              with open('logs/sim_reach_temporal_conv//log_rl_sim_vision_reach.txt', 'a') as f:
                  f.write(exp_string + ':\n')
                  f.write(success_rate_msg + '\n')
              print success_rate_msg


        if FLAGS.train:
            data_generator.generate_batches(noisy=FLAGS.use_noisy_demos)
            train(graph, saver, sess, data_generator, log_dirs, restore_itr=FLAGS.restore_iter)

        else:
            REACH_SUCCESS_THRESH = 0.05
            REACH_SUCCESS_TIME_RANGE = 10
            REACH_DEMO_CONDITIONS = 10
            model_file = tf.train.latest_checkpoint(log_dirs)
            if FLAGS.begin_restore_iter!=FLAGS.end_restore_iter:
                iter_index=FLAGS.begin_restore_iter
                while iter_index<=FLAGS.end_restore_iter:
                    print('iter_index',iter_index)
                    if FLAGS.restore_iter >=0:
                        model_file = model_file[:model_file.index('model')] + 'model_' + str(iter_index)
                    if model_file:
                        ind1 = model_file.index('model')
                        resume_itr = int(model_file[ind1 + 6:])
                        print("Restoring model weights from " + model_file)
                        # saver = tf.train.Saver()
                        saver.restore(sess, model_file)
                    if 'reach' in FLAGS.experiment:
                        env = gym.make('ReacherMILTest-v1')
                        env.reset()
                        generate_test_demos(data_generator)
                        evaluate_vision_reach(env, graph, data_generator, sess, exp_string, FLAGS.record_gifs, log_dirs)
                        # evaluate_rl_vision_reach(graph, data_generator, sess, exp_string, FLAGS.record_gifs, log_dirs)
                    iter_index+=100
            else:
                if FLAGS.restore_iter > 0:
                    model_file = model_file[:model_file.index('model')] + 'model_' + str(FLAGS.restore_iter)
                if model_file:
                    ind1 = model_file.index('model')
                    resume_itr = int(model_file[ind1 + 6:])
                    print("Restoring model weights from " + model_file)
                    # saver = tf.train.Saver()
                    saver.restore(sess, model_file)
                if 'reach' in FLAGS.experiment:
                    env = gym.make('ReacherMILTest-v1')
                    env.reset()
                    generate_test_demos(data_generator)
                    evaluate_vision_reach(env, graph, data_generator, sess, exp_string, FLAGS.record_gifs, log_dirs)
                    # evaluate_rl_vision_reach(graph, data_generator, sess, exp_string, FLAGS.record_gifs, log_dirs)




if __name__ == "__main__":
    main()


