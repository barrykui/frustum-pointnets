''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import provider
from train_util import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--ngpu', type=int, default=4, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4

# Load Frustum Datasets.
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True, overwritten_data_path=FLAGS.data_path,
    from_rgb_detection=FLAGS.from_rgb_detection, one_hot=True)


def tower_loss(scope, inputs, is_training_pl, batch):
    """Calculate the total loss on a single tower running the MNIST model.
    Args:
      scope: unique prefix string identifying the MNIST tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
    heading_class_label_pl, heading_residual_label_pl, \
    size_class_label_pl, size_residual_label_pl = inputs

    # Get model and losses
    end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
        is_training_pl)
    loss = MODEL.get_loss(labels_pl, centers_pl,
        heading_class_label_pl, heading_residual_label_pl,
        size_class_label_pl, size_residual_label_pl, end_points)

    # Write summaries of bounding box IoU and segmentation accuracies
    iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
        end_points['center'], \
        end_points['heading_scores'], end_points['heading_residuals'], \
        end_points['size_scores'], end_points['size_residuals'], \
        centers_pl, \
        heading_class_label_pl, heading_residual_label_pl, \
        size_class_label_pl, size_residual_label_pl], \
        [tf.float32, tf.float32])
    end_points['iou2ds'] = iou2ds
    end_points['iou3ds'] = iou3ds

    return loss, end_points

def gather(end_points_list):
    end_points = {}
    for key in end_points_list[0].keys():
        end_points[key] = tf.concat([ end[key] for end in end_points_list],0)
    return end_points



def slice_batch(X, i):
    return X[i * BATCH_SIZE: (i+1) * BATCH_SIZE]

def slice_input(inputs, i):
    pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
    heading_class_label_pl, heading_residual_label_pl, \
    size_class_label_pl, size_residual_label_pl = inputs

    _pointclouds_pl,                        _one_hot_vec_pl, \
    _labels_pl,                             _centers_pl, \
    _heading_class_label_pl,                _heading_residual_label_pl, \
    _size_class_label_pl,                   _size_residual_label_pl = \
    slice_batch(pointclouds_pl, i),         slice_batch(one_hot_vec_pl,i), \
    slice_batch(labels_pl, i),              slice_batch(centers_pl, i), \
    slice_batch(heading_class_label_pl, i), slice_batch(heading_residual_label_pl,i), \
    slice_batch( size_class_label_pl, i),   slice_batch(size_residual_label_pl,i)
    #print("pointclouds_pl:", pointclouds_pl.shape)
    #print("_pointclouds_pl:", _pointclouds_pl.shape)
    _inputs = (_pointclouds_pl, _one_hot_vec_pl, _labels_pl, _centers_pl, \
    _heading_class_label_pl, _heading_residual_label_pl, \
    _size_class_label_pl, _size_residual_label_pl)
    return _inputs

def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    #with tf.device('/gpu:0'):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    batch = tf.get_variable('batch', [],
       initializer=tf.constant_initializer(0), trainable=False)
    inputs =  MODEL.placeholder_inputs(BATCH_SIZE*FLAGS.ngpu, NUM_POINT)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    tower_grads = []
    end_points_list = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.ngpu):
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('GPU_%d'%i) as scope:
                    print(i)
                    ###=====================
                    _inputs = slice_input(inputs, i)
                    loss, _end_points = tower_loss(scope, _inputs, is_training_pl, batch)
                    end_points_list.append(_end_points)
                    tf.get_variable_scope().reuse_variables()
                    #grads = optimizer.compute_gradients(loss)
                    #tower_grads.append(grads)
                    #print("_center:",_center.shape)
    saver = tf.train.Saver()
    end_points = gather(end_points_list)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
    heading_class_label_pl, heading_residual_label_pl, \
    size_class_label_pl, size_residual_label_pl = inputs
    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    ops = {'pointclouds_pl': pointclouds_pl,
           'one_hot_vec_pl': one_hot_vec_pl,
           'labels_pl': labels_pl,
           'centers_pl': centers_pl,
           'heading_class_label_pl': heading_class_label_pl,
           'heading_residual_label_pl': heading_residual_label_pl,
           'size_class_label_pl': size_class_label_pl,
           'size_residual_label_pl': size_residual_label_pl,
           'is_training_pl': is_training_pl,
           'end_points': end_points}
    return sess, ops

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]//batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],)) # 3D box score

    ep = ops['end_points']
    for i in range(num_batches):
        feed_dict = {\
            ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
            ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ep['mask_logits'], ep['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)

        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
        size_cls, size_res, scores

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def test_from_rgb_detection(output_filename, result_dir=None):
    ''' Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    onehot_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    print(len(TEST_DATASET))
    batch_size = BATCH_SIZE*FLAGS.ngpu
    num_batches = int((len(TEST_DATASET)+batch_size-1)/batch_size)

    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    for batch_idx in range(num_batches):
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec

        # Run one batch inference
        batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, batch_sclass_pred, batch_sres_pred, batch_scores = inference(sess, ops, batch_data_to_feed, batch_one_hot_to_feed, batch_size=batch_size)

        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            #score_list.append(batch_scores[i])
            score_list.append(batch_rgb_prob[i]) # 2D RGB detection score
            onehot_list.append(batch_one_hot_vec[i])

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)
            pickle.dump(onehot_list, fp)

    # Write detection results for KITTI evaluation
    print('Number of point clouds: %d' % (len(ps_list)))
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    output_dir = os.path.join(result_dir, 'data')
    if FLAGS.idx_path is not None:
        to_fill_filename_list = [line.rstrip()+'.txt' \
            for line in open(FLAGS.idx_path)]
        fill_files(output_dir, to_fill_filename_list)

def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE *FLAGS.ngpu
    num_batches = len(TEST_DATASET)//batch_size

    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    correct_cnt = 0
    for batch_idx in range(num_batches):
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data,
                batch_one_hot_vec, batch_size=batch_size)

        correct_cnt += np.sum(batch_output==batch_label)

        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i,...])
            seg_list.append(batch_label[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])

    print("Segmentation accuracy: %f" % \
        (correct_cnt / float(batch_size*num_batches*NUM_POINT)))

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(seg_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)

    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
        heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)


if __name__=='__main__':
    if FLAGS.from_rgb_detection:
        test_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    else:
        test(FLAGS.output+'.pickle', FLAGS.output)
