''' Training Frustum PointNets.

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
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int, default=4, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--tb_logging', action='store_true', help='Whether to log to Tensorboard.')
parser.add_argument('--log_freq', type=int, default=20, help='log frequency')
parser.add_argument('--best', type=float, default=0, help='Initial learning rate [default: 0.001]')
parser.add_argument('--train_split', default='train', help='Model name [default: frustum_pointnets_v1]')

FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE_SLICE = FLAGS.batch_size
BATCH_SIZE = FLAGS.batch_size * FLAGS.ngpu
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes
LOG_FREQ = FLAGS.log_freq

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
print(FLAGS)
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

best_boxacc = FLAGS.best
best_epoch  = 0
# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train',
    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True, one_hot=True)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch*BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def tower_loss(scope, inputs, is_training_pl, batch):

    pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
    heading_class_label_pl, heading_residual_label_pl, \
    size_class_label_pl, size_residual_label_pl = inputs

    bn_decay = get_bn_decay(batch)
    tf.summary.scalar('bn_decay', bn_decay)

    # Get model and losses
    end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
        is_training_pl, bn_decay=bn_decay)
    loss = MODEL.get_loss(labels_pl, centers_pl,
        heading_class_label_pl, heading_residual_label_pl,
        size_class_label_pl, size_residual_label_pl, end_points)
    #tf.summary.scalar('loss', loss)

    losses = tf.get_collection('losses', scope=scope)
    total_loss = tf.add_n(losses, name='total_loss')

    # Write summaries of bounding box IoU and segmentation accuracies
    iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
        end_points['center'], \
        end_points['heading_scores'], end_points['heading_residuals'], \
        end_points['size_scores'], end_points['size_residuals'], centers_pl, \
        heading_class_label_pl, heading_residual_label_pl, \
        size_class_label_pl, size_residual_label_pl], \
        [tf.float32, tf.float32])
    end_points['iou2ds'] = iou2ds
    end_points['iou3ds'] = iou3ds

    correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
        tf.to_int64(labels_pl))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
        float(BATCH_SIZE_SLICE*NUM_POINT)

    if FLAGS.tb_logging:
        for l in losses + [total_loss]:
            loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)

    return total_loss, end_points, accuracy

def gather(end_points_list):
    end_points = {}
    for key in end_points_list[0].keys():
        end_points[key] = tf.concat([ end[key] for end in end_points_list],0)
    return end_points

def slice_batch(X, i):
    return X[i * BATCH_SIZE_SLICE: (i+1) * BATCH_SIZE_SLICE]

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

    _inputs = (_pointclouds_pl, _one_hot_vec_pl, _labels_pl, _centers_pl, \
    _heading_class_label_pl, _heading_residual_label_pl, \
    _size_class_label_pl, _size_residual_label_pl)
    return _inputs

def train():
    ''' Main function for training and simple evaluation. '''
    # Get training operator
    batch = tf.get_variable('batch', [],
       initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = get_learning_rate(batch)
    tf.summary.scalar('learning_rate', learning_rate)
    if OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate,
           momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)

    inputs =  MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    tower_grads = []
    tower_end_points = []
    tower_losses = []
    tower_acc = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.ngpu):
            with tf.device('/gpu:%d'%i), tf.name_scope('GPU_%d'%i) as scope:
                print(i)
                ###=====================
                inputs_ = slice_input(inputs, i)
                loss_, end_points_, acc_ = tower_loss(scope, inputs_, is_training_pl, batch)
                # reuse variable for the next
                tf.get_variable_scope().reuse_variables()
                grads = optimizer.compute_gradients(loss_)
                tower_grads.append(grads)

                tower_end_points.append(end_points_)
                tower_losses.append(loss_)
                tower_acc.append(acc_)
                ###=====================

    #, mask_logits, iou2ds, iou3ds
    grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(grads,  global_step=batch)
    end_points = gather(tower_end_points)
    print("center:",end_points['center'].shape)

    total_loss = tf.reduce_mean(tower_losses)
    #tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('iou_2d', tf.reduce_mean(end_points['iou2ds']))
    tf.summary.scalar('iou_3d', tf.reduce_mean(end_points['iou2ds']))
    tf.summary.scalar('seg acc', tf.reduce_mean(tower_acc))

    saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Add summary writerscocogan_2Asyn_map_mse_mix_d
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

    # Init variables
    if FLAGS.restore_model_path is None:
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        saver.restore(sess, FLAGS.restore_model_path)

    pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
    heading_class_label_pl, heading_residual_label_pl, \
    size_class_label_pl, size_residual_label_pl = inputs

    ops = {'pointclouds_pl': pointclouds_pl,
           'one_hot_vec_pl': one_hot_vec_pl,
           'labels_pl': labels_pl,
           'centers_pl': centers_pl,
           'heading_class_label_pl': heading_class_label_pl,
           'heading_residual_label_pl': heading_residual_label_pl,
           'size_class_label_pl': size_class_label_pl,
           'size_residual_label_pl': size_residual_label_pl,
           'is_training_pl': is_training_pl,
           'end_points': end_points,
           'loss': total_loss,
           'train_op': train_op,
           'merged': merged,
           'step': batch}

    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(sess, ops, train_writer)
        eval_one_epoch(sess, ops, test_writer, epoch, saver)

def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)//BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, logits_val, centers_pred_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                ops['end_points']['mask_logits'], ops['end_points']['center'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)


        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        if (batch_idx+1)%LOG_FREQ == 0:
            segacc = total_correct / float(total_seen)
            iou2, iou3 = iou2ds_sum / float(BATCH_SIZE*LOG_FREQ), iou3ds_sum / float(BATCH_SIZE*LOG_FREQ)
            boxacc = float(iou3d_correct_cnt)/float(BATCH_SIZE*LOG_FREQ)
            content = ' %03d / %03d loss: %2.4f segacc: %.4f IoU(ground/3D): %.4f / %.4f boxAcc(0.7): %.4f' \
                %(batch_idx+1, num_batches, loss_sum / LOG_FREQ, segacc, iou2, iou3, boxacc)
            log_string(content)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0


def eval_one_epoch(sess, ops, test_writer, epoch, saver):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT, best_boxacc, best_epoch
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)//BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Simple evaluation with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['end_points']['mask_logits'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        #print("preds_val:",preds_val.shape)
        #print("batch_label:",batch_label.shape)
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:]
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0):
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))
    mean_loss = loss_sum / float(num_batches)
    segacc = total_correct / float(total_seen)
    segclassacc = np.mean(np.array(total_correct_class) / \
        np.array(total_seen_class,dtype=np.float))
    iou2 = iou2ds_sum / float(num_batches*BATCH_SIZE)
    iou3 = iou3ds_sum / float(num_batches*BATCH_SIZE)
    boxacc = float(iou3d_correct_cnt)/float(num_batches*BATCH_SIZE)
    content = ' loss: %2.4f segacc: %.4f seg_C_acc: %.4f IoU(ground/3D): %.4f / %.4f boxAcc(0.7): %.4f' \
        %(mean_loss, segacc, segclassacc, iou2, iou3, boxacc)
    log_string(content)

    model_name = "model_%s_%1.3f.ckpt" % (epoch, boxacc)
    save_path = saver.save(sess, os.path.join(LOG_DIR, model_name))
    log_string("Model saved in file: %s" % save_path)
    if boxacc> best_boxacc:
        best_boxacc = boxacc
        best_epoch  = epoch
        best_model_path = os.path.join(LOG_DIR, "model_best.ckpt")
        save_path = saver.save(sess, os.path.join(LOG_DIR, model_name))
        log_string("Best Model( %s ) saved in file: %s" % (model_name, best_model_path))

    EPOCH_CNT += 1

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
