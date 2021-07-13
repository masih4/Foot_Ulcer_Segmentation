#########################################################################
## setting hyper-parameters
import params_test
opts = params_test.opts
#########################################################################
## disabeling warning msg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import warnings
warnings.simplefilter('ignore')
import sys
sys.stdout.flush() # resolving tqdm problem
#########################################################################
## gpu configuration
from gpu_setting import gpu_setting
gpu_setting(opts)
#########################################################################
## importing all required libraraies
import numpy as np
import cv2
import os
import random
import time
from glob import glob
from tqdm import tqdm
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.io import imsave, imread
import segmentation_models as sm
import pandas as pd
import matplotlib.pyplot as plt
from model import *
from metric import *


def get_id_from_file_path(file_path, indicator):
    return file_path.split(os.path.sep)[-1].replace(indicator, '')
#########################################################################
## creating required dirS
if not os.path.exists(opts['results_save_path']+ 'temp/linknet/fold1'):
    os.makedirs(opts['results_save_path'] + 'temp/linknet/fold1')
if not os.path.exists(opts['results_save_path']+ 'temp/linknet/fold2'):
    os.makedirs(opts['results_save_path'] + 'temp/linknet/fold2')
if not os.path.exists(opts['results_save_path']+ 'temp/linknet/fold3'):
    os.makedirs(opts['results_save_path'] + 'temp/linknet/fold3')
if not os.path.exists(opts['results_save_path']+ 'temp/linknet/fold4'):
    os.makedirs(opts['results_save_path'] + 'temp/linknet/fold4')
if not os.path.exists(opts['results_save_path']+ 'temp/linknet/fold5'):
    os.makedirs(opts['results_save_path'] + 'temp/linknet/fold5')
if not os.path.exists(opts['results_save_path']+ 'temp/linknet/all_folds'):
    os.makedirs(opts['results_save_path'] + 'temp/linknet/all_folds')


if not os.path.exists(opts['results_save_path']+ 'temp/unet/fold1'):
    os.makedirs(opts['results_save_path'] + 'temp/unet/fold1')
if not os.path.exists(opts['results_save_path']+ 'temp/unet/fold2'):
    os.makedirs(opts['results_save_path'] + 'temp/unet/fold2')
if not os.path.exists(opts['results_save_path']+ 'temp/unet/fold3'):
    os.makedirs(opts['results_save_path'] + 'temp/unet/fold3')
if not os.path.exists(opts['results_save_path']+ 'temp/unet/fold4'):
    os.makedirs(opts['results_save_path'] + 'temp/unet/fold4')
if not os.path.exists(opts['results_save_path']+ 'temp/unet/fold5'):
    os.makedirs(opts['results_save_path'] + 'temp/unet/fold5')
if not os.path.exists(opts['results_save_path']+ 'temp/unet/all_folds'):
    os.makedirs(opts['results_save_path'] + 'temp/unet/all_folds')

if not os.path.exists(opts['results_save_path_final']+ 'figure'):
    os.makedirs(opts['results_save_path_final'] + 'figure/')
#########################################################################
## ids and file names for data loading
test_files = glob('{}*{}'.format(opts['test_dir'], opts['imageType_test']))
test_files.sort()

print("Total number of test images:", len(test_files))
#########################################################################
## for linknet
preprocess_input = sm.get_preprocessing(opts['pretrained_model_1'])
n_classes = 1

for K in range(opts['k_fold']):
    print('========================')
    print('prediction based on linknet {} in ensemble'.format(K+1))
    model_raw = sm.Linknet(opts['pretrained_model_1'], classes= n_classes, activation='sigmoid', encoder_weights='imagenet')
    model_raw.load_weights(opts['models_save_path_1'] + 'linknet_{}.h5'.format(K + 1))

    for i in tqdm(range(len(test_files))):#len(test_files)
        x = cv2.imread(test_files[i])
        if x.shape[0] > x.shape[1]:
            x_padded = cv2.copyMakeBorder(x, 0, 0, 0, (x.shape[0] - x.shape[1]), cv2.BORDER_CONSTANT, value=0)
        else:
            x_padded = cv2.copyMakeBorder(x, 0, (x.shape[1] - x.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=0)
        resize_target = int(int(x_padded.shape[0] / 32) * 32) + 32
        if resize_target < 96:
            resize_target = 96
        x_padded_unpretrain = np.copy(x_padded)
        if opts['use_pretrained_flag'] == 1:
            x_padded = preprocess_input(x_padded)
        x_padded_resized = cv2.resize(x_padded, (resize_target, resize_target))
        x_padded_resized_unpretrain = cv2.resize(x_padded_unpretrain, (resize_target, resize_target))
        ################################################## morphological TTA
        img_rotate_90_clockwise = cv2.rotate(x_padded_resized, cv2.ROTATE_90_CLOCKWISE)
        img_rotate_90_counterclockwise = cv2.rotate(x_padded_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_rotate_180 = cv2.rotate(x_padded_resized, cv2.ROTATE_180)
        img_flip_lr_0 = cv2.flip(x_padded_resized, 1)
        img_flip_lr_90 = cv2.flip(img_rotate_90_clockwise, 1)
        img_flip_lr_180 = cv2.flip(img_rotate_180, 1)
        img_flip_lr_270 = cv2.flip(img_rotate_90_counterclockwise, 1)
        ##################################################
        x_padded_resized_base = np.expand_dims(x_padded_resized, axis=0)
        ##################################################
        img_rotate_90_clockwise = np.expand_dims(img_rotate_90_clockwise, axis=0)
        img_rotate_90_counterclockwise = np.expand_dims(img_rotate_90_counterclockwise, axis=0)
        img_rotate_180 = np.expand_dims(img_rotate_180, axis=0)
        img_flip_lr_0 = np.expand_dims(img_flip_lr_0, axis=0)
        img_flip_lr_90 = np.expand_dims(img_flip_lr_90, axis=0)
        img_flip_lr_180 = np.expand_dims(img_flip_lr_180, axis=0)
        img_flip_lr_270 = np.expand_dims(img_flip_lr_270, axis=0)

        pred_test_base1 = model_raw.predict(x_padded_resized_base, verbose=0, batch_size=1)
        pred_test_base2 = model_raw.predict(img_rotate_90_clockwise, verbose=0, batch_size=1)
        pred_test_base3 = model_raw.predict(img_rotate_90_counterclockwise, verbose=0, batch_size=1)
        pred_test_base4 = model_raw.predict(img_rotate_180, verbose=0, batch_size=1)
        pred_test_base5 = model_raw.predict(img_flip_lr_0, verbose=0, batch_size=1)
        pred_test_base6 = model_raw.predict(img_flip_lr_90, verbose=0, batch_size=1)
        pred_test_base7 = model_raw.predict(img_flip_lr_180, verbose=0, batch_size=1)
        pred_test_base8 = model_raw.predict(img_flip_lr_270, verbose=0, batch_size=1)

        pred_test_base1 = np.squeeze(pred_test_base1)
        pred_test_base2 = np.squeeze(pred_test_base2)
        pred_test_base3 = np.squeeze(pred_test_base3)
        pred_test_base4 = np.squeeze(pred_test_base4)
        pred_test_base5 = np.squeeze(pred_test_base5)
        pred_test_base6 = np.squeeze(pred_test_base6)
        pred_test_base7 = np.squeeze(pred_test_base7)
        pred_test_base8 = np.squeeze(pred_test_base8)

        pred_test_base2 = cv2.rotate(pred_test_base2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred_test_base3 = cv2.rotate(pred_test_base3, cv2.ROTATE_90_CLOCKWISE)
        pred_test_base4 = cv2.rotate(pred_test_base4, cv2.ROTATE_180)
        pred_test_base5 = cv2.flip(pred_test_base5, 1)
        pred_test_base6 = cv2.flip(pred_test_base6, 1)
        pred_test_base6 = cv2.rotate(pred_test_base6, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred_test_base7 = cv2.flip(pred_test_base7, 1)
        pred_test_base7 = cv2.rotate(pred_test_base7, cv2.ROTATE_180)
        pred_test_base8 = cv2.flip(pred_test_base8, 1)
        pred_test_base8 = cv2.rotate(pred_test_base8, cv2.ROTATE_90_CLOCKWISE)

        pred_test_tot = []
        pred_test_tot = (pred_test_base1 + pred_test_base2+ pred_test_base3+ pred_test_base4+ pred_test_base5+ pred_test_base6+ pred_test_base7+ pred_test_base8)/8

        preds_test_resize = cv2.resize(pred_test_tot, (x_padded.shape[1], x_padded.shape[0]))
        preds_test_orgSize = preds_test_resize[0:x.shape[0], 0:x.shape[1]]


        np.save(opts['results_save_path'] + 'temp/linknet/fold{}/{}'.format(K+1,get_id_from_file_path(test_files[i], opts['imageType_test'])), preds_test_orgSize)
        preds_test_orgSize[preds_test_orgSize > 1] = 1
        imsave(opts['results_save_path'] + 'temp/linknet/fold{}/{}.png'.format(K+1,get_id_from_file_path(test_files[i], opts['imageType_test'])), np.uint8(preds_test_orgSize*255))


        del pred_test_tot

if opts['k_fold'] == 5:
    fold1_files = glob(opts['results_save_path'] + 'temp/linknet/fold1/*.npy')
    fold2_files = glob(opts['results_save_path'] + 'temp/linknet/fold2/*.npy')
    fold3_files = glob(opts['results_save_path'] + 'temp/linknet/fold3/*.npy')
    fold4_files = glob(opts['results_save_path'] + 'temp/linknet/fold4/*.npy')
    fold5_files = glob(opts['results_save_path'] + 'temp/linknet/fold5/*.npy')
    fold1_files.sort()
    fold2_files.sort()
    fold3_files.sort()
    fold4_files.sort()
    fold5_files.sort()
    for i in range(len(fold1_files)):
        x1 = np.load(fold1_files[i])
        x2 = np.load(fold2_files[i])
        x3 = np.load(fold3_files[i])
        x4 = np.load(fold4_files[i])
        x5 = np.load(fold5_files[i])

        all_folds_stage1 = (x1 + x2 + x3 + x4 +x5)/5
        np.save(opts['results_save_path'] + 'temp/linknet/all_folds/{}'.format(get_id_from_file_path(test_files[i], opts['imageType_test'])), all_folds_stage1)
##############################################################################
## for unet
preprocess_input = sm.get_preprocessing(opts['pretrained_model_2'])
n_classes = 1

for K in range(opts['k_fold']):
    print('========================')
    print('prediction based on unet {} in ensemble'.format(K+1))
    model_raw = sm.Unet(opts['pretrained_model_2'], classes=n_classes, activation='sigmoid', encoder_weights='imagenet',decoder_block_type='transpose')
    model_raw.load_weights(opts['models_save_path_2'] + 'unet_{}.h5'.format(K + 1))

    for i in tqdm(range(len(test_files))):#len(test_files)
        x = cv2.imread(test_files[i])
        if x.shape[0] > x.shape[1]:
            x_padded = cv2.copyMakeBorder(x, 0, 0, 0, (x.shape[0] - x.shape[1]), cv2.BORDER_CONSTANT, value=0)
        else:
            x_padded = cv2.copyMakeBorder(x, 0, (x.shape[1] - x.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=0)
        resize_target = int(int(x_padded.shape[0] / 32) * 32) + 32
        if resize_target < 96:
            resize_target = 96
        x_padded_unpretrain = np.copy(x_padded)
        if opts['use_pretrained_flag'] == 1:
            x_padded = preprocess_input(x_padded)
        x_padded_resized = cv2.resize(x_padded, (resize_target, resize_target))
        x_padded_resized_unpretrain = cv2.resize(x_padded_unpretrain, (resize_target, resize_target))
        ################################################## morphological TTA
        img_rotate_90_clockwise = cv2.rotate(x_padded_resized, cv2.ROTATE_90_CLOCKWISE)
        img_rotate_90_counterclockwise = cv2.rotate(x_padded_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_rotate_180 = cv2.rotate(x_padded_resized, cv2.ROTATE_180)
        img_flip_lr_0 = cv2.flip(x_padded_resized, 1)
        img_flip_lr_90 = cv2.flip(img_rotate_90_clockwise, 1)
        img_flip_lr_180 = cv2.flip(img_rotate_180, 1)
        img_flip_lr_270 = cv2.flip(img_rotate_90_counterclockwise, 1)
        ##################################################
        x_padded_resized_base = np.expand_dims(x_padded_resized, axis=0)
        ##################################################
        img_rotate_90_clockwise = np.expand_dims(img_rotate_90_clockwise, axis=0)
        img_rotate_90_counterclockwise = np.expand_dims(img_rotate_90_counterclockwise, axis=0)
        img_rotate_180 = np.expand_dims(img_rotate_180, axis=0)
        img_flip_lr_0 = np.expand_dims(img_flip_lr_0, axis=0)
        img_flip_lr_90 = np.expand_dims(img_flip_lr_90, axis=0)
        img_flip_lr_180 = np.expand_dims(img_flip_lr_180, axis=0)
        img_flip_lr_270 = np.expand_dims(img_flip_lr_270, axis=0)

        pred_test_base1 = model_raw.predict(x_padded_resized_base, verbose=0, batch_size=1)
        pred_test_base2 = model_raw.predict(img_rotate_90_clockwise, verbose=0, batch_size=1)
        pred_test_base3 = model_raw.predict(img_rotate_90_counterclockwise, verbose=0, batch_size=1)
        pred_test_base4 = model_raw.predict(img_rotate_180, verbose=0, batch_size=1)
        pred_test_base5 = model_raw.predict(img_flip_lr_0, verbose=0, batch_size=1)
        pred_test_base6 = model_raw.predict(img_flip_lr_90, verbose=0, batch_size=1)
        pred_test_base7 = model_raw.predict(img_flip_lr_180, verbose=0, batch_size=1)
        pred_test_base8 = model_raw.predict(img_flip_lr_270, verbose=0, batch_size=1)

        pred_test_base1 = np.squeeze(pred_test_base1)
        pred_test_base2 = np.squeeze(pred_test_base2)
        pred_test_base3 = np.squeeze(pred_test_base3)
        pred_test_base4 = np.squeeze(pred_test_base4)
        pred_test_base5 = np.squeeze(pred_test_base5)
        pred_test_base6 = np.squeeze(pred_test_base6)
        pred_test_base7 = np.squeeze(pred_test_base7)
        pred_test_base8 = np.squeeze(pred_test_base8)

        pred_test_base2 = cv2.rotate(pred_test_base2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred_test_base3 = cv2.rotate(pred_test_base3, cv2.ROTATE_90_CLOCKWISE)
        pred_test_base4 = cv2.rotate(pred_test_base4, cv2.ROTATE_180)
        pred_test_base5 = cv2.flip(pred_test_base5, 1)
        pred_test_base6 = cv2.flip(pred_test_base6, 1)
        pred_test_base6 = cv2.rotate(pred_test_base6, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred_test_base7 = cv2.flip(pred_test_base7, 1)
        pred_test_base7 = cv2.rotate(pred_test_base7, cv2.ROTATE_180)
        pred_test_base8 = cv2.flip(pred_test_base8, 1)
        pred_test_base8 = cv2.rotate(pred_test_base8, cv2.ROTATE_90_CLOCKWISE)

        pred_test_tot = []
        pred_test_tot = (pred_test_base1 + pred_test_base2+ pred_test_base3+ pred_test_base4+ pred_test_base5+ pred_test_base6+ pred_test_base7+ pred_test_base8)/8

        preds_test_resize = cv2.resize(pred_test_tot, (x_padded.shape[1], x_padded.shape[0]))
        preds_test_orgSize = preds_test_resize[0:x.shape[0], 0:x.shape[1]]


        np.save(opts['results_save_path'] + 'temp/unet/fold{}/{}'.format(K+1,get_id_from_file_path(test_files[i], opts['imageType_test'])), preds_test_orgSize)
        preds_test_orgSize[preds_test_orgSize > 1] = 1
        imsave(opts['results_save_path'] + 'temp/unet/fold{}/{}.png'.format(K+1,get_id_from_file_path(test_files[i], opts['imageType_test'])), np.uint8(preds_test_orgSize*255))


        del pred_test_tot

if opts['k_fold'] == 5:
    fold1_files = glob(opts['results_save_path'] + 'temp/unet/fold1/*.npy')
    fold2_files = glob(opts['results_save_path'] + 'temp/unet/fold2/*.npy')
    fold3_files = glob(opts['results_save_path'] + 'temp/unet/fold3/*.npy')
    fold4_files = glob(opts['results_save_path'] + 'temp/unet/fold4/*.npy')
    fold5_files = glob(opts['results_save_path'] + 'temp/unet/fold5/*.npy')
    fold1_files.sort()
    fold2_files.sort()
    fold3_files.sort()
    fold4_files.sort()
    fold5_files.sort()
    for i in range(len(fold1_files)):
        x1 = np.load(fold1_files[i])
        x2 = np.load(fold2_files[i])
        x3 = np.load(fold3_files[i])
        x4 = np.load(fold4_files[i])
        x5 = np.load(fold5_files[i])

        all_folds_stage1 = (x1 + x2 + x3 + x4 +x5)/5
        np.save(opts['results_save_path'] + 'temp/unet/all_folds/{}'.format(get_id_from_file_path(test_files[i], opts['imageType_test'])), all_folds_stage1)
##############################################################################

print('==================================')
print('final prediction')


stage1_files = glob(opts['results_save_path'] + 'temp/linknet/all_folds/*.npy') # from segmentation u-net
stage1_files.sort()

stage2_files = glob(opts['results_save_path'] + 'temp/unet/all_folds/*.npy') # from segmentation u-net
stage2_files.sort()



for i in tqdm(range(len(test_files))):#len(test_files)
    x = cv2.imread(test_files[i])
    ensemble_stage1 = np.load(stage1_files[i])
    ensemble_stage2 = np.load(stage2_files[i])
    res = np.array((ensemble_stage1+ensemble_stage2)/2)

    stage1_mask_ep = np.uint(res>0.5)

    img_rmv_small = stage1_mask_ep
    img_rmv_small = binary_fill_holes(img_rmv_small > 0).astype(float)
    img_rmv_small = remove_small_objects(img_rmv_small>0.5, min_size=int(100), connectivity=2)



   #########################
    if opts['save_figures']:
        imsave(opts['results_save_path_final'] + '/{}.png'.format(get_id_from_file_path(test_files[i], opts['imageType_test'])),(img_rmv_small*255).astype(np.uint16))
        plt.figure(figsize=(60, 30))
        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.title('test image', fontsize=50)

        plt.subplot(1, 2, 2)
        plt.imshow(img_rmv_small, cmap='flag_r')
        plt.title('prediction', fontsize=50)
        plt.savefig(opts['results_save_path_final'] + 'figure/{}.png'.format(get_id_from_file_path(test_files[i], opts['imageType_test'])))
        plt.close()


