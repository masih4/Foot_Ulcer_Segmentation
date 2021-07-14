opts = {}
opts['resource'] = 'cpu' # 'gpu'
opts['tf_version'] = 1.14
opts['gpu_num'] = '0'
opts['imageType_test'] = '.jpg'
opts['number_of_channel'] = 3
opts['treshold'] = 0.5
## paths
opts['test_dir'] = '../test_images/'
opts['results_save_path'] ='../results/'
opts['models_save_path_1'] ='../saved_models/linknet/'
opts['models_save_path_2'] ='../saved_models/unet/'
opts['results_save_path_final'] ='../results/final/'

opts['k_fold'] = 5
opts['pretrained_model_1'] = 'efficientnetb1'
opts['pretrained_model_2'] = 'efficientnetb2'
opts['use_pretrained_flag'] = 1



opts['save_figures'] = 1

