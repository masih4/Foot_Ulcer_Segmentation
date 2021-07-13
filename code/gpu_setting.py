'''
GPU setting for tensorflow 2.0 and tensorflow 1.14:
GPU 0: Titan v
GPU 1: gtx 1070
Note: adjust it for your persnonal workstation
'''
from __future__ import absolute_import, division, print_function, unicode_literals

def gpu_setting(opts):
    import tensorflow as tf
    if opts['tf_version']==2:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
    else: # i.e. tf_version == 1.14
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        gpu_str = opts['gpu_num']
        config.gpu_options.visible_device_list=gpu_str
        set_session(tf.Session(config=config))
    return
