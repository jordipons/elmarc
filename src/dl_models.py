import tensorflow as tf

def build(config, x_in):
    if config['CNN']['architecture'] == 'cnn_small_filters':
        return cnn_small_filters(config, x_in)
    elif config['CNN']['architecture'] == 'cnn_single':
        return cnn_single(config, x_in)
    elif config['CNN']['architecture'] == 'cnn_music':
        return cnn_music(config, x_in)
    elif config['CNN']['architecture'] == 'sample_level':
        return sample_level(config, x_in)
    elif config['CNN']['architecture'] == 'frame_level':
        return frame_level(config, x_in)
    elif config['CNN']['architecture'] == 'frame_level_many':
        return frame_level_many(config, x_in)
    elif config['CNN']['architecture'] == 'cnn_audio':
        return cnn_audio(config, x_in)
    # + backend!

def cnn_small_filters(config, x_in):

    with tf.name_scope('cnn_small_filters'):

        print('[SMALL FILTERS] Input: ' + str(x_in.get_shape))

        input_layer = tf.reshape(x_in,[-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 2], strides=[4, 2])

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='2CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 3], strides=[4, 3])

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='3CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[5, 2], strides=[5, 2])

        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 name='4CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[4, 2], strides=[4, 2])

        conv5 = tf.layers.conv2d(inputs=pool4, 
                                 filters=config['CNN']['num_filters'], 
                                 kernel_size=[3, 3], 
                                 padding='same', 
                                 activation=tf.nn.elu,
                                 name='5CNN', 
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[4, 4], strides=[4, 4])



    print(pool1.get_shape)
    print(pool2.get_shape)
    print(pool3.get_shape)
    print(pool4.get_shape)
    print(pool5.get_shape)

    return [pool1, pool2, pool3, pool4, pool5]

def cnn_single(config, x_in):

    with tf.name_scope('cnn_single'):

        print('[CNN SINGLE] Input: ' + str(x_in.get_shape))

        input_layer = tf.reshape(x_in,[-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=config['CNN']['num_filters'],
                                 kernel_size=config['CNN']['filter_shape'],
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 name='1CNN',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=config['CNN']['pool_shape'], strides=config['CNN']['pool_shape'])

        
    print(conv1.get_shape)
    print(pool1.get_shape)

    return [conv1, pool1]


def cnn_music(config, x_in):
   
    # remove some temporal filters to have the same ammount of timbral and temporal filters
    if config['CNN']['num_filters'] == 256:
        remove = 64  
    elif config['CNN']['num_filters'] == 128:
        remove = 32    
    elif config['CNN']['num_filters'] == 64:
        remove = 16
    elif config['CNN']['num_filters'] == 32:
        remove = 8
    elif config['CNN']['num_filters'] == 16:
        remove = 4
    elif config['CNN']['num_filters'] == 8:
        remove = 2
    elif config['CNN']['num_filters'] == 4:
        remove = 1

    # define the cnn_music model  
    with tf.name_scope('cnn_music'):

        print('[MUSIC] Input: ' + str(x_in.get_shape))

        input_layer = tf.reshape(x_in, [-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
        input_pad_3 = tf.pad(input_layer, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")

        # [TIMBRE] filter shape 1: 7x0.9f
        conv1 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[1, conv1.shape[2]],
                                    strides=[1, conv1.shape[2]])
        p1 = tf.squeeze(pool1, [2])

        # [TIMBRE] filter shape 2: 3x0.9f
        conv2 = tf.layers.conv2d(inputs=input_pad_3, 
                             filters=config['CNN']['num_filters']*2,
                             kernel_size=[3, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[1, conv2.shape[2]],
                                    strides=[1, conv2.shape[2]])
        p2 = tf.squeeze(pool2, [2])

        # [TIMBRE] filter shape 3: 1x0.9f
        conv3 = tf.layers.conv2d(inputs=input_layer, 
                             filters=config['CNN']['num_filters']*4,
                             kernel_size=[1, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[1, conv3.shape[2]],
                                    strides=[1, conv3.shape[2]])
        p3 = tf.squeeze(pool3, [2])

        # [TIMBRE] filter shape 3: 7x0.4f
        conv4 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, int(0.4 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                    pool_size=[1, conv4.shape[2]],
                                    strides=[1, conv4.shape[2]])
        p4 = tf.squeeze(pool4, [2])

        # [TIMBRE] filter shape 5: 3x0.4f
        conv5 = tf.layers.conv2d(inputs=input_pad_3, 
                             filters=config['CNN']['num_filters']*2,
                             kernel_size=[3, int(0.4 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[1, conv5.shape[2]],
                                    strides=[1, conv5.shape[2]])
        p5 = tf.squeeze(pool5, [2])

        # [TIMBRE] filter shape 6: 1x0.4f
        conv6 = tf.layers.conv2d(inputs=input_layer, 
                             filters=config['CNN']['num_filters']*4,
                             kernel_size=[1, int(0.4 * config['CNN']['n_mels'])], padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, conv6.shape[2]],
                                    strides=[1, conv6.shape[2]])
        p6 = tf.squeeze(pool6, [2])

        # [TEMPORAL-FEATURES] - average pooling + filter shape 7: 165x1
        pool7 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool7_rs = tf.squeeze(pool7, [3])
        conv7 = tf.layers.conv1d(inputs=pool7_rs,
                             filters=config['CNN']['num_filters']-remove,
                             kernel_size=165,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 8: 128x1
        pool8 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool8_rs = tf.squeeze(pool8, [3])
        conv8 = tf.layers.conv1d(inputs=pool8_rs,
                             filters=config['CNN']['num_filters']*2-remove,
                             kernel_size=128,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 9: 64x1
        pool9 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool9_rs = tf.squeeze(pool9, [3])
        conv9 = tf.layers.conv1d(inputs=pool9_rs,
                             filters=config['CNN']['num_filters']*4-remove,
                             kernel_size=64,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 10: 32x1
        pool10 = tf.layers.average_pooling2d(inputs=input_layer,
                                         pool_size=[1, config['CNN']['n_mels']],
                                         strides=[1, config['CNN']['n_mels']])
        pool10_rs = tf.squeeze(pool10, [3])
        conv10 = tf.layers.conv1d(inputs=pool10_rs,
                              filters=config['CNN']['num_filters']*8-remove,
                              kernel_size=32,
                              padding="same",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # concatenate all feature maps
    timbral = tf.concat([p1, p2, p3, p4, p5, p6], 2)
    temporal = tf.concat([conv7, conv8, conv9, conv10], 2)

    print(timbral.get_shape)
    print(temporal.get_shape)

    # check [moving_mean, moving_variance, beta, gamma]
    #    [batch_normalization_8/moving_mean:0, batch_normalization_8/moving_variance:0,
    #    .. 'batch_normalization_8/beta', 'batch_normalization_8/gamma:0']
    #    sess.run('batch_normalization_8/moving_mean:0')

    return [timbral, temporal]


def backend(route_out, config):
    '''Function implementing the proposed back-end.
    - 'route_out': is the output of the front-end, and therefore the input of this function.
    - 'config': dictionary with some configurable parameters like: number of output units - config['numOutputNeurons']
                or number of frequency bins of the spectrogram config['setup_params']['yInput']
    '''

    # conv layer 1 - adapting dimensions
    conv1 = tf.layers.conv2d(inputs=route_out,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, route_out.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             name='1cnnOut',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv1_t = tf.transpose(conv1, [0, 1, 3, 2])

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.layers.conv2d(inputs=bn_conv1_pad,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, bn_conv1_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             name='2cnnOut',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv2_t = tf.transpose(conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2_t, conv1_t)

    # temporal pooling
    pool1 = tf.layers.max_pooling2d(inputs=res_conv2, pool_size=[2, 1], strides=[2, 1], name='poolOut')

    # conv layer 3 - residual connection
    bn_conv4_pad = tf.pad(pool1, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv5 = tf.layers.conv2d(inputs=bn_conv4_pad,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, bn_conv4_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             name='3cnnOut',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv5_t = tf.transpose(conv5, [0, 1, 3, 2])
    res_conv5 = tf.add(conv5_t, pool1)

    return [conv1_t, res_conv2, res_conv5]   

def sample_level(config, x_in):
    '''Function implementing the front-end proposed by Lee et al. 2017.
       Lee, et al. "Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms." 
       arXiv preprint arXiv:1703.01789 (2017).
    - 'x': placeholder whith the input.
    - 'is_training': placeholder indicating weather it is training or test phase, for dropout or batch norm.
    '''
    conv0 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=3,
                             strides=3,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    conv1 = tf.layers.conv1d(inputs=conv0,
                             filters=config['CNN']['num_filters'],
                             kernel_size=3,
                             strides=1,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool_1 = tf.layers.max_pooling1d(conv1, pool_size=3, strides=3)

    conv2 = tf.layers.conv1d(inputs=pool_1,
                             filters=config['CNN']['num_filters'],
                             kernel_size=3,
                             strides=1, 
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool_2 = tf.layers.max_pooling1d(conv2, pool_size=3, strides=3)

    conv3 = tf.layers.conv1d(inputs=pool_2,
                             filters=config['CNN']['num_filters'], # CHANGE NUMBER OF FILTERS?
                             kernel_size=3,
                             strides=1,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool_3 = tf.layers.max_pooling1d(conv3, pool_size=3, strides=3)

    conv4 = tf.layers.conv1d(inputs=pool_3,
                             filters=config['CNN']['num_filters'], # CHANGE NUMBER OF FILTERS?
                             kernel_size=3,
                             strides=1,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool_4 = tf.layers.max_pooling1d(conv4, pool_size=3, strides=3)

    conv5 = tf.layers.conv1d(inputs=pool_4,
                             filters=config['CNN']['num_filters'], # CHANGE NUMBER OF FILTERS?
                             kernel_size=3,
                             strides=1,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool_5 = tf.layers.max_pooling1d(conv5, pool_size=3, strides=3)

    conv6 = tf.layers.conv1d(inputs=pool_5,
                             filters=config['CNN']['num_filters'], # CHANGE NUMBER OF FILTERS?
                             kernel_size=3,
                             strides=1,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool_6 = tf.layers.max_pooling1d(conv6, pool_size=3, strides=3)

    print(pool_1.get_shape)
    print(pool_2.get_shape)
    print(pool_3.get_shape)
    print(pool_4.get_shape)
    print(pool_5.get_shape)
    print(pool_6.get_shape)

    return [conv0, pool_1, pool_2, pool_3, pool_4, pool_5, pool_6]

def frame_level(config, x_in):

    conv1 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=512,
                             strides=32,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    front_end_out = tf.expand_dims(conv1, 3)
    [end_c1, end_cr2, end_cr3] = backend(front_end_out, config)

    print(conv1.get_shape)
    print(end_c1.get_shape)
    print(end_cr2.get_shape)
    print(end_cr3.get_shape)

    return [conv1, end_c1, end_cr2, end_cr3] 

def frame_level_many(config, x_in):
    conv0 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=512,
                             strides=32,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    conv1 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=256,
                             strides=32,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())  

    conv2 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=128,
                             strides=32,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())  

    conv3 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=64,
                             strides=32,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())  

    conv4 = tf.layers.conv1d(inputs=x_in,
                             filters=config['CNN']['num_filters'],
                             kernel_size=32,
                             strides=32,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())  

    many = tf.concat([conv0, conv1, conv2, conv3, conv4], 2)
    front_end_out = tf.expand_dims(many, 3)

    [end_c1, end_cr2, end_cr3] = backend(front_end_out, config)

    print(x_in.get_shape)
    print(conv0.get_shape)
    print(conv1.get_shape)
    print(conv2.get_shape)
    print(conv3.get_shape)
    print(conv4.get_shape)
    print(end_c1.get_shape)
    print(end_cr2.get_shape)
    print(end_cr3.get_shape)

    return [conv0, conv1, conv2, conv3, conv4, end_c1, end_cr2, end_cr3] 


def cnn_audio(config, x_in):
   
    # remove some temporal filters to have the same ammount of timbral and temporal filters
    if config['CNN']['num_filters'] == 256:
        remove = 64  
    elif config['CNN']['num_filters'] == 128:
        remove = 32    
    elif config['CNN']['num_filters'] == 64:
        remove = 16
    elif config['CNN']['num_filters'] == 32:
        remove = 8
    elif config['CNN']['num_filters'] == 16:
        remove = 4
    elif config['CNN']['num_filters'] == 8:
        remove = 2
    elif config['CNN']['num_filters'] == 4:
        remove = 1

    # define the cnn_music model  
    with tf.name_scope('cnn_audio'):

        print('[AUDIO!] Input: ' + str(x_in.get_shape))

        input_layer = tf.reshape(x_in, [-1, config['CNN']['n_frames'], config['CNN']['n_mels'], 1])

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
        input_pad_3 = tf.pad(input_layer, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")

        # [TIMBRE] filter shape 1: 7x0.9f
        conv1 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[1, conv1.shape[2]],
                                    strides=[1, conv1.shape[2]])
        p1 = tf.squeeze(pool1, [2])

        # [TIMBRE] filter shape 2: 3x0.9f
        conv2 = tf.layers.conv2d(inputs=input_pad_3, 
                             filters=config['CNN']['num_filters']*2,
                             kernel_size=[3, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[1, conv2.shape[2]],
                                    strides=[1, conv2.shape[2]])
        p2 = tf.squeeze(pool2, [2])

        # [TIMBRE] filter shape 3: 1x0.9f
        conv3 = tf.layers.conv2d(inputs=input_layer, 
                             filters=config['CNN']['num_filters']*4,
                             kernel_size=[1, int(0.9 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[1, conv3.shape[2]],
                                    strides=[1, conv3.shape[2]])
        p3 = tf.squeeze(pool3, [2])

        # [TIMBRE] filter shape 3: 7x0.4f
        conv4 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=config['CNN']['num_filters'],
                             kernel_size=[7, int(0.4 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                    pool_size=[1, conv4.shape[2]],
                                    strides=[1, conv4.shape[2]])
        p4 = tf.squeeze(pool4, [2])

        # [TIMBRE] filter shape 5: 3x0.4f
        conv5 = tf.layers.conv2d(inputs=input_pad_3, 
                             filters=config['CNN']['num_filters']*2,
                             kernel_size=[3, int(0.4 * config['CNN']['n_mels'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[1, conv5.shape[2]],
                                    strides=[1, conv5.shape[2]])
        p5 = tf.squeeze(pool5, [2])

        # [TIMBRE] filter shape 6: 1x0.4f
        conv6 = tf.layers.conv2d(inputs=input_layer, 
                             filters=config['CNN']['num_filters']*4,
                             kernel_size=[1, int(0.4 * config['CNN']['n_mels'])], padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, conv6.shape[2]],
                                    strides=[1, conv6.shape[2]])
        p6 = tf.squeeze(pool6, [2])

        # [TEMPORAL-FEATURES] - average pooling + filter shape 7: 64x1
        pool7 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool7_rs = tf.squeeze(pool7, [3])
        conv7 = tf.layers.conv1d(inputs=pool7_rs,
                             filters=config['CNN']['num_filters']-remove,
                             kernel_size=64,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 8: 32x1
        pool8 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool8_rs = tf.squeeze(pool8, [3])
        conv8 = tf.layers.conv1d(inputs=pool8_rs,
                             filters=config['CNN']['num_filters']*2-remove,
                             kernel_size=32,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 9: 16x1
        pool9 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['CNN']['n_mels']],
                                        strides=[1, config['CNN']['n_mels']])
        pool9_rs = tf.squeeze(pool9, [3])
        conv9 = tf.layers.conv1d(inputs=pool9_rs,
                             filters=config['CNN']['num_filters']*4-remove,
                             kernel_size=16,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # [TEMPORAL-FEATURES] - average pooling + filter shape 10: 8x1
        pool10 = tf.layers.average_pooling2d(inputs=input_layer,
                                         pool_size=[1, config['CNN']['n_mels']],
                                         strides=[1, config['CNN']['n_mels']])
        pool10_rs = tf.squeeze(pool10, [3])
        conv10 = tf.layers.conv1d(inputs=pool10_rs,
                              filters=config['CNN']['num_filters']*8-remove,
                              kernel_size=8,
                              padding="same",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    # concatenate all feature maps
    timbral = tf.concat([p1, p2, p3, p4, p5, p6], 2)
    temporal = tf.concat([conv7, conv8, conv9, conv10], 2)

    print(timbral.get_shape)
    print(temporal.get_shape)

    # check [moving_mean, moving_variance, beta, gamma]
    #    [batch_normalization_8/moving_mean:0, batch_normalization_8/moving_variance:0,
    #    .. 'batch_normalization_8/beta', 'batch_normalization_8/gamma:0']
    #    sess.run('batch_normalization_8/moving_mean:0')

    return [timbral, temporal]
