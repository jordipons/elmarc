import tensorflow as tf

def cnn_small_filters(config, x_in):

    with tf.name_scope('cnn_as_choi'):

        print('Input: ' + str(x_in.get_shape))

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

        print('Input: ' + str(x_in.get_shape))

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
