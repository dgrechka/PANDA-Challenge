import tensorflow as tf

imageSize = 224


def constructModel(seriesLen, DORate=0.2, l2regAlpha = 1e-3):
    netInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="input")
    denseNet = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(imageSize, imageSize, 3),
        # it should have exactly 3 inputs channels,
        # and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value
        pooling=None)  # Tx8x8x1024 in case of None pooling
    denseNet.trainable = False

    converted = tf.keras.applications.densenet.preprocess_input(netInput)
    print("converted input shape {0}".format(converted.shape))
    cnnOut = tf.keras.layers.TimeDistributed(denseNet, name='cnns')(converted)  # Tx7x7x1024 in case of None pooling
    print("cnn out shape {0}".format(cnnOut.shape))
    cnnPooled = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((7, 7)), name='cnnsPooled')(
        cnnOut)  # Tx1x1x1024
    cnnPooledReshaped = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((1024,)), name='cnnsPooledReshaped')(
        cnnPooled)  # Tx1024
    cnnPooledReshapedDO = tf.keras.layers.Dropout(rate=DORate, name='cnnsPooledReshapedDO')(
        cnnPooledReshaped)  # Tx1024
    perSliceDenseOut = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128,
        activation="selu",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)), name='perSliceDenseOut')(
        cnnPooledReshapedDO)  # 128.   1024*128  parameters
    perSliceDenseOutDO = tf.keras.layers.Dropout(rate=DORate, name='perSliceDenseOutDO')(
        perSliceDenseOut)
    #gru1 = tf.keras.layers.GRU(128, return_sequences=True)
    #gru1back = tf.keras.layers.GRU(128, return_sequences=True, go_backwards=True)
    #gru1out = tf.keras.layers.Bidirectional(gru1, backward_layer=gru1back, name='rnn1')(perSliceDenseOutDO)
    #gru1outDO = tf.keras.layers.Dropout(rate=DORate, name='rnn1DO')(gru1out)

    #, batch_input_shape=(1, seriesLen, 128)
    # , implementation=1

    rnnOut = \
        tf.keras.layers.LSTM(
            32, dropout=DORate,
            kernel_regularizer = tf.keras.regularizers.L1L2(l2=l2regAlpha),
            recurrent_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha))(perSliceDenseOutDO)
    rnnOutDO = tf.keras.layers.Dropout(rate=DORate,name='rnn2DO')(rnnOut)
    predOut = \
        tf.keras.layers.Dense(1,name="resSigmoid",activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)
        )(rnnOutDO)
    predOutScaled = tf.keras.layers.Lambda(lambda x: x*5.0, name="result")(predOut)

    return tf.keras.Model(name="PANDA_A", inputs=netInput, outputs=predOutScaled)
