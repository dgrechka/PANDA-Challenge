import tensorflow as tf

imageSize = 244


def constructModel(seriesLen):
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

    cnnOut = tf.keras.layers.TimeDistributed(denseNet, name='cnns')(converted)  # Tx8x8x1024 in case of None pooling
    cnnPooled = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((8, 8)), name='cnnsPooled')(
        cnnOut)  # Tx1x1x1024
    cnnPooledReshaped = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((1024,)), name='cnnsPooledReshaped')(
        cnnPooled)  # Tx1024
    perSliceDenseOut = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256), name='perSliceDenseOut')(
        cnnPooledReshaped)  # Tx256.   1024*256 = 262144 parameters
    gru1 = tf.keras.layers.GRU(128, return_sequences=True)
    gru1back = tf.keras.layers.GRU(128, return_sequences=True, go_backwards=True)
    gru1out = tf.keras.layers.Bidirectional(gru1, backward_layer=gru1back, name='rnn1')(perSliceDenseOut)
    gru2 = tf.keras.layers.GRU(64)
    gru2back = tf.keras.layers.GRU(64, go_backwards=True)
    gru2out = tf.keras.layers.Bidirectional(gru2, backward_layer=gru2back, name='rnn2')(gru1out)
    predOut = tf.keras.layers.Dense(1,name="resSigmoid",activation="sigmoid")(gru2out)
    predOutScaled = tf.keras.layers.Lambda(lambda x: x*5.0, name="result")(predOut)

    return tf.keras.Model(name="PANDA_A", inputs=netInput, outputs=predOutScaled)
