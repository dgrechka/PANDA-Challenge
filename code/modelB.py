import tensorflow as tf

imageSize = 224


def constructModel(seriesLen, DORate=0.2):
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
    cnnTempPooled = tf.keras.layers.GlobalMaxPool1D(name="TempMapPool")(cnnPooledReshaped) # 1024
    cnnTempPooledDO = tf.keras.layers.Dropout(rate=DORate, name='TempMapPoolDO')(
        cnnTempPooled) 
    denseOut1 = tf.keras.layers.Dense(256, name='DenseOut1', activation="selu")(
        cnnTempPooledDO)  # 256
    denseOut1DO = tf.keras.layers.Dropout(rate=DORate, name='DenseOut1DO')(
        denseOut1)
    predOut = tf.keras.layers.Dense(1,name="resSigmoid",activation="sigmoid")(denseOut1DO)
    predOutScaled = tf.keras.layers.Lambda(lambda x: x*5.0, name="result")(predOut)

    return tf.keras.Model(name="PANDA_B", inputs=netInput, outputs=predOutScaled)
