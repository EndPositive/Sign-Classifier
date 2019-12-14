import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(self, n_classes=43, input_shape=(32, 32, 3), channels_format='channels_last'):
        super(Classifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 7, input_shape=input_shape, activation='relu',
                                            data_format=channels_format, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, 5, input_shape=input_shape, activation='relu',
                                            data_format=channels_format, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, input_shape=input_shape, activation='relu',
                                            data_format=channels_format, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
