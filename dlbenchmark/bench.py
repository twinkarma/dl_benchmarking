import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import tensorflow_hub as hub


class VGGBench:

    def run_benchmark(self):
        input_data = np.random.rand(100, 224, 224, 3)
        labels = np.random.randint(0, 1000, size=[100])
        print(input_data.shape)
        print(labels.shape)
        print(labels)

        # Download model
        url = "https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/4"
        input = Input(shape=(224, 224, 3))
        flatten = tf.keras.layers.Flatten()
        dense = Dense(10)
        mnet_layer = hub.KerasLayer(url)
        softmax = tf.keras.layers.Softmax()

        y = softmax(dense(mnet_layer(input)))

        model = tf.keras.models.Model(inputs=input, outputs=y)
        # print(model.summary())

        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # model.save("testmodel2.tf")
        #
        #
        # m2 = tf.keras.models.load_model("testmodel2.tf")
        # m2.summary()
        #
        #
        # m2i = m2.inputs
        # m2o = m2.outputs
        # print(m2i)
        # print(m2o)
        model.summary()
        tf.keras.utils.plot_model(model, to_file="mplot.png", show_shapes=True)
        for layer in model.layers:
            print(layer.name)

        # tfmodel = hub.load(url)
        #
        # # Train model
        # history = model.fit(x=input_data, y=labels, batch_size=10, epochs=100)
        # print(history.history['loss'])




