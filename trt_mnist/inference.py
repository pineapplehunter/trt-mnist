import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
import logging

save_path = "saved"
save_path_trt = "saved_trt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_mnist_test_dataset(sample_size):
    (_x_train, _y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_test = np.expand_dims(x_test, -1)
    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, 10)

    x_test = x_test[0:sample_size]
    y_test = y_test[0:sample_size]

    return x_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--percision-mode", choices=["native", "FP32", "FP16", "INT8"], required=True
    )
    parser.add_argument("--sample-size", type=int, default=128)
    opts = parser.parse_args()

    x_test, y_test = get_mnist_test_dataset(opts.sample_size)

    if opts.percision_mode == "native":
        loaded = tf.saved_model.load(save_path)
        infer = loaded.signatures["serving_default"]

    else:
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=opts.percision_mode
        )
        print(conversion_params)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=save_path, conversion_params=conversion_params
        )

        def calib_fn():
            return (x_test,)

        if opts.percision_mode == "INT8":
            converter.convert(calib_fn)
        else:
            converter.convert()
        converter.save(save_path_trt)

        loaded = tf.saved_model.load(save_path_trt)
        infer = loaded.signatures["serving_default"]

    output = infer(tf.constant(x_test))["dense"]

    output_argmax = tf.math.argmax(output, axis=1)
    y_argmax = tf.math.argmax(y_test, axis=1)

    accuracy = tf.reduce_mean(
        tf.cast(tf.math.equal(output_argmax, y_argmax), tf.float32)
    )

    print("accuracy =", accuracy.numpy())


if __name__ == "__main__":
    main()
