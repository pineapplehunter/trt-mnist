import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import argparse

save_path = "saved"
save_path_trt = "saved_trt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--percision-mode", choices=["native", "FP32", "FP16", "INT8"], required=True
    )
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--calibration-cycles", type=int, default=10)
    opts = parser.parse_args()

    sample_size = opts.sample_size

    # the data, split between train and test sets
    (_x_train, _y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_test = np.expand_dims(x_test, -1)
    # print(x_test.shape[0], "test samples")

    x_test = x_test[0:sample_size]
    y_test = y_test[0:sample_size]

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, 10)

    if opts.percision_mode == "native":
        loaded = tf.saved_model.load(save_path)
        print(list(loaded.signatures.keys()))  # ["serving_default"]

        infer = loaded.signatures["serving_default"]
        print(infer.structured_outputs)
    else:
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(
            precision_mode=opts.percision_mode
        )
        print(conversion_params)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=save_path, conversion_params=conversion_params
        )

        def my_input_fn():
            for _ in range(opts.calibration_cycles):
                inp1 = x_test
                yield (inp1,)

        if opts.percision_mode == "INT8":
            converter.convert(my_input_fn)
        else:
            converter.convert()
        converter.build(input_fn=my_input_fn)
        converter.save(save_path_trt)

        loaded = tf.saved_model.load(save_path_trt)
        print(list(loaded.signatures.keys()))  # ["serving_default"]

        infer = loaded.signatures["serving_default"]
        print(infer.structured_outputs)

    # time!
    start = time.time()
    for _ in range(10000):
        output = infer(tf.constant(x_test[0:sample_size]))["dense"]
    end = time.time()

    output_argmax = tf.math.argmax(output, axis=1)
    y_argmax = tf.math.argmax(y_test, axis=1)

    print(
        "accuracy =",
        tf.reduce_mean(
            tf.cast(tf.math.equal(output_argmax, y_argmax), tf.float32)
        ).numpy(),
    )
    print("time:", end - start)


if __name__ == "__main__":
    main()
