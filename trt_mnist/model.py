from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout


def build_model(summary: bool = True) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    if summary:
        model.summary()

    return model
