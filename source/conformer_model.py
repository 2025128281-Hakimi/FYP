import tensorflow as tf
from keras import layers

try:
    import keras
    register_keras_serializable = keras.saving.register_keras_serializable
except (ImportError, AttributeError):
    try:
        register_keras_serializable = tf.keras.saving.register_keras_serializable
    except AttributeError:
        register_keras_serializable = tf.keras.utils.register_keras_serializable


@register_keras_serializable(package="conformer_model")
class ConvSubsampling(layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(d_model, kernel_size=3, strides=2, padding="same")
        self.conv2 = layers.Conv2D(d_model, kernel_size=3, strides=2, padding="same")
        self.act = layers.ReLU()
        self.dense = layers.Dense(d_model)

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        _, _, f, c = x.shape
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], f * c))
        return self.dense(x)


@register_keras_serializable(package="conformer_model")
class FeedForwardModule(layers.Layer):
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()
        self.dense1 = layers.Dense(d_model * expansion_factor, activation="swish")
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(d_model)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        y = self.ln(x)
        y = self.dense1(y)
        y = self.dropout1(y, training=training)
        y = self.dense2(y)
        y = self.dropout2(y, training=training)
        return y


@register_keras_serializable(package="conformer_model")
class MHSAModule(layers.Layer):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()
        self.mhsa = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        y = self.ln(x)
        y = self.mhsa(y, y, training=training)
        y = self.dropout(y, training=training)
        return y


@register_keras_serializable(package="conformer_model")
class ConvolutionModule(layers.Layer):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()
        self.pointwise_conv1 = layers.Conv1D(d_model * 2, kernel_size=1)
        self.glu = layers.Activation("sigmoid")
        self.depthwise_conv = layers.SeparableConv1D(
            d_model, kernel_size=kernel_size, padding="same"
        )
        self.bn = layers.BatchNormalization()
        self.swish = layers.Activation("swish")
        self.pointwise_conv2 = layers.Conv1D(d_model, kernel_size=1)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        y = self.ln(x)
        y = self.pointwise_conv1(y)
        a, b = tf.split(y, num_or_size_splits=2, axis=-1)
        y = a * self.glu(b)
        y = self.depthwise_conv(y)
        y = self.bn(y, training=training)
        y = self.swish(y)
        y = self.pointwise_conv2(y)
        y = self.dropout(y, training=training)
        return y


@register_keras_serializable(package="conformer_model")
class ConformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int = 4, ff_expansion: int = 4,
                 conv_kernel_size: int = 31, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MHSAModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_ln = layers.LayerNormalization()

    def call(self, x, training=False):
        x = x + 0.5 * self.ff1(x, training=training)
        x = x + self.mhsa(x, training=training)
        x = x + self.conv(x, training=training)
        x = x + 0.5 * self.ff2(x, training=training)
        return self.final_ln(x)


def build_conformer_model(
    input_shape: tuple,
    num_classes: int = 3,
    d_model: int = 144,
    num_blocks: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    x = ConvSubsampling(d_model)(inputs)
    for i in range(num_blocks):
        x = ConformerBlock(d_model, num_heads, dropout=dropout, name=f"conformer_block_{i}")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d_model, activation="swish")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="quality_output")(x)

    model = tf.keras.Model(inputs, outputs, name="presentation_conformer")
    return model


if __name__ == "__main__":
    model = build_conformer_model(input_shape=(None, 40), num_classes=3)
    model.summary()
