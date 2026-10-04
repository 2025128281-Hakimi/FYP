"""
conformer_model.py
-------------------
Conformer (Convolution-augmented Transformer) architecture, matching
Section 2.4.3.3 and Section 3.5.4.2 of the proposal.

Structure per Conformer block (Gulati et al., 2020):
    input -> 1/2 FeedForward -> Multi-Head Self-Attention -> Convolution
          -> 1/2 FeedForward -> LayerNorm -> output
    (each sub-module wrapped in a residual connection)

This file gives you:
    - ConvSubsampling   : downsamples the input spectrogram/MFCC sequence
    - FeedForwardModule
    - MHSAModule
    - ConvolutionModule
    - ConformerBlock
    - build_conformer_model(...) : full model with a classification head
      (for predicting presentation quality: good / average / poor)

This is a from-scratch implementation (no external conformer package),
so it will run with just `tensorflow` installed - matching your software
requirements table (3.5.1).
"""

import tensorflow as tf
from tensorflow.keras import layers

# Where register_keras_serializable actually lives varies by TF/Keras
# version - on some installs it's tf.keras.saving (Keras 3 exposed
# through tf.keras), on others tf.keras doesn't have a .saving submodule
# at all and it's only on the standalone `keras` package, and on older
# TF/Keras 2 installs it's tf.keras.utils instead. Try each rather than
# hard-coding one path that breaks on a different environment.
try:
    import keras
    register_keras_serializable = keras.saving.register_keras_serializable
except (ImportError, AttributeError):
    try:
        register_keras_serializable = tf.keras.saving.register_keras_serializable
    except AttributeError:
        register_keras_serializable = tf.keras.utils.register_keras_serializable


# ---------------------------------------------------------------------------
# 1. Convolutional subsampling (reduces sequence length before the encoder)
# ---------------------------------------------------------------------------
@register_keras_serializable(package="conformer_model")
class ConvSubsampling(layers.Layer):
    """Two stride-2 Conv2D layers -> ~4x downsampling in the time dimension."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(d_model, kernel_size=3, strides=2, padding="same")
        self.conv2 = layers.Conv2D(d_model, kernel_size=3, strides=2, padding="same")
        self.act = layers.ReLU()
        self.dense = layers.Dense(d_model)

    def call(self, x):
        # x: (batch, time, feature) -> add channel dim for Conv2D
        x = tf.expand_dims(x, axis=-1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        b, t, f, c = x.shape
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], f * c))
        return self.dense(x)


# ---------------------------------------------------------------------------
# 2. Feed-forward module
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Multi-head self-attention module
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Convolution module (the part that makes a Conformer a Conformer)
# ---------------------------------------------------------------------------
@register_keras_serializable(package="conformer_model")
class ConvolutionModule(layers.Layer):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()
        self.pointwise_conv1 = layers.Conv1D(d_model * 2, kernel_size=1)
        self.glu = layers.Activation("sigmoid")  # used to gate half the channels (GLU)
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
        y = a * self.glu(b)  # Gated Linear Unit
        y = self.depthwise_conv(y)
        y = self.bn(y, training=training)
        y = self.swish(y)
        y = self.pointwise_conv2(y)
        y = self.dropout(y, training=training)
        return y


# ---------------------------------------------------------------------------
# 5. Full Conformer block
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 6. Full model: subsampling -> N conformer blocks -> pooling -> classifier
# ---------------------------------------------------------------------------
def build_conformer_model(
    input_shape: tuple,          # e.g. (time_steps, n_mfcc) -> (None, 40)
    num_classes: int = 3,        # e.g. good / average / poor
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
    # Quick shape sanity check - run this file directly to confirm the model builds
    model = build_conformer_model(input_shape=(None, 40), num_classes=3)
    model.summary()
