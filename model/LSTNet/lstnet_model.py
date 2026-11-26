"""Kiến trúc LSTNet cho bài toán dự đoán ``delta = nextPrice - Price``.

Mô hình gồm 4 nhánh theo bài báo:

- CNN (Conv1D): bắt mẫu cục bộ ngắn hạn.
- GRU: phụ thuộc dài hạn trên toàn cửa sổ.
- Skip-GRU: các điểm cách nhau ``skip`` bước, bắt tính chu kỳ.
- AR (highway): hồi quy tuyến tính trên ``highway`` giá trị ``delta`` quá khứ.

Đầu ra là một giá trị ``delta`` đã scale; notebook sẽ ``inverse_transform``
rồi cộng với ``Price`` của ngày cuối cửa sổ.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


@tf.keras.utils.register_keras_serializable(package="lstnet")
class PreSkipTrans(tf.keras.layers.Layer):
    """Biến đổi đầu vào để GRU nhìn các điểm cách nhau ``skip`` bước."""

    def __init__(self, pt, skip, **kwargs):
        super().__init__(**kwargs)
        self.pt = int(pt)
        self.skip = int(skip)

    def call(self, inputs):
        x = inputs
        batch = tf.shape(x)[0]
        channels = x.shape[2]
        out = x[:, -self.pt * self.skip:, :]
        out = tf.reshape(out, [batch, self.pt, self.skip, channels])
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, [batch * self.skip, self.pt, channels])
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[0] = None
        shape[1] = self.pt
        return tf.TensorShape(shape)

    def get_config(self):
        config = {"pt": self.pt, "skip": self.skip}
        base = super().get_config()
        return {**base, **config}


@tf.keras.utils.register_keras_serializable(package="lstnet")
class PostSkipTrans(tf.keras.layers.Layer):
    """Ghép kết quả Skip-GRU trở lại kích thước batch ban đầu."""

    def __init__(self, skip, **kwargs):
        super().__init__(**kwargs)
        self.skip = int(skip)

    def call(self, inputs):
        x, original_input = inputs
        batch = tf.shape(original_input)[0]
        units = x.shape[1]
        out = tf.reshape(x, [batch, self.skip, units])
        out = tf.reshape(out, [batch, self.skip * units])
        return out

    def compute_output_shape(self, input_shape):
        # Layer nhận 2 input nên input_shape là list [shape_x, shape_gốc];
        # mỗi phần tử có thể là TensorShape hoặc tuple.
        x_shape = input_shape[0]
        units = x_shape[1] if not hasattr(x_shape, "as_list") else x_shape.as_list()[1]
        return tf.TensorShape([None, None if units is None else self.skip * units])

    def get_config(self):
        config = {"skip": self.skip}
        base = super().get_config()
        return {**base, **config}


def LSTNetModel(init, window, n_features):
    """Dựng model với 2 input: feature cửa sổ và lịch sử delta."""
    x_features = Input(shape=(window, n_features), name="features")
    x_delta = Input(shape=(init.highway, 1), name="delta_history")

    # --- CNN ---
    c = Conv1D(
        filters=init.CNNFilters,
        kernel_size=init.CNNKernel,
        activation="relu",
        kernel_initializer=init.initialiser,
        name="cnn",
    )(x_features)
    c = Dropout(init.dropout, name="dropout_cnn")(c)

    # --- GRU ---
    _, r = GRU(
        init.GRUUnits,
        activation="relu",
        return_state=True,
        name="gru",
    )(c)
    r = Dropout(init.dropout, name="dropout_gru")(r)

    # --- Skip-GRU ---
    if init.skip > 0 and c.shape[1] is not None:
        time_steps = int(c.shape[1])
        pt = time_steps // init.skip
        if pt >= 1:
            """
            Biến đổi đầu vào để GRU nhìn các điểm cách nhau ``skip`` bước.
            - Lấy ``pt * skip`` bước cuối của feature.
            - Reshape thành (batch, pt, skip, channels).
            - Transpose thành (batch, skip, pt, channels).
            - Reshape thành (batch * skip, pt, channels) để GRU chạy trên ``skip`` chuỗi song song.
            
            """
            s = PreSkipTrans(pt, init.skip, name="pre_skip")(c)
            _, s = GRU(
                init.SkipGRUUnits,
                activation="relu",
                return_state=True,
                name="skip_gru",
            )(s)
            s = PostSkipTrans(init.skip, name="post_skip")([s, x_features])
            r = Concatenate(axis=1, name="concat")([r, s])

    # --- Nhánh phi tuyến ---
    dense = Dense(1, kernel_initializer=init.initialiser, name="dense")(r)

    # --- AR (highway) trên delta quá khứ ---
    if init.highway > 0:
        ar = Flatten(name="ar_flatten")(x_delta)
        ar = Dense(1, kernel_initializer=init.initialiser, name="ar_dense")(ar)
        output = Add(name="output")([dense, ar])
    else:
        output = dense

    return tf.keras.Model(inputs=[x_features, x_delta], outputs=output, name="lstnet")


def ModelCompile(model, init):
    """Biên dịch model theo optimizer/loss trong cấu hình."""
    if init.optimiser == "SGD":
        optimizer = SGD(learning_rate=init.lr, momentum=0.0, nesterov=False)
    elif init.optimiser == "RMSprop":
        optimizer = RMSprop(learning_rate=init.lr, rho=0.9)
    else:
        optimizer = Adam(learning_rate=init.lr)

    model.compile(optimizer=optimizer, loss=init.loss, metrics=["mae"])
