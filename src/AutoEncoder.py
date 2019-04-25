import numpy as np
import keras
import sklearn.metrics as metrics
from keras.layers import Input, Dense, \
    Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, \
    UpSampling1D, UpSampling2D, \
    Reshape, Flatten, Dropout, concatenate
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import TensorBoard
import keras.backend as K


class AE:
    def __init__(self, input_layer, encode_layer, decode_layer, loss):
        encoder = Model(input_layer, encode_layer)
        autoencoder = Model(input_layer, decode_layer)

        # decoder_input = Input(shape=encode_layer._keras_shape)
        # decoder_h = decoder_input
        # for l in decode_layers:
        #    decoder_h = l(decoder_h)
        # decoder = Model(decoder_input, decoder_h)

        encoder.compile(optimizer='adadelta', loss=loss)
        autoencoder.compile(optimizer='adadelta', loss=loss)
        # decoder.compile(optimizer='adadelta', loss=loss)

        self.encode_layer = encode_layer
        self.loss = loss

        self.encoder = encoder
        self.autoencoder = autoencoder
        # self.decoder = decoder

        self.autoencoder.summary()

    def fit_(self, X, y, epochs=500, batch_size=1000, verbose=1):
        self.history = \
            self.autoencoder.fit(X, y,
                                 epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history

    def predict_raw(self, X):
        return self.autoencoder.predict(X)

    def encode(self, X, output_argmax=False):
        encoded = self.encoder.predict(X)
        return encoded.argmax(axis=-1) if output_argmax else encoded

    # def decode(self, X):
    #     return self.decoder.predict(X)

    def save(self, path, name):
        self.autoencoder.save(path + '/' + name + '_ae.h5')
        self.encoder.save(path + '/' + name + '_enc.h5')

    def load(self, path, name):
        self.autoencoder = load_model(path + '/' + name + '_ae.h5')
        self.encoder = load_model(path + '/' + name + '_enc.h5')
        # self.decoder = load_model(path + '/' + name + '_dec.h5')

    def save_weights(self, path, name):
        self.autoencoder.save_weights(path + '/' + name + '_weights.h5')

    def load_weights(self, path, name):
        self.autoencoder.load_weights(path + '/' + name + '_weights.h5')


class Standard(AE):

    def __init__(self, dims,
                 input_shape,
                 input_dropout=0.3,
                 activation_hidden='relu',
                 activation_encoder='relu',
                 activation_output='linear',
                 loss_balance_t=0.5,
                 loss_confidence_t=0.5,
                 loss_type='mse'):
        dims = np.array(dims)
        n_layers = dims.shape[0]
        input_data = Input(shape=input_shape)

        flatten = None
        if (len(input_shape) > 1):
            flatten = Flatten()(Dropout(input_dropout)(input_data))
        else:
            flatten = Dropout(input_dropout)(input_data)
        h = flatten

        if (n_layers > 1):
            for n in dims[:n_layers-1]:
                h = Dense(n, activation=activation_hidden)(h)

        encoded = Dense(dims[n_layers - 1], activation=activation_encoder)(h)

        h = encoded

        if (n_layers > 1):
            for n in np.flip(dims[0:n_layers-1]):
                h = Dense(n, activation=activation_hidden)(h)

        decoded = Dense(
            flatten._keras_shape[-1], activation=activation_output)(h)
        decoded = Reshape(input_shape)(decoded)

        super().__init__(input_data, encoded, decoded, loss_type)


class Conv_2D(AE):
    def __init__(self, dims,
                 input_shape,
                 input_dropout=0.3,
                 activation_hidden='relu',
                 activation_encoder='relu',
                 activation_output='linear',
                 loss_type='mse'):

        dims = np.array(dims)
        n_layers = dims.shape[0]
        h_layers = []
        input_data = Input(shape=input_shape)
        h = Dropout(0.3)(input_data)

        if (n_layers > 1):
            for n in dims[:n_layers-1]:
                h = Conv2D(n, kernel_size=3, padding='same',
                           activation=activation_hidden)(h)
                h = MaxPooling2D()(h)
                h_layers.append(h)

        flatten_h = Flatten()(h)
        encoded = Dense(dims[n_layers - 1],
                        activation=activation_encoder)(Flatten()(h))

        h = Reshape(h._keras_shape[1:])(
            Dense(flatten_h._keras_shape[-1])(encoded))
        if (n_layers > 1):
            for n in np.flip(dims[:n_layers-1]):
                h = UpSampling2D()(h)
                h = Conv2D(n, kernel_size=3,
                           activation=activation_hidden, padding='same')(h)

        decoded = Conv2D(1, kernel_size=1,
                         activation=activation_hidden, padding='same')(h)

        super().__init__(input_data, encoded, decoded, loss_type)


class Conv_2D_Clustering(AE):
    def __init__(self, dims,
                 input_shape,
                 n_sideencoder=5,
                 input_dropout=0.3,
                 activation_hidden='relu',
                 activation_encoder='relu',
                 activation_output='linear',
                 activation_sideencoder='sigmoid',
                 loss_balance_t=0.5,
                 loss_confidence_t=0.5,
                 loss_side_kernel_t=0.5,
                 loss_side_activation_t=0.5):

        dims = np.array(dims)
        n_layers = dims.shape[0]
        h_layers = []
        input_data = Input(shape=input_shape)
        h = Dropout(0.3)(input_data)

        if (n_layers > 1):
            for n in dims[:n_layers-1]:
                h = Conv2D(n, kernel_size=3, padding='same',
                           activation=activation_hidden)(h)
                h = MaxPooling2D()(h)
                h_layers.append(h)

        flatten_h = Flatten()(h)

        encoded = Dense(dims[n_layers - 1],
                        activation=activation_encoder)(Flatten()(h))

        h = Reshape(h._keras_shape[1:])(
            Dense(flatten_h._keras_shape[-1])(encoded))
        if (n_layers > 1):
            for n in np.flip(dims[:n_layers-1]):
                h = UpSampling2D()(h)
                h = Conv2D(n, kernel_size=3,
                           activation=activation_hidden, padding='same')(h)

        decoded = Conv2D(1, kernel_size=1,
                         activation=activation_hidden, padding='same')(h)

        def loss(y_true, y_pred):

            # encoded.shape = (n, 10)
            n_encoded = dims[n_layers-1]

            # confidence (n, 1)
            confidence_loss = (1 - K.max(encoded, axis=-1)) * loss_confidence_t

            # activation_agg (1, 10)
            activation_agg = K.mean(encoded, axis=0)
            balance_loss_ = K.mean(K.abs(1/n_encoded - activation_agg))
            balance_loss = balance_loss_ * loss_balance_t

            mse_loss = K.mean(K.mean(
                keras.losses.mse(y_true, y_pred),
                axis=-1), axis=-1)

            return mse_loss * (1 + confidence_loss + balance_loss)

        super().__init__(input_data, encoded, decoded, loss)


class Clustering(AE):

    def __init__(self, dims,
                 input_shape,
                 input_dropout=0.3,
                 activation_hidden='relu',
                 activation_encoder='softmax',
                 activation_output='linear',
                 loss_balance_t=0.5,
                 loss_confidence_t=0.5):

        dims = np.array(dims)
        n_layers = dims.shape[0]
        input_data = Input(shape=input_shape)

        flatten = None
        if (len(input_shape) > 1):
            flatten = Flatten()(Dropout(input_dropout)(input_data))
        else:
            flatten = Dropout(input_dropout)(input_data)
        h = flatten

        if (n_layers > 1):
            for n in dims[:n_layers-1]:
                h = Dense(n, activation=activation_hidden)(h)

        encoded = Dense(dims[n_layers - 1], activation=activation_encoder)(h)
        h = encoded

        if (n_layers > 1):
            for n in np.flip(dims[:n_layers-1]):
                h = Dense(n, activation=activation_hidden)(h)

        decoded = Dense(
            flatten._keras_shape[-1], activation=activation_output)(h)
        decoded = Reshape(input_shape)(decoded)

        def loss(y_true, y_pred):

            # encoded.shape = (n, 10)
            n_encoded = dims[n_layers-1]

            # confidence (n, 1)
            confidence_loss = (1 - K.max(encoded, axis=-1)) * loss_confidence_t

            # activation_agg (1, 10)
            activation_agg = K.mean(encoded, axis=0)
            balance_loss_ = K.mean(K.abs(1/n_encoded - activation_agg))
            balance_loss = balance_loss_ * loss_balance_t

            mse_loss = K.mean(keras.losses.mse(y_true, y_pred), axis=-1)

            # Centroid of each cluster
            # centroids = K.sum(
            #    K.expand_dims(y_true, axis=1) *
            #    K.expand_dims(encoded, axis=-1) *
            #    K.expand_dims(encoded, axis=-1),
            #    axis=0)
            # centroid_dist_matrix = K.sqrt(K.sum(K.square(
            #    K.repeat_elements(
            #        K.expand_dims(centroids, axis=0), n_encoded, axis=0
            #    ) -
            #    K.repeat_elements(
            #        K.expand_dims(centroids, axis=1), n_encoded, axis=1)
            # ), axis=-1))

            # centroid_dist_loss = 1 / K.mean(centroid_dist_matrix)

            return mse_loss * (1 + confidence_loss + balance_loss)
            # + loss_centroid_dist_t * centroid_dist_loss)

        super().__init__(input_data, encoded, decoded, loss)


class Duet(AE):
    def __init__(self, dims,
                 input_shape,
                 input_dropout=0.3,
                 n_sideencoder=10,
                 activation_hidden='relu',
                 activation_encoder='softmax',
                 activation_sideencoder='sigmoid',
                 activation_output='linear',
                 side_sparseness=10e-5,
                 loss_balance_t=0.5,
                 loss_confidence_t=0.5,
                 loss_side_kernel_t=0.5,
                 loss_side_activation_t=0.5
                 ):

        dims = np.array(dims)
        n_layers = dims.shape[0]
        input_data = Input(shape=input_shape)

        flatten = None
        if (len(input_shape) > 1):
            flatten = Flatten()(Dropout(input_dropout)(input_data))
        else:
            flatten = Dropout(input_dropout)(input_data)

        h = flatten

        if (n_layers > 1):
            for n in dims[:n_layers-1]:
                h = Dense(n, activation=activation_hidden)(h)

        encoded = Dense(dims[n_layers - 1], activation=activation_encoder)(h)

        side_encoded = Dense(
            n_sideencoder,
            activation=activation_sideencoder,
            activity_regularizer=keras.regularizers.l1(side_sparseness))(h)

        h = encoded
        h = concatenate([side_encoded, encoded])

        def l2_on_side(weight_matrix):
            return loss_side_kernel_t * K.sum(K.square(weight_matrix[:n_sideencoder]))

        if (n_layers > 1):
            h = Dense(dims[n_layers-2], activation=activation_hidden,
                      kernel_regularizer=l2_on_side)(h)
            for n in np.flip(dims[0:n_layers-2]):
                h = Dense(n, activation=activation_hidden)(h)
            decoded = Dense(
                flatten._keras_shape[-1], activation=activation_output)(h)
            decoded = Reshape(input_shape)(decoded)

        else:
            decoded = Dense(
                flatten._keras_shape[-1], activation=activation_output,
                kernel_regularizer=l2_on_side
            )(h)
            decoded = Reshape(input_shape)(decoded)

        def loss(y_true, y_pred):
            n_encoded = dims[n_layers-1]
            confidence_loss = (1 - K.max(encoded, axis=-1)) * loss_confidence_t
            activation_agg = K.mean(encoded, axis=0)
            balance_loss_ = K.mean(K.abs(1/n_encoded - activation_agg))
            balance_loss = balance_loss_ * loss_balance_t
            mse_loss = K.mean(keras.losses.mse(y_true, y_pred), axis=-1)
            return mse_loss * (1 + confidence_loss + balance_loss)

        super().__init__(input_data, encoded, decoded, loss)
        side = Model(input_data, side_encoded)
        side.compile('adadelta', loss=loss)
        self.side = side

    def encode_side(self, X, output_argmax=False):
        return self.side.predict(X)
