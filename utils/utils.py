import os

from braindecode.datasets import MOABBDataset
import tensorflow as tf


def load_dataset(moabb_dataset_name: str = "BNCI2014_001", subject_id = None, verbose: int = 0):
    """Load a dataset from MOABB.

    Args:
        moabb_dataset_name (str, optional): The name of the dataset to load. Defaults to "BNCI2014_001".
        subject_id (Any, optional): The subject to load. If None, load all subjects. Defaults to None.
        verbose (int, optional): The verbosity level. Defaults to 0.

    Returns:
        [type]: [description]
    """
    dataset = MOABBDataset(moabb_dataset_name, subject_ids=subject_id)
    if verbose>0:
        print(dataset.description)
    return dataset

def preprocess_dataset(local_dataset):
    from braindecode.preprocessing import exponential_moving_standardize
    from braindecode.preprocessing import Preprocessor, preprocess

    low_cut_hz = 4.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    min_sfreq = min([ds.raw.info['sfreq'] for ds in local_dataset.datasets])
    print("Dataset Frequency:", min_sfreq)
    preprocessors = [# keep only EEG sensors
        Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False),
        # convert from volt to microvolt, directly modifying the numpy array
        Preprocessor(fn=lambda x: x * 1e6), # bandpass filter
        Preprocessor(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz), # exponential moving standardization
        Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size),
        Preprocessor(fn='resample', sfreq=min_sfreq)]

    # Preprocess the data
    preprocess(local_dataset, preprocessors, n_jobs=20)
    return local_dataset


def split_dataset_by_label(local_dataset, verbose: int = 0):
    """
    Split the dataset by label.
    """
    dataset_by_label = dict()
    for x, Y, _ in local_dataset:
        if Y not in dataset_by_label.keys():
            dataset_by_label[Y] = []
        dataset_by_label[Y].append(x)
    if verbose>0:
        print(dataset_by_label.keys())
        print(type(dataset_by_label[0]))

    return dataset_by_label

def two_layer_blstm_with_dropout(network_input):
    bi_lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, activation='tanh', return_sequences=True))(
        network_input)
    bi_lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(30, activation='tanh', return_sequences=True))(
        bi_lstm_1)
    dropout = tf.keras.layers.Dropout(0.5)(bi_lstm_2)
    return dropout

def eeg_generator(generator_index: str = None, base_path: str = './', return_is_new: bool = False, is_training: bool = True):
    # if generator.h5 exists, load it
    if generator_index is not None:
        if os.path.exists(base_path + 'generator_' + generator_index + '.h5'):
            print("Loading Model from File for Generator")
            model = tf.keras.models.load_model(base_path + 'generator_' + generator_index + '.h5', compile=is_training)
            if return_is_new:
                return model, False
            return model
    if os.path.exists(base_path + 'generator.h5'):
        print("Loading Model from File for Generator")
        model = tf.keras.models.load_model(base_path + 'generator.h5', compile=is_training)
        if return_is_new:
            return model, False
        return model

    # Loss function: Categorical Cross-Entropy
    network_input = tf.keras.layers.Input(shape=(500, 50))
    dropout = two_layer_blstm_with_dropout(network_input)
    dropout2 = two_layer_blstm_with_dropout(dropout)
    output = tf.keras.layers.Dense(1, activation='tanh')(dropout2)
    model = tf.keras.Model(inputs=[network_input], outputs=[output], name='generator')
    if return_is_new:
        return model, True
    return model


def eeg_discriminator(discriminator_index: str = None, base_path: str = './', return_is_new: bool = False, is_training: bool = True):
    # if discriminator.h5 exists, load it
    if discriminator_index is not None:
        if os.path.exists(base_path + 'discriminator_' + discriminator_index + '.h5'):
            print("Loading Model from File for Discriminator")
            model = tf.keras.models.load_model(base_path + 'discriminator_' + discriminator_index + '.h5', compile=is_training)
            if return_is_new:
                return model, False
            return model

    if os.path.exists(base_path + 'discriminator.h5'):
        print("Loading Model from File for Discriminator")
        model = tf.keras.models.load_model(base_path + 'discriminator.h5', compile=is_training)
        if return_is_new:
            return model, False
        return model

    # Loss function: Categorical Cross-Entropy
    network_input = tf.keras.layers.Input(shape=(500,))
    reshape = tf.keras.layers.Reshape((500, 1))(network_input)
    dropout = two_layer_blstm_with_dropout(reshape)
    flatten = tf.keras.layers.Flatten()(dropout)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
    model = tf.keras.Model(inputs=[network_input], outputs=[output], name='discriminator')
    if return_is_new:
        return model, True
    return model

def eeg_gan_network(generator, discriminator, generator_loss, discriminator_loss, gan_loss, generator_optimizer, discriminator_optimizer, gan_optimizer, gan_index: str = None, base_path: str = './', return_is_new: bool = False, is_training: bool = True):
    if gan_index is not None:
        if os.path.exists(base_path + 'gan_' + gan_index + '.h5'):
            print("Loading Model from File for GAN")
            model = tf.keras.models.load_model(base_path + 'gan_' + gan_index + '.h5', compile=is_training)
            if return_is_new:
                return model, False
            return model

    if os.path.exists(base_path + 'gan.h5'):
        print("Loading Model from File for GAN")
        model = tf.keras.models.load_model(base_path + 'gan.h5', compile=is_training)
        if return_is_new:
            return model, False
        return model

    discriminator.compile(optimizer=discriminator_optimizer, loss=discriminator_loss)
    gan = tf.keras.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False
    gan.compile(optimizer=gan_optimizer, loss=gan_loss, metrics=['mae'])
    if return_is_new:
        return gan, True
    return gan

def eeg_gan_default_loss():
    generator_loss = lambda x: tf.keras.losses.binary_crossentropy(tf.ones_like(x), x)
    discriminator_loss = tf.keras.losses.binary_crossentropy
    gan_loss = tf.keras.losses.binary_crossentropy
    return generator_loss, discriminator_loss, gan_loss

def eeg_gan_default_optimizer():
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    gan_optimizer = tf.keras.optimizers.Adam(1e-4)
    return generator_optimizer, discriminator_optimizer, gan_optimizer

