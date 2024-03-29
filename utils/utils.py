import os

import moabb.datasets
import tensorflow as tf
from moabb.paradigms import MotorImagery
from typing_extensions import deprecated

OUTPUT_CLASSES = 3
SAMPLE_RATE = 128  # Hz (samples per second)
SECOND_DURATION = 0.5  # seconds

def load_dataset(moabb_dataset_class = moabb.datasets.BNCI2014001, subject_id = None, verbose: int = 0):
    """Load a dataset from MOABB.

    Args:
        moabb_dataset_class: The name of the dataset to load. Defaults to "BNCI2014_001".
        subject_id (Any, optional): The subject to load. If None, load all subjects. Defaults to None.
        verbose (int, optional): The verbosity level. Defaults to 0.

    Returns:
        [type]: [description]
    """

    ALL_EEG_CHANNELS = ['FC4', 'P3', 'CP2', 'Fp2', 'Fpz', 'PO4', 'Fp1', 'F3', 'CP3', 'Fz', 'Pz', 'F1', 'AF4', 'CP1',
                        'PO3', 'Cz', 'FC1', 'F4', 'P1', 'O1', 'F8', 'CP6', 'POz', 'FC5', 'FT8', 'P4', 'T8', 'CP4', 'F6',
                        'O2', 'C1', 'Oz', 'C2', 'P6', 'C4', 'F2', 'F5', 'PO7', 'C3', 'FC2', 'FC3', 'TP7', 'P5', 'C5',
                        'T7', 'C6', 'TP8', 'P8', 'FT7', 'CPz', 'AF3', 'FC6', 'P7', 'F7', 'PO8', 'CP5', 'P2', 'FCz']
    INPUT_CHANNELS = len(ALL_EEG_CHANNELS)
    print("Using the ", INPUT_CHANNELS, "Channels:", ALL_EEG_CHANNELS)

    paradigm = MotorImagery(channels=ALL_EEG_CHANNELS, events=['left_hand', 'right_hand', 'feet'],
                            n_classes=OUTPUT_CLASSES, fmin=0.5, fmax=40, tmin=0, tmax=SECOND_DURATION,
                            resample=SAMPLE_RATE, )

    x, y, _ = paradigm.get_data(moabb_dataset_class(), subjects=subject_id)
    if verbose>0:
        print("X Shape:", x.shape)
        print("Y Shape:", y.shape)
    return x, y

def __to_mV(x):
    return x * 1e6

def preprocess_dataset(local_dataset, n_jobs: int = 20):
    from braindecode.preprocessing import exponential_moving_standardize
    from braindecode.preprocessing import Preprocessor, preprocess

    low_cut_hz = 0.5  # low cut frequency for filtering
    high_cut_hz = 40.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    min_sfreq = min([ds.raw.info['sfreq'] for ds in local_dataset.datasets])
    print("Dataset Frequency:", min_sfreq)
    preprocessors = [# keep only EEG sensors
        Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False),
        # convert from volt to microvolt, directly modifying the numpy array
        Preprocessor(fn=__to_mV),
        # bandpass filter
        Preprocessor(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz), # exponential moving standardization
        Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size),
        Preprocessor(fn='resample', sfreq=min_sfreq)]

    # Preprocess the data
    preprocess(local_dataset, preprocessors, n_jobs=n_jobs)
    return local_dataset

def split_dataset_by_label(x, y, verbose: int = 0):
    """
    Split the dataset by label.
    """
    dataset_by_label = dict()
    for x, Y in zip(x, y):
        if Y not in dataset_by_label.keys():
            dataset_by_label[Y] = []
        dataset_by_label[Y].append(x)
    if verbose>0:
        print(dataset_by_label.keys())

    return dataset_by_label

def two_layer_blstm_with_dropout(network_input):
    bi_lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(65, activation='tanh', return_sequences=True))(
        network_input)
    bi_lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(65, activation='tanh', return_sequences=True))(
        bi_lstm_1)
    dropout = tf.keras.layers.Dropout(0.5)(bi_lstm_2)
    return dropout

def eeg_generator(generator_index: str = None, base_path: str = './', return_is_new: bool = False, is_training: bool = True, graph: bool = False):
    # if generator.h5 exists, load it
    if generator_index is not None:
        if os.path.exists(base_path + 'generator_' + generator_index + '/'):
            print("Loading Model from File for Generator")
            model = tf.keras.layers.TFSMLayer(f"{base_path}/generator_{generator_index}/",
                                                        call_endpoint='serving_default')
            if return_is_new:
                return model, False
            return model
    if os.path.exists(base_path + 'generator.h5'):
        print("Loading Model from File for Generator")
        model = tf.keras.layers.TFSMLayer(f"{base_path}/generator/",
                                                    call_endpoint='serving_default')
        if return_is_new:
            return model, False
        return model

    # Loss function: Categorical Cross-Entropy
    network_input = tf.keras.layers.Input(shape=(58, 65))
    dropout = two_layer_blstm_with_dropout(network_input)
    # dropout2 = two_layer_blstm_with_dropout(dropout)
    output = tf.keras.layers.Dense(65, activation='tanh')(dropout)
    model = tf.keras.Model(inputs=[network_input], outputs=[output], name='generator')
    if return_is_new:
        return model, True
    return model


def eeg_discriminator(discriminator_index: str = None, base_path: str = './', return_is_new: bool = False, is_training: bool = True, graph: bool = False):
    # if discriminator.h5 exists, load it
    if discriminator_index is not None:
        if os.path.exists(base_path + 'discriminator_' + discriminator_index + '/'):
            print("Loading Model from File for Discriminator")
            model = tf.keras.layers.TFSMLayer(f"{base_path}/discriminator_{discriminator_index}/",
                                                        call_endpoint='serving_default')
            if return_is_new:
                return model, False
            return model

    if os.path.exists(base_path + 'discriminator.h5'):
        print("Loading Model from File for Discriminator")
        model = tf.keras.layers.TFSMLayer(f"{base_path}/discriminator/",
                                                    call_endpoint='serving_default')
        if return_is_new:
            return model, False
        return model

    # Loss function: Categorical Cross-Entropy
    network_input = tf.keras.layers.Input(shape=(58,65))
    dropout = two_layer_blstm_with_dropout(network_input)
    flatten = tf.keras.layers.Flatten()(dropout)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
    model = tf.keras.Model(inputs=[network_input], outputs=[output], name='discriminator')
    if return_is_new:
        return model, True
    return model

def discriminator_loss(real_output, predicted_output):
    return tf.keras.losses.binary_crossentropy(real_output, predicted_output, from_logits=True)

def generator_loss(true_output, predicted_output):
    return tf.keras.losses.binary_crossentropy(true_output, predicted_output, from_logits=True)

@deprecated("Do not use!")
def eeg_gan_network(generator, discriminator, discriminator_optimizer, gan_optimizer, gan_index: str = None, base_path: str = './', return_is_new: bool = False, is_training: bool = True, graph: bool = False):
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

    discriminator.compile(optimizer=discriminator_optimizer, loss=discriminator_loss, run_eagerly=not graph)
    gan = tf.keras.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False
    gan.compile(optimizer=gan_optimizer, loss=generator_loss, metrics=['mae'], run_eagerly=not graph)
    if return_is_new:
        return gan, True
    return gan

def eeg_gan_default_optimizer():
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    return generator_optimizer, discriminator_optimizer

