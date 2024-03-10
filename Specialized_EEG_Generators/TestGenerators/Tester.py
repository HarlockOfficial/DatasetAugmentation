import random
import sys

import numpy as np
import tensorflow as tf
from braindecode.preprocessing import create_windows_from_events

from DatasetAugmentation import utils

random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
np.random.RandomState(42)
seed = tf.random.normal([22, 500, 50])


def main(out_class, path_to_neural_network):
    if out_class.lower() == "none":
        out_class = None

    generator, is_generator_new = utils.eeg_generator(out_class, path_to_neural_network, return_is_new=True, is_training=False)
    discriminator, is_discriminator_new = utils.eeg_discriminator(out_class, path_to_neural_network, return_is_new=True, is_training=False)
    generator_loss, discriminator_loss, gan_loss = utils.eeg_gan_default_loss()
    generator_optimizer, discriminator_optimizer, gan_optimizer = utils.eeg_gan_default_optimizer()
    gan, is_gan_new = utils.eeg_gan_network(generator, discriminator, generator_loss, discriminator_loss, gan_loss, generator_optimizer, discriminator_optimizer, gan_optimizer, out_class, path_to_neural_network, return_is_new=True, is_training=False)

    assert not is_generator_new and not is_discriminator_new and not is_gan_new, "At least one of the models is new, please train the models first"

    testing_dataset = utils.load_dataset()
    windows_dataset = create_windows_from_events(
        testing_dataset,
        preload=True,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=500,
        window_stride_samples=50,
        drop_last_window=False,
        n_jobs=20,
    )
    if out_class is not None:
        dataset_by_label = utils.split_dataset_by_label(windows_dataset)
        testing_dataset = dataset_by_label[int(out_class)]

    dataset_element = random.choice(testing_dataset)
    prediction = discriminator.predict(dataset_element)
    print(f"Discriminator prediction for dataset element: {prediction}")

    prediction = generator.predict(seed)
    print(f"Generator prediction for seed: {prediction.shape}")
    prediction = np.reshape(prediction, (22, 500))
    generated_data = prediction
    prediction = discriminator.predict(prediction)
    print(f"Discriminator prediction for generator prediction: {prediction}")

    prediction = gan.predict(seed)
    print(f"GAN prediction for seed: {prediction}")

    print("All predictions are done!")

    print("Dataset Element:", dataset_element)
    print("Generated Data:", generated_data)




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: Tester.py <network_expected_output_class> <path_to_network_files_to_load>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
