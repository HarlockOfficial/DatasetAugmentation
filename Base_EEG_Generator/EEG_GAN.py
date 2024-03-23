# Network presented in: https://pure.ulster.ac.uk/en/publications/mieeg-gan-generating-artificial-motor-imagery-electroencephalogra
import os
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events

from DatasetAugmentation import utils

seed = tf.random.normal([22, 500, 50])
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
np.random.RandomState(42)


def train(training_dataset, generator, discriminator, gan, steps_per_epoch, epochs, batch_size, checkpoint, checkpoint_prefix):
    avg_fake_input_loss_discriminator = []
    avg_real_input_loss_discriminator = []
    avg_loss_discriminator = []
    avg_loss_generator = []
    training_dataset_list = list(training_dataset)

    # main train loop
    for epoch in range(200):
        print("Starting Epoch:", epoch + 1, "of", epochs)
        start = time()

        fake_input_loss_discriminator = []
        real_input_loss_discriminator = []
        loss_discriminator = []
        loss_generator = []
        for step in range(steps_per_epoch):
            print("Starting Step:", step + 1, "of", steps_per_epoch)
            # Train discriminator
            for iteration in range(steps_per_epoch):
                print("Starting Discriminator Train Iteration:", iteration + 1, "of", steps_per_epoch)
                real_data = random.choice(training_dataset_list)[0]
                noise = tf.random.normal([batch_size, 500, 50])
                fake_data = generator.predict(noise)
                discriminator_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
                fake_data = np.reshape(fake_data, (batch_size, 500))
                discriminator_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

            # Train generator
            generator_loss = 0
            y = np.ones((batch_size, 1))
            for iteration in range(steps_per_epoch):
                print("Starting GAN Train Iteration:", iteration + 1, "of", steps_per_epoch)
                noise = tf.random.normal([batch_size, 500, 50])
                generator_loss += gan.train_on_batch(noise, y)[0]
            fake_input_loss_discriminator.append(discriminator_loss_fake)
            real_input_loss_discriminator.append(discriminator_loss_real)
            loss_discriminator.append(0.5 * np.add(discriminator_loss_real, discriminator_loss_fake))
            loss_generator.append(generator_loss/steps_per_epoch)
        epoch_loss_generator = np.mean(loss_generator)
        # Save loss
        if len(avg_loss_generator) == 0 or epoch_loss_generator < min(avg_loss_generator):
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save('./generator.keras')
            discriminator.save('./discriminator.keras')
            gan.save('./gan.keras')
            print("Checkpoint Saved")
        avg_fake_input_loss_discriminator.append(np.mean(fake_input_loss_discriminator))
        avg_real_input_loss_discriminator.append(np.mean(real_input_loss_discriminator))
        avg_loss_discriminator.append(np.mean(loss_discriminator))
        avg_loss_generator.append(epoch_loss_generator)
        plt.plot(range(len(avg_loss_discriminator)), avg_loss_discriminator, label='discriminator')
        plt.plot(range(len(avg_loss_generator)), avg_loss_generator, label='generator')
        plt.plot(range(len(avg_fake_input_loss_discriminator)), avg_fake_input_loss_discriminator, label='fake_input_loss_discriminator')
        plt.plot(range(len(avg_real_input_loss_discriminator)), avg_real_input_loss_discriminator, label='real_input_loss_discriminator')
        plt.legend(['discriminator_loss', 'generator_loss', 'fake_input_loss_discriminator', 'real_input_loss_discriminator'])
        plt.show()
        print('Time for epoch {} is {} sec'.format(epoch + 1, time() - start))

    return generator, discriminator, gan, avg_loss_discriminator, avg_loss_generator

def main():
    # Should be BCI competition IV-2b dataset (it actually is the 2a)
    dataset = MOABBDataset("BNCI2014_001", subject_ids=None)
    print(dataset.description)

    dataset = utils.preprocess_dataset(dataset)

    windows_dataset = create_windows_from_events(
        dataset,
        preload=True,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=500,
        window_stride_samples=50,
        drop_last_window=False,
        n_jobs=20,
    )

    dataset_dict = windows_dataset.split("session")
    # print(dataset_dict.keys())
    train_dataset = dataset_dict['0train']
    test_dataset = dataset_dict['1test']

    generator = utils.eeg_generator()
    print(generator.summary())
    discriminator = utils.eeg_discriminator()
    print(discriminator.summary())

    generator_loss = lambda x: tf.keras.losses.binary_crossentropy(tf.ones_like(x), x)
    discriminator_loss = tf.keras.losses.binary_crossentropy
    gan_loss = tf.keras.losses.binary_crossentropy

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    gan_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer, generator=generator,
                                     discriminator=discriminator)
    if os.path.exists(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    gan = utils.eeg_gan_network(generator, discriminator, generator_loss, gan_loss, generator_optimizer, gan_optimizer)
    print(gan.summary())
    generator, discriminator, gan, avg_loss_discriminator, avg_loss_generator = train(train_dataset, generator, discriminator, gan, steps_per_epoch=10, epochs=10, batch_size=22, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix)
    print("Training Complete")
    print("Average Loss Discriminator:", avg_loss_discriminator)
    print("Average Loss Generator:", avg_loss_generator)

if __name__ == "__main__":
    main()
