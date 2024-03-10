# Network presented in: https://pure.ulster.ac.uk/en/publications/mieeg-gan-generating-artificial-motor-imagery-electroencephalogra
import os
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from braindecode.preprocessing import create_windows_from_events

import sys
sys.path.append("\home\harlock\thesis_project\DatasetAugmentation")
from DatasetAugmentation import utils

seed = tf.random.normal([22, 500, 50])
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
np.random.RandomState(42)


def train(training_dataset, generator, discriminator, gan, steps_per_epoch, epochs, batch_size, checkpoint, checkpoint_prefix, dataset_label=None):
    avg_fake_input_loss_discriminator = []
    avg_real_input_loss_discriminator = []
    avg_loss_discriminator = []
    avg_loss_generator = []
    #training_dataset_list = list(training_dataset)

    discriminator_increase_factor = 2
    generator_increase_factor = 2
    # main train loop
    for epoch in range(50):
        print("Starting Epoch:", epoch + 1, "of 50")
        start = time()

        fake_input_loss_discriminator = []
        real_input_loss_discriminator = []
        loss_discriminator = []
        loss_generator = []
        changed_steps_per_epoch_discriminator = False
        changed_steps_per_epoch_generator = False
        if len(avg_loss_discriminator) > 0 and avg_loss_discriminator[-1] > 0.4:
            changed_steps_per_epoch_discriminator = True
        if len(avg_loss_discriminator) > 0 and avg_loss_discriminator[-1] < 0.2:
            changed_steps_per_epoch_generator = True

        for step in range(steps_per_epoch):
            if changed_steps_per_epoch_discriminator:
                steps_per_epoch *= discriminator_increase_factor
            if len(avg_loss_discriminator) > 0 and avg_loss_discriminator[-1] <= 0.2:
                steps_per_epoch //= 2
            print("Starting Step:", step + 1, "of", steps_per_epoch)
            # Train discriminator
            discriminator_loss_fake = float('inf')
            discriminator_loss_real = float('inf')
            for iteration in range(2*steps_per_epoch):
                print("Starting Discriminator Train Iteration:", iteration + 1, "of", 2*steps_per_epoch)
                real_data = random.choice(training_dataset)
                noise = tf.random.normal([batch_size, 500, 50])
                fake_data = generator.predict(noise)
                discriminator_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
                fake_data = np.reshape(fake_data, (batch_size, 500))
                discriminator_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

            if len(avg_loss_discriminator) > 0 and avg_loss_discriminator[-1] <= 0.2:
                steps_per_epoch *= 2

            if changed_steps_per_epoch_discriminator:
                steps_per_epoch //= discriminator_increase_factor

            steps_per_epoch = int(steps_per_epoch)

            if changed_steps_per_epoch_generator:
                steps_per_epoch *= generator_increase_factor
            # Train generator
            generator_loss = 0
            y = np.ones((batch_size, 1))
            for iteration in range(steps_per_epoch):
                print("Starting GAN Train Iteration:", iteration + 1, "of", steps_per_epoch)
                noise = tf.random.normal([batch_size, 500, 50])
                generator_loss += gan.train_on_batch(noise, y)[0]
            if changed_steps_per_epoch_generator:
                steps_per_epoch //= generator_increase_factor
            fake_input_loss_discriminator.append(discriminator_loss_fake)
            real_input_loss_discriminator.append(discriminator_loss_real)
            loss_discriminator.append(0.5 * np.add(discriminator_loss_real, discriminator_loss_fake))
            loss_generator.append(generator_loss/steps_per_epoch)

        if changed_steps_per_epoch_discriminator:
            discriminator_increase_factor += 1

        if discriminator_increase_factor > 20:
            discriminator_increase_factor = 20

        if changed_steps_per_epoch_generator:
            generator_increase_factor += 1

        if generator_increase_factor > 20:
            generator_increase_factor = 20

        epoch_loss_generator = np.mean(loss_generator)
        # Save loss
        if len(avg_loss_generator) == 0 or epoch_loss_generator < min(avg_loss_generator):
            checkpoint.save(file_prefix=checkpoint_prefix)
            if dataset_label is not None:
                generator.save('./generator_' + str(dataset_label) + '.h5')
                discriminator.save('./discriminator_' + str(dataset_label) + '.h5')
                gan.save('./gan_' + str(dataset_label) + '.h5')
            else:
                generator.save('./generator.h5')
                discriminator.save('./discriminator.h5')
                gan.save('./gan.h5')
            print("Checkpoint Saved")
        avg_fake_input_loss_discriminator.append(np.mean(fake_input_loss_discriminator))
        avg_real_input_loss_discriminator.append(np.mean(real_input_loss_discriminator))
        avg_loss_discriminator.append(np.mean(loss_discriminator))
        avg_loss_generator.append(epoch_loss_generator)
        plt.plot(range(len(avg_loss_discriminator)), avg_loss_discriminator, label='discriminator' + ('_' + str(dataset_label)) if dataset_label is not None else '')
        plt.plot(range(len(avg_loss_generator)), avg_loss_generator, label='generator'+ ('_' + str(dataset_label)) if dataset_label is not None else '')
        plt.plot(range(len(avg_fake_input_loss_discriminator)), avg_fake_input_loss_discriminator, label='fake_input_loss_discriminator' + ('_' + str(dataset_label)) if dataset_label is not None else '')
        plt.plot(range(len(avg_real_input_loss_discriminator)), avg_real_input_loss_discriminator, label='real_input_loss_discriminator' + ('_' + str(dataset_label)) if dataset_label is not None else '')
        if dataset_label is not None:
            plt.title('GAN Losses for Dataset Label: ' + str(dataset_label))
            plt.legend(['discriminator_loss_' + str(dataset_label), 'generator_loss_' + str(dataset_label), 'fake_input_loss_discriminator_' + str(dataset_label), 'real_input_loss_discriminator_' + str(dataset_label)])
            plt.savefig('gan_loss_' + str(dataset_label) + '.png')
        else:
            plt.legend(['discriminator_loss', 'generator_loss', 'fake_input_loss_discriminator', 'real_input_loss_discriminator'])
            plt.savefig('gan_loss.png')
        print('Epoch:', epoch + 1, 'Discriminator Loss:', avg_loss_discriminator[-1], 'Generator Loss:', avg_loss_generator[-1])
        print('Time for epoch {} is {} sec'.format(epoch + 1, time() - start))

    return generator, discriminator, gan, avg_loss_discriminator, avg_loss_generator


def main():
    print("Starting EEG GAN Training process, with pid = ", os.getpid(), " and ppid = ", os.getppid())
    dataset = utils.load_dataset(verbose=1)
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

    print(type(train_dataset))

    train_dataset_by_label = utils.split_dataset_by_label(train_dataset)
    generator_list = []
    discriminator_list = []
    gan_list = []

    for index, label in enumerate(train_dataset_by_label.keys()):
        generator_list.append(utils.eeg_generator())
        discriminator_list.append(utils.eeg_discriminator())
        print(generator_list[index].summary())
        print(discriminator_list[index].summary())
        generator_loss, discriminator_loss, gan_loss = utils.eeg_gan_default_loss()
        generator_optimizer, discriminator_optimizer, gan_optimizer = utils.eeg_gan_default_optimizer()
        checkpoint_dir = './training_checkpoints_' + str(label)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer, generator=generator_list[index],
                                         discriminator=discriminator_list[index])
        if os.path.exists(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        gan = utils.eeg_gan_network(generator_list[index], discriminator_list[index], generator_loss, discriminator_loss, gan_loss, generator_optimizer, discriminator_optimizer, gan_optimizer)
        print(gan.summary())
        generator, discriminator, gan, avg_loss_discriminator, avg_loss_generator = train(train_dataset_by_label[label], generator_list[index], discriminator_list[index], gan, steps_per_epoch=10, epochs=10, batch_size=22, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix, dataset_label=label)
        gan_list.append(gan)
        generator_list[index] = generator
        discriminator_list[index] = discriminator
        print("Training Complete")
        print("Average Loss Discriminator:", avg_loss_discriminator)
        print("Average Loss Generator:", avg_loss_generator)
        print("Min Loss Discriminator:", min(avg_loss_discriminator))
        print("Min Loss GAN:", min(avg_loss_generator))

if __name__ == "__main__":
    main()
