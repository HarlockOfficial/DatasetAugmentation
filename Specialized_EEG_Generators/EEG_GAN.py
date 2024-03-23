# Network presented in: https://pure.ulster.ac.uk/en/publications/mieeg-gan-generating-artificial-motor-imagery-electroencephalogra
import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from moabb.datasets import PhysionetMI

sys.path.append("\home\harlock\thesis_project\DatasetAugmentation")
from DatasetAugmentation import utils


tf.random.set_seed(42)
np.random.seed(42)
np.random.RandomState(42)
noise_generator = None


def __train_step(training_dataset, generator, discriminator, batch_size, discriminator_increase_factor, generator_increase_factor, changed_steps_per_epoch_discriminator, changed_steps_per_epoch_generator, steps_per_epoch, step, avg_loss_discriminator, generator_optimizer, discriminator_optimizer):
    global noise_generator
    if noise_generator is None:
        noise_generator = tf.random.Generator.from_seed(42)
    if changed_steps_per_epoch_discriminator:
        steps_per_epoch *= discriminator_increase_factor
    if len(avg_loss_discriminator) > 0 and avg_loss_discriminator[-1] <= 0.2:
        steps_per_epoch //= 2
    # tf.print("Starting Step: " + str(step + 1) + " of " + str(steps_per_epoch), output_stream=sys.stdout)
    # Train discriminator
    discriminator_loss_fake = np.full((1,), 0.0, dtype=np.float64)
    discriminator_loss_real = np.full((1,), 0.0, dtype=np.float64)
    for iteration in range(2 * steps_per_epoch):
        #tf.print("Starting Discriminator Train Iteration: " + str(iteration + 1) + " of " + str(2 * steps_per_epoch),
        #          output_stream=sys.stdout)
        real_data = tf.random.shuffle(training_dataset)[:batch_size]
        noise = noise_generator.normal(shape=[batch_size, 58, 65])
        fake_data = generator(noise, training=False)
        with tf.GradientTape() as tape:
            result_real = discriminator(real_data, training=True)
            tmp_discriminator_loss_real = utils.discriminator_loss(result_real, np.ones((batch_size, 1)))
        gradients = tape.gradient(tmp_discriminator_loss_real, discriminator.trainable_weights)
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))
        discriminator_loss_real = discriminator_loss_real + tf.reduce_mean(tmp_discriminator_loss_real)
        with tf.GradientTape() as tape:
            result_fake = discriminator(fake_data, training=True)
            tmp_discriminator_loss_fake = utils.discriminator_loss(result_fake, np.zeros((batch_size, 1)))
        gradients = tape.gradient(tmp_discriminator_loss_fake, discriminator.trainable_weights)
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))
        discriminator_loss_fake = discriminator_loss_fake + tf.reduce_mean(tmp_discriminator_loss_fake)

    discriminator_loss_real = discriminator_loss_real / tf.cast((2 * steps_per_epoch), dtype=tf.float64)
    discriminator_loss_fake = discriminator_loss_fake / tf.cast((2 * steps_per_epoch), dtype=tf.float64)
    discriminator_loss = (discriminator_loss_fake + discriminator_loss_real) * 0.5

    if len(avg_loss_discriminator) > 0 and avg_loss_discriminator[-1] <= 0.2:
        steps_per_epoch *= 2

    if changed_steps_per_epoch_discriminator:
        steps_per_epoch //= discriminator_increase_factor

    steps_per_epoch = int(steps_per_epoch)

    if changed_steps_per_epoch_generator:
        steps_per_epoch *= generator_increase_factor
    # Train generator
    generator_loss = np.full((1,), 0.0, dtype=np.float64)
    y = np.ones((batch_size, 1))
    for iteration in range(steps_per_epoch):
        # tf.print("Starting GAN Train Iteration: " + str(iteration + 1) + " of " + str(steps_per_epoch),
        #          output_stream=sys.stdout)
        noise = noise_generator.normal(shape=[batch_size, 58, 65])
        with tf.GradientTape() as tape:
            result = discriminator(generator(noise, training=True), training=False)
            gen_loss = utils.generator_loss(result, y)
        gradients = tape.gradient(gen_loss, generator.trainable_weights)
        generator_optimizer.apply_gradients(zip(gradients, generator.trainable_weights))
        generator_loss = generator_loss + tf.reduce_mean(gen_loss)

    generator_loss = generator_loss / tf.cast(steps_per_epoch, dtype=tf.float64)

    if changed_steps_per_epoch_generator:
        steps_per_epoch //= generator_increase_factor

    return discriminator_loss_fake, discriminator_loss_real, discriminator_loss, generator_loss


def train(training_dataset, generator, discriminator, steps_per_epoch, batch_size, generator_optimizer, discriminator_optimizer, dataset_label=None):
    avg_loss_discriminator = []
    avg_loss_generator = []
    avg_fake_input_loss_discriminator = []
    avg_real_input_loss_discriminator = []
    discriminator_increase_factor = 2
    generator_increase_factor = 2

    train_step = tf.function(__train_step)
    # main train loop
    for epoch in range(20):
        tf.print("Starting Epoch: " + str(epoch + 1) + " of 20", output_stream=sys.stdout)
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
            min_discriminator_loss_fake, min_discriminator_loss_real, min_discriminator_loss, min_generator_loss = train_step(training_dataset, generator, discriminator, batch_size, discriminator_increase_factor, generator_increase_factor, changed_steps_per_epoch_discriminator, changed_steps_per_epoch_generator, steps_per_epoch, step, avg_loss_discriminator, generator_optimizer, discriminator_optimizer)
            fake_input_loss_discriminator.append(min_discriminator_loss_fake)
            real_input_loss_discriminator.append(min_discriminator_loss_real)
            loss_discriminator.append(min_discriminator_loss)
            loss_generator.append(min_generator_loss)

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
            if not os.path.exists('./saved_model/'):
                os.makedirs('./saved_model/')
            if dataset_label is not None:
                tf.saved_model.save(generator, './saved_model/generator_' + str(dataset_label) + '/')
            else:
                tf.saved_model.save(generator, './saved_model/generator/')
            tf.print("Model Saved", output_stream=sys.stdout)
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
        tf.print('Epoch: ' + str(epoch + 1) + ' Discriminator Loss: ' + str(avg_loss_discriminator[-1]) + ' Generator Loss: ' + str(avg_loss_generator[-1]), output_stream=sys.stdout)
        tf.print('Time for epoch {} is {} sec'.format(epoch + 1, time() - start), output_stream=sys.stdout)
    tf.print("Training Complete", output_stream=sys.stdout)
    tf.print("Average Loss Discriminator: " + str(avg_loss_discriminator), output_stream=sys.stdout)
    tf.print("Average Loss Generator: " + str(avg_loss_generator), output_stream=sys.stdout)
    return min(avg_loss_discriminator), min(avg_loss_generator)


def main():
    # Setting gpu for limit memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        #Restrict Tensorflow to only allocate 6gb of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            #virtual devices must be set before GPUs have been initialized
            print(e)
            exit(-1)
    print("Starting EEG GAN Training process, with pid = ", os.getpid(), " and ppid = ", os.getppid())
    x, y = utils.load_dataset(PhysionetMI, verbose=1)

    train_dataset_by_label = utils.split_dataset_by_label(x, y, verbose=1)

    for index, label in enumerate(train_dataset_by_label.keys()):
        generator = utils.eeg_generator(graph=True)
        discriminator = utils.eeg_discriminator(graph=True)
        print(generator.summary())
        print(discriminator.summary())
        generator_optimizer, discriminator_optimizer = utils.eeg_gan_default_optimizer()
        """
        checkpoint_dir = './training_checkpoints_' + str(label)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer, generator=generator_list[index],
                                         discriminator=discriminator_list[index])
        if os.path.exists(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        """
        min_avg_loss_discriminator, min_avg_loss_generator = train(train_dataset_by_label[label], generator, discriminator, steps_per_epoch=10, batch_size=22, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, dataset_label=label)
        print("Training Complete")
        print("Min Loss Discriminator:", min_avg_loss_discriminator)
        print("Min Loss Generator:", min_avg_loss_generator)

if __name__ == "__main__":
    main()
