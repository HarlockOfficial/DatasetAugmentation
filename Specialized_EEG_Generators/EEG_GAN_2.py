import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from moabb.datasets import PhysionetMI
from torcheeg.models import EEGfuseNet, EFDiscriminator

from DatasetAugmentation.utils import utils

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def train(dataset, label, device='cuda', train_epochs=1000, path_to_save='./models'):
    dataset = np.array(dataset)
    dataset = torch.tensor(dataset).to(device).to(torch.float32).unsqueeze(1)
    ds_shape = dataset.shape

    discriminator_loss_fn = torch.nn.BCELoss()
    generator_loss_fn = torch.nn.BCELoss()
    generator = EEGfuseNet(in_channels=1, num_electrodes=ds_shape[-2], chunk_size=ds_shape[-1])
    discriminator = EFDiscriminator(in_channels=1, num_electrodes=ds_shape[-2], chunk_size=ds_shape[-1])
    generator.to(device)
    discriminator.to(device)

    generator_optim = torch.optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    generator.train()
    discriminator.train()

    discriminator_loss = float('inf')
    generator_loss = float('inf')
    discriminator_loss_list = []
    generator_loss_list = []
    train_repeat_per_epoch = 10
    total_epochs = train_epochs//train_repeat_per_epoch
    for i in range(total_epochs):
        start_time = datetime.now()
        for _ in range((total_epochs-i) * train_repeat_per_epoch):
            x = torch.rand(dataset.shape).to(device).to(torch.float32)
            # Train discriminator
            discriminator_optim.zero_grad()
            fake_y, _ = generator(x)
            assert fake_y.shape == dataset.shape
            fake_result = discriminator(fake_y)
            real_result = discriminator(dataset)
            fake_loss = discriminator_loss_fn(fake_result, torch.zeros_like(fake_result))
            real_loss = discriminator_loss_fn(real_result, torch.ones_like(real_result))
            loss = fake_loss + real_loss
            loss.backward()
            discriminator_optim.step()
            discriminator_loss_list.append(loss.item())

        for _ in range((i+1) * train_repeat_per_epoch):
            x = torch.rand(dataset.shape).to(device).to(torch.float32)
            # Train generator
            generator_optim.zero_grad()
            fake_y, _ = generator(x)
            fake_result = discriminator(fake_y)
            fake_loss = generator_loss_fn(fake_result, torch.ones_like(fake_result))
            fake_loss.backward()
            generator_optim.step()
            generator_loss_list.append(fake_loss.item())

        disc_loss = sum(discriminator_loss_list[-train_repeat_per_epoch:])/train_repeat_per_epoch
        gen_loss = sum(generator_loss_list[-train_repeat_per_epoch:])/train_repeat_per_epoch
        print(f'Epoch {i+1}, Discriminator Loss: {disc_loss}, Generator Loss: {gen_loss}, step time {datetime.now() - start_time}', flush=True)
        if disc_loss < discriminator_loss:
            discriminator_loss = disc_loss
            discriminator.eval()
            with open(f'{path_to_save}/discriminator_{label}_{i+1}_{discriminator_loss}.pkl', 'wb') as f:
                pickle.dump(discriminator, f)
            discriminator.train()
        if gen_loss < generator_loss:
            generator_loss = gen_loss
            generator.eval()
            with open(f'{path_to_save}/generator_{label}_{i+1}_{generator_loss}.pkl', 'wb') as f:
                pickle.dump(generator, f)
            generator.train()
        discriminator_loss_list[:] = discriminator_loss_list[:-train_repeat_per_epoch]
        generator_loss_list[:] = generator_loss_list[:-train_repeat_per_epoch]
        discriminator_loss_list.append(discriminator_loss)
        generator_loss_list.append(generator_loss)

    return discriminator_loss_list, generator_loss_list

def main(path_to_save = None):
    if path_to_save is None:
        t = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        path_to_save = f'./models/{t}/'
    x, y = utils.load_dataset(PhysionetMI, verbose=0)
    train_dataset_by_label = utils.split_dataset_by_label(x, y, verbose=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for label, dataset in train_dataset_by_label.items():
        print(f'Training for label {label}', flush=True)
        train(dataset, label, device=device, path_to_save=path_to_save, train_epochs=200)


if __name__ == '__main__':
    main()