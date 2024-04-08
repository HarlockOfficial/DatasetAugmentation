import os
import pickle
import random
import sys

import numpy as np
import torch

import EEGClassificator.utils
import VirtualController.main

random.seed(42)
np.random.seed(42)
np.random.RandomState(42)
torch.manual_seed(42)

def load_all(path, out_class=None):
    generator_list = []
    files = os.listdir(path)
    files = filter(lambda x: x.endswith('.pkl') and 'generator' in x, files)
    if out_class is not None:
        files = filter(lambda x: out_class in x, files)
    for file in files:
        generator = pickle.load(open(path + file, 'rb'))
        generator_list.append((file, generator))
    return generator_list

def main(out_class, path_to_generator_network, path_to_classificator_network):
    if out_class.lower() == "none":
        out_class = None

    class_to_value = EEGClassificator.utils.to_categorical(out_class)

    generator_list = load_all(path_to_generator_network, out_class)
    classificator = VirtualController.load_classificator(path_to_classificator_network)
    print(f"Expected class: {out_class}/{class_to_value}")
    for filename, generator in generator_list:
        print(f"Testing generator: {filename}", flush=True)
        seed = torch.rand([1000, 1, 58, 65]).to('cuda' if torch.cuda.is_available() else 'cpu').to(torch.float32)
        # score the generator
        generated_element, _ = generator(seed)
        generated_element = generated_element.detach().cpu().numpy().reshape(1000, 58, 65)
        result = classificator.predict(generated_element)
        correct_elements = np.sum(result == class_to_value)
        print(f"Number of generated elements: {result.shape[0]}")
        print(f"Number of correct elements: {correct_elements}")
        print(f"Accuracy: {correct_elements/result.shape[0]}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: Tester.py <network_expected_output_class> <path_to_generator_network_file_to_load> <path_to_classificator_network_file_to_load>")
        sys.exit(1)
    expected_output_class = sys.argv[1]
    path_to_network_files = sys.argv[2]
    path_to_classificator_network = sys.argv[3]
    """
    expected_output_class = "feet"
    path_to_network_files = '../../../models/2024_04_02_20_00_22/'
    path_to_classificator_network = '../../../models/2024_03_25_23_21_59/LSTMNet_0.5943600867678959.pkl'
    """
    main(expected_output_class, path_to_network_files, path_to_classificator_network)
