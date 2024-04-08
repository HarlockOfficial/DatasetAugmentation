import os.path
import sys

import pandas as pd
import torch

import PrettyPrintConfusionMatrix
import VirtualController


def prediction_to_label(prediction):
    if prediction == 0:
        return 'Feet'
    elif prediction == 1:
        return 'Left Hand'
    elif prediction == 2:
        return 'Right Hand'
    else:
        raise ValueError(f'Invalid prediction: {prediction}')

def main(path_to_generators, path_to_model, generated_samples_amount=None):
    if generated_samples_amount is None:
        generated_samples_amount = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    right_generator, left_generator, feet_generator = VirtualController.load_generators(path_to_generators, device=device)
    classificator = VirtualController.load_classificator(path_to_model)
    predictions = {'Feet':
                       {'Feet': 0,
                        'Left Hand': 0,
                        'Right Hand': 0},
                   'Left Hand':
                       {'Feet': 0,
                        'Left Hand': 0,
                        'Right Hand': 0},
                   'Right Hand':
                       {'Feet': 0,
                        'Left Hand': 0,
                        'Right Hand': 0}}
    for i in range(generated_samples_amount):
        noise = torch.rand(1, 1, 58, 65).to(torch.float32).to(device)
        right_sample = right_generator(noise)
        left_sample = left_generator(noise)
        feet_sample = feet_generator(noise)
        right_sample = right_sample.detach().cpu().numpy().reshape(1, 58, 65)
        left_sample = left_sample.detach().cpu().numpy().reshape(1, 58, 65)
        feet_sample = feet_sample.detach().cpu().numpy().reshape(1, 58, 65)
        right_prediction = classificator.predict(right_sample)[0]
        left_prediction = classificator.predict(left_sample)[0]
        feet_prediction = classificator.predict(feet_sample)[0]
        predictions['Feet'][prediction_to_label(feet_prediction)] += 1
        predictions['Left Hand'][prediction_to_label(left_prediction)] += 1
        predictions['Right Hand'][prediction_to_label(right_prediction)] += 1

    df = pd.DataFrame(predictions)
    print(df)
    fig, ax = PrettyPrintConfusionMatrix.pp_matrix(df, pred_val_axis='y', show_null_values=2, annot=True, fz=10, lw=0.5, cbar=True, cmap='Blues',
                    title='Confusion Matrix Generators')
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    fig.savefig(f'./figures/confusion_matrix_generators_{path_to_generators.split("/")[-1]}_using_{path_to_model.split("/")[-1]}.png')


if __name__ == '__main__':
    """
    generators_path = sys.argv[1]
    model_path = sys.argv[2]
    samples = None
    if len(sys.argv) >= 4:
        samples = int(sys.argv[3])
    """
    generators_path = '../models/generators'
    model_path = '../models/2024_03_25_23_21_59/LSTMNet_0.5943600867678959.pkl'
    samples = None
    main(generators_path, model_path, samples)