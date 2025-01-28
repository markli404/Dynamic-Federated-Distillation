import os
import torch
import numpy as np
import pandas


def flatten_model(model_dict):
    tensor = np.array([])

    for key in model_dict.keys():
        tensor = np.concatenate((tensor, model_dict[key].cpu().numpy().flatten()))

    return torch.tensor(tensor).squeeze()


def get_cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_magnitude(a, b):
    return (2*np.linalg.norm(a) * np.linalg.norm(b))/(np.linalg.norm(a)**2 + np.linalg.norm(b)**2)


def get_gradients(name):
    path = os.path.join('/content/gdrive/MyDrive/FGT-0606/models', name)
    # path = os.path.join('../../models', name)
    print(path)
    if not os.path.exists(path):
        raise Exception('Unable to find model: {}'.format(name))
        # os.mkdir(path) 

    num_files = len(os.listdir(path))
    previous = None

    gradients = []
    for i in range(0, num_files-1):
        j=i*5+1
        model_path = os.path.join(path, name + '_{}.pth'.format(j))
        model_dict = torch.load(model_path)['model']
        current = flatten_model(model_dict)
        if previous is not None:
            gradients.append(current - previous)
        previous = current

    return gradients


def calculate_gradient_similarity(name_1, name_2):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    g1 = get_gradients(name_1)
    g2 = get_gradients(name_2)

    cossims = []
    for i in range(min(len(g1), len(g2))):
        cossim = get_cossim(g1[i], g2[i])
        cossims.append(cossim)

    df = pandas.DataFrame(cossims)
    df.to_csv('cossim_{}_{}.csv'.format(name_1, name_2), index=False)

def calculate_magnitude_similarity(name_1, name_2):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    g1 = get_gradients(name_1)
    g2 = get_gradients(name_2)

    magnitudes = []
    for i in range(min(len(g1), len(g2))):
        magnitude = get_magnitude(g1[i], g2[i])
        magnitudes.append(magnitude)

    df = pandas.DataFrame(magnitudes)
    df.to_csv('magnitude_{}_{}.csv'.format(name_1, name_2), index=False)
    print('magnitude_{}_{}.csv'.format(name_1, name_2))

for i in ['aggregate_pairwise_vertical_both','aggregate_pairwise_vertical_both_magnitude_calibration','fedaverage']:
   calculate_gradient_similarity('fedaverage_10',i)


for i in ['aggregate_pairwise_vertical_both','aggregate_pairwise_vertical_both_magnitude_calibration','fedaverage']:
  calculate_magnitude_similarity('fedaverage_10',i)

# calculate_gradient_similarity('pairwise_vertical_cossim1', 'iid')
# calculate_magnitude_similarity('pairwise_vertical_cossim1', 'iid')

# calculate_gradient_similarity('iid', 'iid')
# calculate_magnitude_similarity('iid', 'iid')