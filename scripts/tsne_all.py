import context

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler

import src

# DATASET = 'creditcard.csv'
DATASET = 'test_mbti.csv'

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

GAN_MODELS = [
    'orgin',
    # src.gans.GAN,
]
import pickle
if __name__ == '__main__':
    result = dict()
    src.utils.set_random_state()
    src.utils.prepare_dataset(DATASET)

    dataset = src.datasets.FullDataset(training=True)

    gan_datasets = []

    with open(file='ganbti_dataset.p', mode='rb') as f:
        gan_dataset=pickle.load(f)

    gan_datasets = [dataset, gan_dataset]
    
    raw_x, raw_y = dataset[:]
    raw_x = raw_x.numpy()
    raw_y = raw_y.numpy()

    idx = 0
    embedded_x = TSNE(  
    # learning_rate='auto',
    init='random',
    random_state=src.config.seed,
    ).fit_transform(raw_x)
    result['Original'] = [embedded_x, raw_y]

    # for M in GAN_MODELS:
    #     src.utils.set_random_state()
    #     if(M=='orgin'):
    #         embedded_x = TSNE(
    #         # learning_rate='auto',
    #         init='random',
    #         random_state=src.config.seed,
    #         ).fit_transform(raw_x)
    #         result['Original'] = [embedded_x, raw_y]
    #     else:
    #         print("== START {} ==".format(M.__name__))
    #         gan = M()
    #         gan.fit()
    #         z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.models.z_size], device=src.config.device)
    #         x = np.concatenate([raw_x, gan.g(z).detach().cpu().numpy()])
    #         y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    #         # x = gan_datasets[1].samples
    #         # y = gan_datasets[1].labels

    #         # x = x.numpy()
    #         # y = y.numpy()

    #         embedded_x = TSNE(
    #             # learning_rate='auto',
    #             init='random',
    #             random_state=src.config.seed,
    #         ).fit_transform(x)
    #         model_name = M.__name__
    #         result[model_name] = [embedded_x, y]
    #         idx += 1

    # for M in TRADITIONAL_METHODS:
    #     print("== START {} ==".format(M.__name__))
    #     x, _ = M(random_state=src.config.seed).fit_resample(raw_x, raw_y)
    #     y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
    #     embedded_x = TSNE(
    #         # learning_rate='auto',
    #         init='random',
    #         random_state=src.config.seed,
    #     ).fit_transform(x)
    #     result[M.__name__] = [embedded_x, y]

    palette = sns.color_palette("bright", 16) 
    sns.scatterplot(embedded_x[:,0], embedded_x[:,1], hue=raw_y, legend='full', palette=palette) 

    plt.savefig(src.config.path.test_results / 'all_distribution.jpg')
    # plt.savefig(src.config.path.test_results / "shap_t-SNE.pdf", format='pdf', dpi=1000, bbox_inches='tight')

    plt.show()
