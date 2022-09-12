import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import src
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE

# FILE_NAME = 'gan_mbti.csv'
# FILE_NAME = 'test.csv'
# FILE_NAME = 'test2.csv'
FILE_NAME = 'test_mbti.csv'

if __name__ == '__main__':
    # sys.stdout = open('stdout1.txt', 'w')
    
    print('START')

    src.utils.set_random_state()
    src.utils.prepare_dataset(FILE_NAME)

    full_dataset = src.datasets.FullDataset()
    test_dataset = src.datasets.FullDataset(training=False)
 
    print("============ LGBM ============")
    src.jun_classifier.LGBM(src.datasets.training_samples, src.datasets.training_labels, src.datasets.test_samples, src.datasets.test_labels)
    
    # sys.stdout.close()
    gan_dataset = src.utils.get_gan_dataset(src.gans.GAN())

    with open('junganc_dataset.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
        pickle.dump(gan_dataset, file)

    ############ GAN ############
    print("============ LGBM with GAN ============")
    src.jun_classifier.LGBM(gan_dataset.samples, gan_dataset.labels, src.datasets.test_samples, src.datasets.test_labels)