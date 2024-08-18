import os

from dataset import DataLoaderTrain, DataLoaderVal
def get_training_data(rgb_dir, img_options, aug_ments):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, aug_ments, None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)
