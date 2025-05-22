
from utils.NOAHminiDataset import *



#################################################################
# MODEL PARMETER
#################################################################

num_channels = 5 # static rs
conditioning_dim = (24 * 6) + 4 # 24hrs for 6 features + lat, long, elev, cloud cover

learning_rate = 0.00001
batch_size = 2
num_epochs = 100


# Select the device you want to run the model on 
# device = 'cpu'
# device = 'cuda:0'
device = 'cuda:1'


# Specify the path to Model
model_path = "<PATH_TO_MODEL_YOU_WANT_TO_RUN>" # enure the band match in the  dataset


###################################################################
# DATASET CONFIG
###################################################################
dataset_dict = {
    'B1': {
        'train_dataset': NOAHminiB1Dataset(device = device),
        'test_dataset': NOAHminiB1Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB1Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B2': {
        'train_dataset': NOAHminiB2Dataset(device = device),
        'test_dataset': NOAHminiB2Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB2Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B3': {
        'train_dataset': NOAHminiB3Dataset(device = device),
        'test_dataset': NOAHminiB3Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB3Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B4': {
        'train_dataset': NOAHminiB4Dataset(device = device),
        'test_dataset': NOAHminiB4Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB4Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B5': {
        'train_dataset': NOAHminiB5Dataset(device = device),
        'test_dataset': NOAHminiB5Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB5Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B6': {
        'train_dataset': NOAHminiB6Dataset(device = device),
        'test_dataset': NOAHminiB6Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB6Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B7': {
        'train_dataset': NOAHminiB7Dataset(device = device),
        'test_dataset': NOAHminiB7Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB7Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B8': {
        'train_dataset': NOAHminiB8Dataset(device = device),
        'test_dataset': NOAHminiB8Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB8Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B9': {
        'train_dataset': NOAHminiB9Dataset(device = device),
        'test_dataset': NOAHminiB9Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB9Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B10': {
        'train_dataset': NOAHminiB10Dataset(device = device),
        'test_dataset': NOAHminiB10Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB10Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
    'B11': {
        'train_dataset': NOAHminiB11Dataset(device = device),
        'test_dataset': NOAHminiB11Dataset(
            device = device,
            dataset_csv = 'test.csv'
        ),
        'val_dataset': NOAHminiB11Dataset(
            device = device,
            dataset_csv = 'val.csv'
        ),
    },
}

# Select the band you want to run the model for 
band = 'B1'
# band = 'B2'
# band = 'B3'
# band = 'B4'
# band = 'B5'
# band = 'B6'
# band = 'B7'
# band = 'B8'
# band = 'B9'
# band = 'B10'
# band = 'B11'

# Select the dataset type
# set_type = 'train_dataset'
# set_type = 'test_dataset'
set_type = 'val_dataset'

dataset = dataset_dict[band][set_type]
