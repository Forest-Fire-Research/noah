
import os
import zarr
import torchvision  
import numpy as np
import pandas as pd
from torch import tensor, float32
from torch.utils.data import Dataset
from dotenv import load_dotenv

# load the .env file variables 
load_dotenv()

NOAH_MINI_DIR = os.getenv("NOAH_MINI_DIR")

class NOAHminiDataset(Dataset):
    def __init__(
            self, 
            bands:list = [1,2,3,4,5,6,7,8,9,10,11], 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        self.dataset_dir = dataset_dir
        self.dataset_csv = dataset_csv
        self.device = device
        self.bands = bands

        self.static_rs_transform = torchvision.transforms.Normalize(
            mean = [43.1937, 26.8497, 44.4554,  9.1292,  5.9803],
            std = [50.3019, 26.3059, 47.2773,  6.0937,  6.2454]
        )     
        
        self.target_transform = torchvision.transforms.Normalize(
            mean = [11416.8799, 11066.1670, 10255.2549, 10395.8926, 12451.8037, 6798.9302, 6104.8594, 10284.2461,  4375.1392, 14649.8682, 14116.4111],
            std = [6639.4092, 6743.3730, 6392.4634, 6847.2998, 7056.6699, 3608.1509, 2961.2961, 6569.3779, 1816.1825, 8036.9028, 7366.7769]
        )

        self.gbs_transform = {
            'mean': tensor([ 5.5939e+01, -1.1455e+02,  5.5965e+02,  7.8167e-01, -2.9308e+00, -1.1101e+01,  5.8700e+01,  1.9926e+01,  9.1270e+00,  9.5521e+01,-4.1743e+00, -1.1306e+01,  6.2089e+01,  1.9805e+01,  8.9335e+00, 9.5524e+01, -5.2493e+00, -1.1566e+01,  6.4655e+01,  1.9418e+01, 9.0335e+00,  9.5535e+01, -6.0677e+00, -1.1646e+01,  6.7155e+01, 1.9727e+01,  9.6002e+00,  9.5550e+01, -6.7460e+00, -1.1866e+01, 6.8905e+01,  1.9084e+01,  8.8835e+00,  9.5563e+01, -7.5410e+00, -1.2239e+01,  7.0872e+01,  2.0753e+01,  8.2668e+00,  9.5576e+01,-8.2127e+00, -1.2529e+01,  7.2505e+01,  2.2099e+01,  7.6168e+00, 9.5587e+01, -8.6543e+00, -1.2821e+01,  7.3139e+01,  2.1316e+01, 7.3335e+00,  9.5599e+01, -9.0467e+00, -1.3096e+01,  7.3710e+01, 2.0766e+01,  6.9688e+00,  9.5607e+01, -1.0395e+01, -1.3775e+01, 7.6969e+01,  2.0715e+01,  6.8000e+00,  9.5645e+01, -1.0553e+01,-1.3835e+01,  7.7453e+01,  2.1490e+01,  6.4667e+00,  9.5652e+01,-1.0725e+01, -1.4093e+01,  7.7020e+01,  2.0248e+01,  6.6333e+00, 9.5661e+01, -9.9913e+00, -1.3896e+01,  7.4186e+01,  1.9852e+01, 6.8167e+00,  9.5661e+01, -8.8397e+00, -1.3608e+01,  7.0186e+01, 2.0504e+01,  7.9833e+00,  9.5664e+01, -7.1280e+00, -1.3053e+01, 6.5053e+01,  2.0134e+01,  8.0500e+00,  9.5655e+01, -5.0380e+00,-1.2411e+01,  5.9503e+01,  2.0187e+01,  8.7500e+00,  9.5641e+01,-2.9013e+00, -1.1831e+01,  5.4186e+01,  2.0136e+01,  9.2500e+00, 9.5620e+01, -1.3113e+00, -1.1533e+01,  4.9953e+01,  1.9771e+01, 1.0683e+01,  9.5595e+01, -1.7500e-01, -1.1480e+01,  4.6900e+01, 2.0006e+01,  1.1683e+01,  9.5562e+01,  7.0833e-01, -1.1362e+01, 4.4867e+01,  1.9389e+01,  1.1500e+01,  9.5531e+01,  1.1100e+00,-1.1248e+01,  4.4417e+01,  1.8225e+01,  1.1867e+01,  9.5504e+01, 9.7000e-01, -1.1198e+01,  4.5150e+01,  1.9467e+01,  1.2100e+01, 9.5484e+01,  5.6333e-01, -1.1082e+01,  4.6650e+01,  1.8058e+01, 1.1633e+01,  9.5471e+01, -4.1667e-01, -1.1017e+01,  5.0067e+01, 1.8800e+01,  1.0617e+01,  9.5464e+01]),
            'std': tensor([  2.6639,   2.6115, 165.8703,   1.0437,  16.2144,  12.7526,  18.9693, 10.6798,   6.8332,   2.1557,  15.7734,  12.9891,  17.2523,  11.0711,  7.0247,   2.1481,  15.4159,  13.4499,  15.6361,  10.8253,   7.0763,  2.1365,  15.0624,  13.9390,  13.4212,  10.2977,   6.9089,   2.1309, 14.8676,  14.0768,  12.0053,  10.2959,   7.2928,   2.1297,  14.6793, 14.1419,  11.6003,  10.3189,   6.4116,   2.1280,  14.6158,  14.3997, 10.6130,   9.8354,   6.6162,   2.1276,  14.6230,  14.5159,  10.8571,  9.7660,   6.3929,   2.1283,  14.6355,  14.6628,  10.9433,   9.6921,  5.7855,   2.1265,  14.7920,  14.8488,   9.7638,   9.8847,   4.9532,  2.1431,  15.0438,  15.0168,   9.3684,   8.8719,   5.1687,   2.1413, 15.4948,  15.3341,   9.4419,   9.3521,   5.4270,   2.1389,  16.0230, 15.5588,   9.8832,   9.1974,   5.4802,   2.1400,  16.4916,  15.6456, 10.9670,   9.5299,   5.0025,   2.1445,  16.8240,  15.3441,  12.5978,  8.9006,   5.0471,   2.1510,  17.1644,  14.9804,  14.8271,   8.8646,  5.8569,   2.1612,  17.2745,  14.0842,  16.2341,   9.2092,   6.0695,  2.1692,  17.0286,  13.3144,  17.1791,   8.8780,   7.2290,   2.1779, 16.9046,  12.7512,  18.0137,   9.9219,   7.3113,   2.1872,  16.9776, 12.5616,  17.6381,   8.9120,   6.8796,   2.1999,  17.1010,  12.3996, 18.4088,   9.6851,   6.3405,   2.2096,  17.0559,  12.3110,  19.0112, 10.4967,   6.1422,   2.2161,  16.9859,  12.3870,  18.9438,  10.2131,  6.5559,   2.2228,  16.7163,  12.6634,  18.5655,   9.9546,   5.6266,  2.2203])
        }

        self.modalities = [
            'gbs', 
            'biomass', 
            'crowncover', 
            'fueltype', 
            'landcover', 
            'topography',
        ]

        # get metadata
        self.__metadata = pd.read_csv(
            self.__get_dataset_csv_path()
        )

        # set columns for 24h GBS weather modality
        self.weather_col = self.__metadata.columns[:-6]

    def __len__(self):
        return len(self.__metadata)
    
    def __getitem__(self, index):
        metadata = self.__metadata.iloc[index]

        gbs_modality = metadata[self.weather_col].values.astype(float)

        biomass_modality = zarr.open(
            self.__get_modality_path(
                metadata['biomass modality']
            ), 
            mode='r'
        )

        crowncover_modality = zarr.open(
            self.__get_modality_path(
                metadata['crowncover modality']
            ), 
            mode='r'
        )

        fueltype_modality = zarr.open(
            self.__get_modality_path(
                metadata['fueltype modality']
            ), 
            mode='r'
        )

        landcover_modality = zarr.open(
            self.__get_modality_path(
                metadata['landcover modality']
            ), 
            mode='r'
        )

        topography_modality = zarr.open(
            self.__get_modality_path(
                metadata['topography modality']
            ), 
            mode='r'
        )

        satellite_modality = zarr.open(
            self.__get_modality_path(
                metadata['satellite modality']
            ), 
            mode='r'
        )

        del metadata

        gbs_data = tensor(np.array(gbs_modality))
        gbs_data = (gbs_data - self.gbs_transform['mean']) / self.gbs_transform['std']
        gbs_data = gbs_data.to(
            self.device,
            dtype = float32
        )
        del gbs_modality

        static_rs_data = tensor(
            np.stack(
                (
                    np.array(biomass_modality), 
                    np.array(crowncover_modality), 
                    np.array(fueltype_modality),
                    np.array(landcover_modality), 
                    np.array(topography_modality),
                ), 
                axis = 0
            )
        ).to(
            self.device,
            dtype = float32
        )
        static_rs_data = self.static_rs_transform(static_rs_data)
        del biomass_modality
        del crowncover_modality
        del fueltype_modality
        del landcover_modality
        del topography_modality

        target_rs_data = tensor(
            np.array(satellite_modality)
        ).to(
            self.device,
            dtype = float32
        )
        target_rs_data = self.target_transform(target_rs_data)
        target_rs_data = target_rs_data[[
                band-1 for band in self.bands
            ]]
        del satellite_modality

        return gbs_data, static_rs_data, target_rs_data

    def __get_dataset_csv_path(self):
        return f"{self.dataset_dir}{os.sep}{self.dataset_csv}"
    
    def __get_modality_path(self, modality_file):
        return f"{self.dataset_dir}{os.sep}{modality_file}"

        
        
class NOAHminiB1Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [1],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB2Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [2],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB3Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [3],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB4Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [4],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB5Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [5],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB6Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [6],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB7Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [7],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB8Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [8],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB9Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [9],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data

class NOAHminiB10Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [10],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data
        
class NOAHminiB11Dataset(NOAHminiDataset):
    def __init__(
            self, 
            dataset_dir = NOAH_MINI_DIR, 
            dataset_csv = "train.csv",
            device = 'cpu',
            transforms = None
        ):
        super().__init__(
            bands = [11],
            dataset_dir = dataset_dir, 
            dataset_csv = dataset_csv,
            transforms = transforms,
            device = device
        )

    def __getitem__(self, index):
        gbs_data, static_rs_data, target_rs_data = super().__getitem__(index)
        return gbs_data, static_rs_data, target_rs_data
