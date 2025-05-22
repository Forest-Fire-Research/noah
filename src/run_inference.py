
from utils.config import *
from utils.NOAHModelUNetFiLM import *

from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # init model
    model = UNetFiLM(
        in_channels = num_channels, 
        conditioning_dim = conditioning_dim
    ).to(device)
    # load save state dict
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # predict
    model.eval()

    # build loader
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = True, 
        drop_last = True
    )

    # loop over validation data
    val_batch_progress_bar = tqdm(
        dataloader,
        desc = "Batch"
    )
    for gbs, static_rs, targets in val_batch_progress_bar:
        outputs = model(
            x_img = static_rs, 
            x_weather = gbs,
        )

        plt.imshow(
            outputs.cpu().detach().numpy()[0].squeeze()
        )
        plt.show()

        del static_rs
        del targets
        del outputs
        del gbs

        break 