
import imageio
from PIL import Image
from trellis.pipelines import TrainingPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.dataset.dataset_vae import VAEDataset
from trellis.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder

import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from IPython import embed

class DiceLoss(nn.Module):
    def  __init__ (self, smooth=1e-6):
        super(DiceLoss, self). __init__ ()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Sigmoid function
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice  # return Dice Loss
 # when use_fp16 is true, it results in nan the training loss
json_config = '''
{
    "name": "SparseStructureEncoder",
    "args": {
        "in_channels": 1,
        "latent_channels": 8,
        "num_res_blocks": 2,
        "num_res_blocks_middle": 2,
        "channels": [32, 128, 512],
        "use_fp16": false               
    },
    "pretrained": "/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_enc_conv3d_16l8_fp16.safetensors"
}
'''
config = json.loads(json_config)

sparse_vae_encoder = SparseStructureEncoder(**(config['args']))
#sparse_vae_encoder.load_state_dict(load_file(config['pretrained']))


json_config = '''
{
    "name": "SparseStructureDecoder",
    "args": {
        "out_channels": 1,
        "latent_channels": 8,
        "num_res_blocks": 2,
        "num_res_blocks_middle": 2,
        "channels": [512, 128, 32],
        "use_fp16": false
    },
    "pretrained": "/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
}
'''
config = json.loads(json_config)

sparse_vae_decoder = SparseStructureDecoder(**(config['args']))
#sparse_vae_decoder.load_state_dict(load_file(config['pretrained']))

# decoder range: [-220, 220]

models = {
    "encoder": sparse_vae_encoder,
    "decoder": sparse_vae_decoder
}

params = []
for model in models.values():
    params.extend(model.parameters())

#criterion = nn.CrossEntropyLoss()
dice_loss = DiceLoss()
#criterion = nn.BCEWithLogitsLoss() 

optimizer = optim.Adam(params, lr=0.000001)

training_pipeline = TrainingPipeline(models=models, criterion=dice_loss, optimizer=optimizer)
training_pipeline.cuda()

voxel_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/voxels'
dataset = VAEDataset(voxel_dir)

print(f"dataset size: {len(dataset)}")
train_loader = DataLoader(dataset, batch_size = 16)
test_loader = DataLoader(dataset, batch_size =16)

# a = dataset.__getitem__(0)  or a[0]
# a.(1, 64, 64 , 64)
# from pretrained models
training_pipeline.evaluate(test_loader)

training_pipeline.train(train_loader, epochs=1000)
training_pipeline.evaluate(test_loader)

training_pipeline.save('./dev/vae.pth')

#training_pipeline.load('./dev/vae.pth')

embed()
