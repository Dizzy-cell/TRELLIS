
import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
from PIL import Image
from trellis.pipelines import TrainingFeatureMeshPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.dataset.dataset_vae import VAEDataset
from trellis.dataset.dataset_smesh import SMeshDataset
from trellis.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder

from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.models.structured_latent_vae import SLatMeshDecoder

import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from IPython import embed

json_config = '''
{
    "name": "SLatMeshDecoder",
    "args": {
        "resolution": 64,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "use_fp16": false,
        "representation_config": {
            "use_color": true
        }
    },
    "pretrained": "/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
}
'''
config = json.loads(json_config)

meshdecoder = SLatMeshDecoder(**(config['args']))
#meshdecoder.load_state_dict(load_file(config['pretrained'])) # loss from 1.24 to 0.33

# decoder range: [-220, 220]

models = {
    "decoder": meshdecoder,
}

params = []
for model in models.values():
    params.extend(model.parameters())

criterion = nn.BCEWithLogitsLoss() 

optimizer = optim.AdamW(params, lr=0.0001)

training_pipeline = TrainingFeatureMeshPipeline(models=models, criterion=criterion, optimizer=optimizer)
training_pipeline.cuda()

#training_pipeline.load('./dev/svae_90.pth')

feature_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/features/dinov2_vitl14_reg'
latent_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16'
image_dir = "/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/renders"
dataset = SMeshDataset(feature_dir, latent_dir, image_dir)

print(f"dataset size: {len(dataset)}")
train_loader = DataLoader(dataset, batch_size = 1, shuffle=True)
test_loader = DataLoader(dataset, batch_size = 1)

# a = dataset.__getitem__(0)  or a[0]
# a.(1, 64, 64 , 64)
# from pretrained models
training_pipeline.evaluate(test_loader)

training_pipeline.train(train_loader, epochs=100)
training_pipeline.evaluate(test_loader)

training_pipeline.save('./dev/vae.pth')

embed()
