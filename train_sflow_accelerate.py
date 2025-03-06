
import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
from PIL import Image
from trellis.pipelines import TrainingFlowPipeline_accelerate
from trellis.utils import render_utils, postprocessing_utils
from trellis.dataset.dataset_vae import VAEDataset
from trellis.dataset.dataset_flow import FlowDataset
from trellis.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder

from trellis.models.sparse_structure_flow import SparseStructureFlowModel

import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from IPython import embed

json_config = '''
{
    "name": "SparseStructureFlowModel",
    "args": {
        "resolution": 16,
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 1024,
        "cond_channels": 1024,
        "num_blocks": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 1,
        "pe_mode": "ape",
        "qk_rms_norm": true,
        "use_fp16": false
    },
    "pretrained": "/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors"
}
'''
config = json.loads(json_config)

ss_flow = SparseStructureFlowModel(**(config['args']))
#ss_flow.load_state_dict(load_file(config['pretrained'])) # loss from 1.24 to 0.33

# decoder range: [-220, 220]

models = {
    "ss_flow": ss_flow,
}

params = []
for model in models.values():
    params.extend(model.parameters())

criterion = nn.BCEWithLogitsLoss() 

optimizer = optim.AdamW(params, lr=0.000001)

training_pipeline = TrainingFlowPipeline_accelerate(models=models, criterion=criterion, optimizer=optimizer)
training_pipeline.cuda()

#training_pipeline.load('./dev/3000.pth')

feature_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/features/dinov2_vitl14_reg'
latent_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16'
dataset = FlowDataset(feature_dir, latent_dir)


print(f"dataset size: {len(dataset)}")
train_loader = DataLoader(dataset, batch_size = 8, shuffle=True)
test_loader = DataLoader(dataset, batch_size = 2)

# a = dataset.__getitem__(0)  or a[0]
# a.(1, 64, 64 , 64)
# from pretrained models
training_pipeline.evaluate(test_loader)

training_pipeline.train(train_loader, epochs=100000)
training_pipeline.evaluate(test_loader)

training_pipeline.save('./dev/vae.pth')

embed()
