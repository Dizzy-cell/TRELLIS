
import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
from PIL import Image
from trellis.pipelines import TrainingPipeline, TrainingFeaturePipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.dataset.dataset_vae import VAEDataset
from trellis.dataset.dataset_svae import SVAEDataset
#from trellis.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder

from trellis.models.structured_latent_vae import SLatEncoder, SLatGaussianDecoder
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from IPython import embed

json_config = '''
{
    "name": "SLatEncoder",
    "args": {
        "resolution": 64,
        "in_channels": 1024,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "use_fp16": false
    },
    "pretrained": "/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors"
}
'''
config = json.loads(json_config)

feature_vae_encoder = SLatEncoder(**(config['args']))
#feature_vae_encoder.load_state_dict(load_file(config['pretrained']))

# (1024, 64, 64, 64) -> (8, 64, 64, 64) feature

json_config = '''
{
    "name": "SLatGaussianDecoder",
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
            "lr": {
                "_xyz": 1.0,
                "_features_dc": 1.0,
                "_opacity": 1.0,
                "_scaling": 1.0,
                "_rotation": 0.1
            },
            "perturb_offset": true,
            "voxel_size": 1.5,
            "num_gaussians": 32,
            "2d_filter_kernel_size": 0.1,
            "3d_filter_kernel_size": 9e-4,
            "scaling_bias": 4e-3,
            "opacity_bias": 0.1,
            "scaling_activation": "softplus"
        }
    },
    "pretrained": "/root/.cache/huggingface/hub/models--JeffreyXiang--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
}
'''
config = json.loads(json_config)

feature_vae_decoder_gs = SLatGaussianDecoder(**(config['args']))
#feature_vae_decoder_gs.load_state_dict(load_file(config['pretrained']))

# # decoder range: (8, 64,64,64)-> rgb

models = {
    "encoder": feature_vae_encoder,
    "decoder": feature_vae_decoder_gs
}

params = []
for model in models.values():
    params.extend(model.parameters())

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss() 

optimizer = optim.AdamW(params, lr=0.000001)

training_pipeline = TrainingFeaturePipeline(models=models, criterion=criterion, optimizer=optimizer)
training_pipeline.cuda()

# training_pipeline.load('./dev/3000.pth')

struct_latent_dir = '/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16'
features_dir = "/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/features/dinov2_vitl14_reg"
image_dir = "/opt/nas/n/local/yyj/TRELLIS/datasets/ObjaverseXL_sketchfab/renders"
dataset = SVAEDataset(features_dir, image_dir)

print(f"dataset size: {len(dataset)}")
train_loader = DataLoader(dataset, batch_size = 1, shuffle=True)
test_loader = DataLoader(dataset, batch_size = 1)

# a = dataset.__getitem__(0)  or a[0]
# a.(1, 64, 64 , 64)
# from pretrained models
training_pipeline.evaluate(test_loader)

training_pipeline.train(train_loader, epochs=100000)
training_pipeline.evaluate(test_loader)

training_pipeline.save('./dev/vae.pth')
embed()
