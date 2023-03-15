import os
from urllib.request import urlretrieve
import numpy as np
from torch import nn
from src.VIT_model.modeling import VisionTransformer, CONFIGS


def load_pretrained_VIT_model_triplet():
    os.makedirs("model_checkpoints", exist_ok=True)
    if not os.path.isfile("model_checkpoints/ViT-B_16-224.npz"):
        urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
                    "model_checkpoints/ViT-B_16-224.npz")

    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
    model.load_from(np.load("model_checkpoints/ViT-B_16-224.npz"))
    encoder = nn.Sequential(*[model.transformer.embeddings, model.transformer.encoder])
    return encoder


def load_pretrained_VIT_model_cls(n_classes):
    os.makedirs("model_checkpoints", exist_ok=True)
    if not os.path.isfile("model_checkpoints/ViT-B_16-224.npz"):
        urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
                    "model_checkpoints/ViT-B_16-224.npz")

    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
    model.load_from(np.load("model_checkpoints/ViT-B_16-224.npz"))
    encoder = nn.Sequential(*[model.transformer.embeddings, model.transformer.encoder])
    ViT_embed_dim = 768  # final embedding dimension for ViT-B
    classifier = nn.Linear(ViT_embed_dim, n_classes)

    return classifier, encoder
