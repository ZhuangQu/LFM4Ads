from sys import argv

import pandas as pd
import torch

from dataset import Split
from model import DCNv2
from train import infer, train

LFM4Ads = DCNv2().to(argv[1])
train(LFM4Ads, "all")
LFM4Ads.requires_grad_(False)


LFM4Ads.CRs = torch.zeros(1000, 6, 360).to(argv[1])
train_valid_set = pd.concat(Split("all")[:2])
for _ in infer(LFM4Ads, train_valid_set):
    pass
LFM4Ads.CRs = torch.nn.functional.layer_norm(LFM4Ads.CRs, [360])


torch.save(LFM4Ads, argv[2] + ".pt")
