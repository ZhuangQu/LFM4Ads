from sys import argv

import pandas as pd
import torch

from dataset import Split
from model import DCNv2, FeatureUsage, ModelUsage, ModuleUsage
from train import infer, train

_, device, path = argv


def run(Usage, method):
    model = Usage(LFM4Ads, method).to(device)
    auc = train(model, scenario)
    file = open(path, "a", 1)
    if file.tell() == 0:
        file.write("Scenario,      Method,    AUC\n")
    file.write(f"{scenario:8}, {method:>11}, {auc:.4f}\n")


print("pretrain LFM4Ads ...")
LFM4Ads = DCNv2().to(device)
train(LFM4Ads, "all")
LFM4Ads.requires_grad_(False)


print("aggregate CRs ...")
LFM4Ads.CRs = torch.zeros(1000, 6, 360).to(device)
train_valid_set = pd.concat(Split("all")[:2])
for _ in infer(LFM4Ads, train_valid_set):
    pass
LFM4Ads.CRs = torch.nn.functional.layer_norm(LFM4Ads.CRs, [360])


print("train downstream models ...")
for scenario in [1, 0, 4, 2, 6, 3, 8, 5]:
    run(FeatureUsage, "SUM")
    run(FeatureUsage, "concat CR_0")  # Towers
    run(FeatureUsage, "concat CR_1")  # Cross_1
    run(FeatureUsage, "concat CR_2")  # Cross_2
    run(FeatureUsage, "concat CR_3")  # Cross_3
    run(FeatureUsage, "concat CR_4")  # DNN_1
    run(FeatureUsage, "concat CR_5")  # DNN_2
    run(FeatureUsage, "gate CR_0")  # Towers
    run(FeatureUsage, "gate CR_1")  # Cross_1
    run(FeatureUsage, "gate CR_2")  # Cross_2
    run(FeatureUsage, "gate CR_3")  # Cross_3
    run(FeatureUsage, "gate CR_4")  # DNN_1
    run(FeatureUsage, "gate CR_5")  # DNN_2

    run(ModuleUsage, "Vanilla")
    run(ModuleUsage, "CR'_0")  # Towers
    run(ModuleUsage, "CR'_1")  # Cross_1
    run(ModuleUsage, "CR'_2")  # Cross_2
    run(ModuleUsage, "CR'_3")  # Cross_3
    run(ModuleUsage, "CR'_4")  # DNN_1
    run(ModuleUsage, "CR'_5")  # DNN_2
    run(ModuleUsage, "CR'_6")  # DNN_3

    run(ModelUsage, "Retriever")
    run(ModelUsage, "IR & UR")
    run(ModelUsage, "IR & CR_0")  # Towers
    run(ModelUsage, "IR & CR_1")  # Cross_1
    run(ModelUsage, "IR & CR_2")  # Cross_2
    run(ModelUsage, "IR & CR_3")  # Cross_3
    run(ModelUsage, "IR & CR_4")  # DNN_1
    run(ModelUsage, "IR & CR_5")  # DNN_2
