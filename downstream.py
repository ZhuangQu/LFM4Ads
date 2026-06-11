from sys import argv

import torch

from model import FeatureUsage, ModelUsage, ModuleUsage
from train import train

LFM4Ads = torch.load(argv[2] + ".pt", argv[1], weights_only=False)
file = open(argv[2] + ".csv", "w", 1)
file.write("Scenario,      Method,    AUC\n")


def run(Usage, method):
    model = Usage(LFM4Ads, method).to(argv[1])
    auc = train(model, scenario)
    file.write(f"{scenario:8}, {method:>11}, {auc:.4f}\n")


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
