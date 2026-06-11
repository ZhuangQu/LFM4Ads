from copy import deepcopy

import torch
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm

import fields
from dataset import Dataset, Split


def infer(model, dataset, train=False):
    model.train(train)
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        Dataset(dataset),
        batch_size=10000,
        num_workers=10,
    )
    for batch in tqdm(loader):
        for field in fields.all:
            batch[field] = batch[field].to(device).int()
        with torch.inference_mode(not train):
            model(batch)
            yield batch["logit"], batch["is_click"].float()


def evaluate(model, dataset):
    AUC = BinaryAUROC()
    for logit, label in infer(model, dataset):
        AUC.update(logit, label)
    return AUC.compute()


def train(model, scenario):
    train_set, valid_set, test_set = Split(scenario)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    auc_best = 0
    while True:
        for logit, label in infer(model, train_set, True):
            criterion(logit, label).backward()
            optimizer.step()
            optimizer.zero_grad()
        auc = evaluate(model, valid_set)
        print(f"valid AUC: {auc:.4f}")
        if auc_best < auc - 0.001:
            auc_best = auc
            state_dict = deepcopy(model.state_dict())
        else:
            model.load_state_dict(state_dict)
            return evaluate(model, test_set)
