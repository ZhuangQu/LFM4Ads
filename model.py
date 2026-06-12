import torch
from torch import nn

import fields


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleDict()
        for field, size in (fields.user | fields.video).items():
            self.tables[field] = nn.Embedding(size, 10)

    def forward(self, batch):
        embeddings = [table(batch[field]) for field, table in self.tables.items()]
        return torch.cat(embeddings, -1)


class DCNv2(nn.Module):
    def __init__(self, dim=360):
        super().__init__()
        self.sparse = Sparse()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(5)])
        self.layers += [nn.Linear(dim, 15)]

    def embed(self, batch):
        return self.sparse(batch)

    def forward(self, batch):
        CRs = [self.embed(batch)]
        for i, layer in enumerate(self.layers):
            if i < 3:
                CRs += [layer(CRs[-1]) * CRs[0] + CRs[-1]]
            else:
                CRs += [layer(CRs[-1].relu())]
        batch["logit"] = CRs[-1][range(len(CRs[-1])), batch["tab"]]
        if hasattr(self, "CRs"):
            for id, CRs in zip(batch["user_id"], torch.stack(CRs[:-1], 1)):
                self.CRs[id] *= 0.99
                self.CRs[id] += CRs


class FeatureUsage(DCNv2):
    def __init__(self, LFM4Ads, method):
        super().__init__(360 if "gate" in method else 370)
        self.LFM4Ads = LFM4Ads
        self.method = method
        self.linear = nn.Linear(
            360 if "CR" in method else 270,
            360 if "gate" in method else 10,
        )

    def embed(self, batch):
        E = self.sparse(batch)
        if "CR" in self.method:
            UR = self.LFM4Ads.CRs[batch["user_id"], int(self.method[-1])]
        else:
            UR = self.LFM4Ads.sparse(batch)[:, :270]
        UR = self.linear(UR)
        if "gate" in self.method:
            return E * UR.sigmoid()
        else:
            return torch.cat([E, UR], -1)


class ModuleUsage(DCNv2):
    def __init__(self, LFM4Ads, method):
        super().__init__()
        if method != "Vanilla":
            self.sparse.load_state_dict(LFM4Ads.sparse.state_dict())
            for i in range(int(method[-1])):
                self.layers[i].load_state_dict(LFM4Ads.layers[i].state_dict())


class ModelUsage(nn.Module):
    def __init__(self, LFM4Ads, method):
        super().__init__()
        self.LFM4Ads = LFM4Ads
        self.method = method
        self.sparse = Sparse()
        self.linear = nn.Linear(360 if "CR" in method else 270, 90)

    def forward(self, batch):
        if self.method == "Retriever":
            E = self.sparse(batch)
        else:
            E = self.LFM4Ads.sparse(batch)
        if "CR" in self.method:
            UR = self.LFM4Ads.CRs[batch["user_id"], int(self.method[-1])]
        else:
            UR = E[:, :270]
        UR = self.linear(UR)
        IR = E[:, -90:]
        batch["logit"] = (UR * IR).sum(-1)
