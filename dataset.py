import pandas as pd

import fields


class Dataset(pd.DataFrame):
    def __getitem__(self, index):
        return self.iloc[index].to_dict()


def Split(scenario):
    df = pd.read_feather("dataset.feather")
    if scenario != "all":
        df = df.query(f"tab == {scenario}")
    return [
        df.query("            date < 20220503"),
        df.query("20220503 <= date < 20220506"),
        df.query("20220506 <= date           "),
    ]


if __name__ == "__main__":
    df1 = pd.read_csv("KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv")
    df2 = pd.read_csv("KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv")
    df = pd.concat([df1, df2]).sort_values("time_ms")
    df = pd.merge(df, pd.read_csv("KuaiRand-1K/data/user_features_1k.csv"))
    df = pd.merge(df, pd.read_csv("KuaiRand-1K/data/video_features_basic_1k.csv"))
    for field in fields.user | fields.video:
        df[field] = df[field].factorize(use_na_sentinel=False)[0]
    df[fields.all].to_feather("dataset.feather")
