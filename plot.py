from os import makedirs

import matplotlib.pyplot as plt
import pandas as pd


def query(method):
    df = pd.read_csv("result.csv", skipinitialspace=True)
    mask = (df["Scenario"] == scenario) & (df["Method"] == method)
    return df[mask]["AUC"].mean()


def plot(usage, *methods):
    xticks = ["$Towers$", "$Cross_1$", "$Cross_2$", "$Cross_3$", "$DNN_1$", "$DNN_2$"]
    if usage == "Module":
        xticks += ["$DNN_3$"]
    n = len(xticks)
    plt.figure(figsize=(5, 2))
    for method in methods:
        if "CR" in method:
            data = [query(f"{method}_{i}") for i in range(n)]
            plt.plot(data, marker="o")
            plt.text(n - 0.6, data[-1], method, va="center")
        else:
            plt.axhline(query(method), color="y")
            plt.text(n - 0.6, query(method), method, va="center")
    plt.xticks(range(n), xticks)
    plt.ylabel("AUC")
    plt.title(f"{usage}-Level Usage for Scenario {scenario}")
    makedirs(usage, exist_ok=True)
    plt.savefig(f"{usage}/{scenario}.pdf", bbox_inches="tight")


for scenario in [1, 0, 4, 2, 6, 3, 8, 5]:
    plot("Feature", "SUM", "concat CR", "gating CR")
    plot("Module", "Vanilla", "CR'")
    plot("Model", "TwoTower", "IR & UR", "IR & CR")
