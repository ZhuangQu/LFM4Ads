from os import makedirs

import matplotlib.pyplot as plt
import pandas as pd


def query(method):
    df = pd.read_csv("result.csv", skipinitialspace=True)
    mask = (df["Scenario"] == scenario) & (df["Method"] == method)
    return df[mask]["AUC"].mean()


def plot(usage, *methods):
    plt.figure(figsize=(4, 2))
    for method in methods:
        if "CR" in method:
            plt.plot([query(f"{method}_{i}") for i in range(6)], marker="o")
            plt.text(5.3, query(f"{method}_5"), method, va="center")
        else:
            plt.axhline(query(method), color="y")
            plt.text(5.3, query(method), method, va="center")
    xticks = ["$Towers$", "$Cross_1$", "$Cross_2$", "$Cross_3$", "$DNN_1$", "$DNN_2$"]
    plt.xticks(range(6), xticks)
    plt.ylabel("AUC")
    plt.title(f"{usage}-Level Usage for Scenario {scenario}")
    makedirs(usage, exist_ok=True)
    plt.savefig(f"{usage}/{scenario}.pdf", bbox_inches="tight")


for scenario in [1, 0, 4, 2, 6, 3, 8, 5]:
    plot("Feature", "SUM", "concat CR", "gate CR")
    plot("Module", "Vanilla", "CR'")
    plot("Model", "Retriever", "IR & UR", "IR & CR")
