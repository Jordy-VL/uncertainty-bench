import pandas as pd
import regex as re

"""
using weights:  /mnt/lerna/models/20news_aleatorics_concrete/weights_05-0.41272.hdf5

Mode nonbayesian ; Batch Inference time: 0.010841s
Mode nonbayesian ; Sample Inference time: 0.00033878125s
"""

"""
model: CLINC150_nodropout ; time per epoch: 32s ; last epoch: 8 ; total time: 256s
"""

columns = ["model", "architecture", "device", "batch", "sample", "batch_mc", "sample_mc"]
cols = ["model", "train time/epoch", "epoch finished", "train runtime"]

with open("train.log", "r") as f:
    train = f.read().split("done")

with open("gpu-timings.log", "r") as f:
    gpus = f.read().split("Not saving!")

with open("cpu-timings.log", "r") as f:
    cpus = f.read().split("Not saving!")


def parse_element(el, device=None):
    d = {col: "" for col in columns}
    d["model"] = re.search("(?:using weights:  /mnt/lerna/models/)(.*)(?:/)", el)
    # d["mode"] = re.search("(?:Mode) (\w+) (?:;)", el)
    d["device"] = device

    d["batch"] = re.search("(?:Mode nonbayesian ; Batch Inference time: )(.+)s", el)
    d["sample"] = re.search("(?:Mode nonbayesian ; :Sample Inference time: )(.+)s", el)

    # if "Mode mc" in el:
    d["batch_mc"] = re.search("(?:Mode mc ; Batch Inference time: )(.+)s", el)
    d["sample_mc"] = re.search("(?:Mode mc ; Sample Inference time: )(.+)s", el)

    for col in columns:
        if col == "device":
            continue
        if d[col]:
            d[col] = d[col].group(1)
    d["architecture"] = "BERT" if "BERT" in d["model"] else "TextCNN"
    return d


def parse_log(el):
    e = {}
    e["model"] = re.search("(?:model: )(.+?) ;", el)
    e["train time/epoch"] = re.search("(?:time per epoch: )(.+?)s", el)
    e["epoch finished"] = re.search("(?:last epoch: )(.+?);", el)
    e["train runtime"] = re.search("(?:total time: )(.+?)s", el)

    for col in cols:
        if e[col]:
            e[col] = e[col].group(1)
    if e["train time/epoch"] and e["epoch finished"]:
        e["train runtime"] = float(e["train time/epoch"]) * int(float(e["epoch finished"]))
    return e


collector = []
for el in gpus:
    if not "using weights" in el:
        continue
    collector.append(parse_element(el, device="gpu"))
for el in cpus:
    if not "using weights" in el:
        continue
    collector.append(parse_element(el, device="cpu"))

df = pd.DataFrame(collector, columns=columns)
df.to_csv("timings.csv", index=False)

################

collector = []
for el in train:
    if not "model" in el:
        continue
    collector.append(parse_log(el))

df = pd.DataFrame(collector, columns=cols)
df.to_csv("train-timings.csv", index=False)
