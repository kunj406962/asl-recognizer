"""MLP definition and checkpoint loading (checkpoint written by notebook 03)."""
import numpy as np
import torch
import torch.nn as nn


def build(hidden, dropout, n_classes, n_features=63):
    layers, d = [], n_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, n_classes))   # raw logits
    return nn.Sequential(*layers)


def load_checkpoint(path):
    # the checkpoint holds numpy arrays, so weights_only=False (it is my own file)
    ckpt = torch.load(path, weights_only=False)
    classes = ckpt["classes"]
    model = build(ckpt["hidden"], ckpt["dropout"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, ckpt["mean"], ckpt["std"]


def predict_letter(model, classes, mean, std, feats):
    """63 normalized features -> (label, probability)."""
    x = torch.tensor(((feats - mean) / std)[None], dtype=torch.float32)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    k = int(p.argmax())
    return classes[k], float(p[k])
