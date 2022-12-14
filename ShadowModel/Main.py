import pickle

import pandas as pd

from JacobianMatrix import run
import numpy as np

data = run("heartbeat", "LR", 1e-5, 1000, seed=14, label=0)

df = pd.DataFrame(data)

df.to_csv("C:/Users/prasa/Desktop/FALL GRA/Aster-main/Aster-main/ShadowModel/attack_data/heartbeat/train_0.csv", header=False, index=False)


