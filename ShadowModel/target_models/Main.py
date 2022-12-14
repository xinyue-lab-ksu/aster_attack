import pickle

import pandas as pd

from JacobianMatrix import run
import numpy as np

data = run("diabetes", "LR", 1e-5, 180, seed=14, label=1)

df = pd.DataFrame(data)

df.to_csv("C:/Users/prasa/Desktop/FALL GRA/Aster-main/Aster-main/ShadowModel/attack_data/train_1.csv", header=False, index=False)


