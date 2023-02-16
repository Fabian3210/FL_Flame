import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = os.path.join(os.getcwd(), "experiment_setting.csv")

df = pd.read_csv(path)
df = df.set_index("number")
df.server_acc = [eval(x) if isinstance(x, str) else np.nan for x in df.server_acc]

fig, ax = plt.subplots()
ax.plot(df.loc[1].server_acc, color="black", label="FL_IID")
ax.plot(df.loc[3].server_acc, color="orange", label="FL_IID_BD")
ax.plot(df.loc[5].server_acc, color="blue", label="FL_IID_RL")

ax.plot(df.loc[4].server_acc, color="black", label="FL_nonIID", linestyle="dashed")
#ax.plot(df.loc[6].server_acc, color="orange", label="FL_nonIID_BD", linestyle="dashed")
ax.plot(df.loc[7].server_acc, color="blue", label="FL_nonIID_RL", linestyle="dashed")

ax.grid()
ax.legend()
fig.tight_layout()
plt.show()


from models import *
import copy

model = FashionCNN()
poison_rate = 0.8
benign_model = copy.deepcopy(model)
benign_model.load_state_dict(model.state_dict())
state_dict = {}
for key, value in model.state_dict().items():
    mul = (np.random.uniform(-poison_rate, poison_rate, list(value.shape)))
    state_dict[key] = torch.multiply(value, torch.Tensor(mul))
model.load_state_dict(state_dict)
self.model.load_state_dict(state_dict)
