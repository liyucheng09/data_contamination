# %%
import datasets

# %%
ds = datasets.load_from_disk('pred_out')

# %%
ds

# %%
len([i['Min_20.0% Prob'] for i in ds['pred']])

# %%
import numpy as np

# %%
p = np.array([i['Min_20.0% Prob'] for i in ds['pred']])

# %%
p

# %%
np.nanpercentile(p,8.4)

# %%
import pandas

# %%
import json

# %%
with open('model_predictions/llama_2_70b.json') as f:
    data = json.load(f)

# %%
data

# %%
d = data['hellaswag']

# %%
d

# %%
clean_g = []
clean_p = []
dirty_g = []
dirty_p = []

# %%
for i in ds:
    minK = i['pred']['Min_20.0% Prob']
    id_ = i['ind']

    # dirty
    if mink < 6.876:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(d[id_]['pred'])
    else:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(d[id_]['pred'])
        

# %%
for i in ds:
    minK = i['pred']['Min_20.0% Prob']
    id_ = i['ind']

    # dirty
    if minK < 6.876:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(d[id_]['pred'])
    else:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(d[id_]['pred'])
        

# %%
for i in ds:
    minK = i['pred']['Min_20.0% Prob']
    id_ = str(i['ind'])

    # dirty
    if minK < 6.876:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(d[id_]['pred'])
    else:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(d[id_]['pred'])
        

# %%
from sklearn.metrics import accuracy_score


# %%
clean_g = []
clean_p = []
dirty_g = []
dirty_p = []

# %%
for i in ds:
    minK = i['pred']['Min_20.0% Prob']
    id_ = str(i['ind'])

    # dirty
    if minK < 6.876:
        dirty_g.append(d[id_]['gold'])
        dirty_p.append(str(d[id_]['pred']))
    else:
        clean_g.append(d[id_]['gold'])
        clean_p.append(str(d[id_]['pred']))
        

# %%
accuracy_score(clean_g, clean_p)

# %%
accuracy_score(dirty_g, dirty_p)


