import pandas as pd
from CONFIG import *
import pickle
import os

a=pd.read_csv(os.path.join(LINK_DATA,"score.csv"))
print(a)

score = dict((j["id_event"],j["score"]) for i, j in a.iterrows())
with open(os.path.join(LINK_DATA,"score.pickle"),"wb") as output:
    pickle.dump(score,output)