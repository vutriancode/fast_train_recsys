from pyvi import ViTokenizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import os
nltk.download('stopwords')
import re
import math
import torch
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE

from CONFIG import *
phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
phoBERT.eval()  # disable dropout (or leave in train mode to finetune)
class BPE():
  bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

args = BPE()
phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
def embedding_document(document):
    doc = ViTokenizer.tokenize(document)
    tokens = phoBERT.encode(doc) 
    if len(tokens) > 256:
        chunks = math.ceil(len(tokens)/256)
        emb = []
        sum_tokens = len(tokens)
        chunks=min(chunks,3)
        for i in range(chunks):
            sum_tokens = sum_tokens - 256
            if sum_tokens > 0:
                emb.append(phoBERT.extract_features(tokens[i*256:(i+1)*256])[0][0])
            else:
                emb.append(phoBERT.extract_features(tokens[i*256:])[0][0])
        emb = torch.mean(torch.stack(emb),0)
    else:
        emb = phoBERT.extract_features(tokens)[0][0]
    return emb
