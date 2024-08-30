import csv, pickle, gzip
import pandas as pd
import numpy as np
from functools import reduce
from tqdm import tqdm 
import torch 
torch.manual_seed(0)
from torch import nn 
import torch.nn.functional as F

def get_mesh_term_embedding():
    with open('../data/mesh-embeddings/mesh_ui_to_id.pickle', 'rb') as stream:
        mesh_ui_to_id = pickle.load(stream)

    with open('../data/mesh-embeddings/term_to_ui.pkl', 'rb') as stream:
        term2ui = pickle.load(stream)
                
    embeddings = {}
    with gzip.open('../data/mesh-embeddings/mesh_embeddings.txt.gz', 'rt') as stream:
        n_embeddings, embedding_dim = stream.readline().strip().split()
        for line in stream:
            splitline = str(line).strip().split()
            idx = int(splitline[0])
            vector = list(map(float, splitline[1:]))
            embeddings[idx] = vector
            assert len(vector) == 256
    return mesh_ui_to_id, term2ui, embeddings
        
mesh_ui_to_id, term2ui, embeddings = get_mesh_term_embedding()

def mesh_term2feature(term):
            MAX_LEN = 40
            get_embeddings = []
            if isinstance(term, list):
                for t in term:
                    try:
                        ui = term2ui[t]
                        get_embeddings.append(embeddings[mesh_ui_to_id[ui]])
                    except:
                        continue
                        # get_embeddings.append(np.zeros(256))
                        # print(f'{t} not found in names')
            elif isinstance(term, str) and '[' in term:
                term = eval(term)
                for t in term:
                    try:
                        ui = term2ui[t]
                        get_embeddings.append(embeddings[mesh_ui_to_id[ui]])
                    except:
                        continue
                        # get_embeddings.append(np.zeros(256))
                        # print(f'{t} not found in embeddings')
            elif isinstance(term, str):
                try:
                    ui = term2ui[term]
                    get_embeddings.append(embeddings[mesh_ui_to_id[ui]])
                except:
                    get_embeddings.append(np.zeros((1, 256)))
                    # print(f'{term} not found in embeddings')
            else:
                get_embeddings.append(np.zeros((1, 256)))
            if len(get_embeddings) > 1:
                get_embeddings = np.array(get_embeddings) # MAX_LEN x 256
            elif len(get_embeddings) == 1:
                get_embeddings = np.array(get_embeddings)
            else:
                get_embeddings = np.zeros((1, 256))
            # assert len(get_embeddings) == 256
            return get_embeddings.astype(np.float32)


class Mesh_Embedding(nn.Sequential):
    def __init__(self, output_dim, highway_num, device ):
        super(Mesh_Embedding, self).__init__()	
        self.input_dim = 256  
        self.output_dim = output_dim 
        self.highway_num = highway_num 
        self.fc = nn.Linear(self.input_dim, output_dim)
        self.f = F.relu
        self.device = device 
        self = self.to(device)

    def forward_single(self, mesh_embedding):
        ## mesh_embedding : xxx,256
        mesh_embedding = torch.tensor(mesh_embedding).to(self.device)
        text_vec = torch.mean(mesh_embedding, 0)
        text_vec = text_vec.view(1,-1)
        return text_vec  # 1, 256

    def forward(self, mesh_embedding):
        result = [self.forward_single(t_mat) for t_mat in mesh_embedding]
        text_mat = torch.cat(result, 0)  #### 32,256
        output = self.f(self.fc(text_mat))
        return output 

    @property
    def embedding_size(self):
        return self.output_dim 

