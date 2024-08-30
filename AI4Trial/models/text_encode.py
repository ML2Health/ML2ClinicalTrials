'''
input:

All_data.csv: All data from the clinical trial dataset

output:

sentence2embedding.pkl (preprocessing)
'''
import csv, pickle 
import pandas as pd
from functools import reduce
from tqdm import tqdm 
import torch 
torch.manual_seed(0)
from torch import nn 
import torch.nn.functional as F

def clean_text(text):
    if pd.isnull(text):
        return None
    if type(text) != str:
        text = ' '.join(text)
    text = text.lower()
    text_split = text.split('\n')
    filter_out_empty_fn = lambda x: len(x.strip())>0
    strip_fn = lambda x:x.strip()
    text_split = list(filter(filter_out_empty_fn, text_split))	
    text_split = list(map(strip_fn, text_split))
    return ''.join(text_split)

def get_all_texts():
    input_file = '../data/All_data.csv'
    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    raw_data = pd.read_csv(input_file)
    raw_data = raw_data.loc[:, [c for c in raw_data.columns if c in text_feature]]
    text_feature = raw_data.columns
    print(len(text_feature), len(raw_data))
    texts = []
    for idx, row in raw_data.iterrows():
        texts.extend([clean_text(row[feature]) for feature in text_feature if type(row[feature])==str or type(row[feature])==list and not pd.isnull(row[feature])])
    return set(texts)

def get_column_texts(c):
    input_file = '../data/All_data.csv'
    raw_data = pd.read_csv(input_file).loc[:, [c]]
    texts = raw_data[c].apply(clean_text).dropna()
    texts = texts.tolist()
    print(len(texts))
    return set(texts)

def save_sentence_bert_dict_pkl_c(c):
    cleaned_sentence_set = get_column_texts(c) 
    from biobert_embedding.embedding import BiobertEmbedding
    biobert = BiobertEmbedding(model_path = '../data/Bio_ClinaBert/')
    def text2vec(text):
        return biobert.sentence_vector(text)
    text_sentence_2_embedding = dict()
    for sentence in tqdm(cleaned_sentence_set):
        try:
            text_sentence_2_embedding[sentence] = text2vec(sentence)
        except:
            continue
    pickle.dump(text_sentence_2_embedding, open(f'../data/{c}_sentences2embedding.pkl', 'wb'))
    return

def save_sentence_bert_dict_pkl():
    cleaned_sentence_set = get_all_texts() 
    from biobert_embedding.embedding import BiobertEmbedding
    biobert = BiobertEmbedding(model_path = '../data/Bio_ClinaBert/')
    def text2vec(text):
        return biobert.sentence_vector(text)
    text_sentence_2_embedding = dict()
    for sentence in tqdm(cleaned_sentence_set):
        try:
            text_sentence_2_embedding[sentence] = text2vec(sentence)
        except:
            continue
    pickle.dump(text_sentence_2_embedding, open('../data/othersentences2embedding.pkl', 'wb'))
    return 

def load_sentence_2_vec():
    sentence_2_vec = pickle.load(open('../data/othersentences2embedding.pkl', 'rb'))
    return sentence_2_vec 

sentence_2_vec = load_sentence_2_vec()
def text2feature(text_lst):
    text_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in text_lst if sentence in sentence_2_vec]
    if text_feature == []:
        text_feature = torch.zeros(1,768)
    else:
        text_feature = torch.cat(text_feature, 0)
    return text_feature

class Text_Embedding(nn.Sequential):
    def __init__(self, output_dim, highway_num, device ):
        super(Text_Embedding, self).__init__()	
        self.input_dim = 768  
        self.output_dim = output_dim 
        self.highway_num = highway_num 
        self.fc = nn.Linear(self.input_dim, output_dim)
        self.f = F.relu
        self.device = device 
        self = self.to(device)

    def forward_single(self, text_feature):
        ## text_feature : xxx,768 
        text_feature = text_feature.to(self.device)
        text_vec = torch.mean(text_feature, 0)
        text_vec = text_vec.view(1,-1)
        return text_vec  # 1, 768 

    def forward(self, text_feature):
        result = [self.forward_single(t_mat) for t_mat in text_feature]
        text_mat = torch.cat(result, 0)  #### 32,768
        output = self.f(self.fc(text_mat))
        return output 

    @property
    def embedding_size(self):
        return self.output_dim 

if __name__ == "__main__":
    # texts = get_all_texts()
    # split_texts(texts)
    input_file = '../data/All_data.csv'
    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    with open(input_file, 'r') as csvfile:
        row = csvfile.readline()
        # rows = list(csv.reader(csvfile, delimiter = ','))[0]
    col = [c for c in row.split(',') if c in text_feature]
    for c in col[1:]:
        print(c)
        save_sentence_bert_dict_pkl_c(c) 