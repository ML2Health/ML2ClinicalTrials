from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score, precision_score, auc,
    recall_score, accuracy_score, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from copy import deepcopy 
import numpy as np 
from tqdm import tqdm 
import torch, os 
torch.manual_seed(0)
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
from models.module import Highway, GCN 
# from DANet.model.TrialDANet import TrialDANet
from functools import reduce 
import pickle

def print2file(file_name, *content):
    content = ' '.join([str(key) for key in content])
    with open(file_name, 'a') as fout:
        fout.write(content + '\n')
    print(content)



class Interaction(nn.Sequential):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, tabular_encoder, text_encoder, mesh_encoder,
                    device, 
                    num_classes,
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0, 
                    ):
        super(Interaction, self).__init__()
        self.molecule_encoder = molecule_encoder 
        self.disease_encoder = disease_encoder 
        self.protocol_encoder = protocol_encoder
        self.tabular_encoder = tabular_encoder
        self.text_encoder = text_encoder
        self.mesh_encoder = mesh_encoder
        self.num_classes = num_classes
        self.global_embed_size = global_embed_size 
        self.highway_num_layer = highway_num_layer 
        self.feature_dim = self.molecule_encoder.embedding_size + self.disease_encoder.embedding_size + self.protocol_encoder.embedding_size + self.tabular_encoder.embedding_size + \
                            self.text_encoder.embedding_size + self.mesh_encoder.embedding_size
        self.epoch = epoch 
        self.lr = lr 
        self.weight_decay = weight_decay 
        self.save_name = prefix_name + '_interaction'

        self.f = F.relu
        if self.num_classes == 1:
            self.loss = nn.BCEWithLogitsLoss()
        elif self.num_classes > 1:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()
        
        ##### NN 
        self.encoder2interaction_fc = nn.Linear(self.feature_dim, self.global_embed_size).to(device)
        self.encoder2interaction_highway = Highway(self.global_embed_size, self.highway_num_layer).to(device)
        self.pred_nn = nn.Linear(self.global_embed_size, num_classes if num_classes > 0 else 1)

        self.device = device 
        self = self.to(device)

    def feed_lst_of_module(self, input_feature, lst_of_module):
        x = input_feature
        for single_module in lst_of_module:
            x = self.f(single_module(x))
        return x

    def forward_get_six_encoders(self, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst):
        molecule_embed = self.molecule_encoder.forward_smiles_lst_lst(smiles_lst2)
        icd_embed = self.disease_encoder.forward_code_lst3(icdcode_lst3)
        protocol_embed = self.protocol_encoder.forward(criteria_lst)
        tabular_embed = self.tabular_encoder.forward(tabular_lst)
        text_embed = self.text_encoder.forward(text_lst)
        mesh_embed = self.mesh_encoder.forward(mesh_lst)
        return molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed

    def forward_encoder_2_interaction(self, molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed):
        encoder_embedding = torch.cat([molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed], 1)
        # interaction_embedding = self.feed_lst_of_module(encoder_embedding, [self.encoder2interaction_fc, self.encoder2interaction_highway])
        h = self.encoder2interaction_fc(encoder_embedding)
        h = self.f(h)
        h = self.encoder2interaction_highway(h)
        interaction_embedding = self.f(h)
        return interaction_embedding 

    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst):
        molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed = \
            self.forward_get_six_encoders(smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst)
        interaction_embedding = self.forward_encoder_2_interaction(molecule_embed, 
                        icd_embed, protocol_embed, tabular_embed,  text_embed, mesh_embed)
        output = self.pred_nn(interaction_embedding)
        return output ### 32, 1

    def evaluation(self, predict_all, label_all, threshold = 0.5):
        import pickle, os
        from sklearn.metrics import roc_curve, precision_recall_curve
        with open("predict_label.txt", 'w') as fout:
            for i,j in zip(predict_all, label_all):
                fout.write(str(i)[:4] + '\t' + str(j)[:4]+'\n')
        if self.num_classes == 1:
            auc_score = roc_auc_score(label_all, predict_all)
            figure_folder = "figure"
            #### ROC-curve 
            fpr, tpr, thresholds = roc_curve(label_all, predict_all, pos_label=1)
            # roc_curve =plt.figure()
            # plt.plot(fpr,tpr,'-',label=self.save_name + ' ROC Curve ')
            # plt.legend(fontsize = 15)
            #plt.savefig(os.path.join(figure_folder,name+"_roc_curve.png"))
            #### PR-curve
            precision, recall, thresholds = precision_recall_curve(label_all, predict_all)
            # plt.plot(recall,precision, label = self.save_name + ' PR Curve')
            # plt.legend(fontsize = 15)
            # plt.savefig(os.path.join(figure_folder,self.save_name + "_pr_curve.png"))
            label_all = [int(i) for i in label_all]
            float2binary = lambda x:0 if x<threshold else 1
            predict_all = list(map(float2binary, predict_all))
            f1score = f1_score(label_all, predict_all)
            prauc_score = average_precision_score(label_all, predict_all)
            # print(predict_all)
            precision = precision_score(label_all, predict_all, zero_division=0.0)
            recall = recall_score(label_all, predict_all)
            accuracy = accuracy_score(label_all, predict_all)
            predict_1_ratio = sum(predict_all) / len(predict_all)
            label_1_ratio = sum(label_all) / len(label_all)
            return auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio 

        elif self.num_classes > 1:
            predict_all_x = np.max(predict_all, axis=1, keepdims=True)
            e_x = np.exp(predict_all - predict_all_x)
            predict_all = e_x / e_x.sum(axis=1, keepdims=True)
            auc_score = roc_auc_score(label_all, predict_all, average="macro", multi_class="ovr")
            label_all = [int(i) for i in label_all]
            float2cls = lambda x:np.argmax(x)
            predict_cls = list(map(float2cls, predict_all))
            f1score = f1_score(label_all, predict_cls, average="macro")
            precision = precision_score(label_all, predict_cls, average="macro", zero_division=0.0)
            recall = recall_score(label_all, predict_cls, average="macro")
            accuracy = accuracy_score(label_all, predict_cls)

            encoder = OneHotEncoder(sparse_output=False)
            one_hot_labels = encoder.fit_transform(np.array(label_all).reshape(-1, 1))
            # print("One-hot encoded labels:")
            # print(one_hot_labels)
            pr_aucs = []
            for i in range(predict_all.shape[1]):
                prec, recal, _ = precision_recall_curve(one_hot_labels[:, i],predict_all[:, i])
                pr_auc = auc(recal, prec)
                pr_aucs.append(pr_auc)
            prauc_score = np.mean(pr_aucs)
            
            predict_1_ratio = 0#sum(label_all) / len(predict_cls)  # no sense
            label_1_ratio = 0# sum(label_all) / len(label_cls) # no sense

            return auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio 
            # print(predict_all)
        elif self.num_classes == 0:
            mae = mean_absolute_error(label_all, predict_all)
            mse = mean_squared_error(label_all, predict_all)
            correlation, p_value = pearsonr(label_all, predict_all)
            rmse = np.sqrt(mse)
            return mae, mse, rmse, correlation, p_value
        
    def testloader_to_lst(self, dataloader):
        '''
        '''
        nctid_lst, label_lst, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst = [], [], [], [], [], [], []
        for nctid, label, smiles, icdcode, criteria, tabular, text, mesh in dataloader:
            nctid_lst.extend(nctid)
            label_lst.extend([i.item() for i in label])
            smiles_lst2.extend(smiles)
            icdcode_lst3.extend(icdcode)
            criteria_lst.extend(criteria)
            tabular_lst.extend(tabular)
            text_lst.extend(text)
            mesh_lst.extend(mesh)
        length = len(nctid_lst)
        assert length == len(smiles_lst2) and length == len(icdcode_lst3) and length == len(tabular_lst)
        return nctid_lst, label_lst, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst, length 

    def generate_predict(self, dataloader):
        '''
        '''
        whole_loss = 0 
        label_all, predict_all, nctid_all = [], [], []
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst in dataloader:
            nctid_all.extend(nctid_lst)
            label_vec = label_vec.to(self.device)
            output = self.forward(smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst)
            if self.num_classes < 2:
                output = output.view(-1)
                label_vec = label_vec.float()
            loss = self.loss(output, label_vec)
            whole_loss += loss.item()
            predict_all.extend([i.detach().numpy() for i in output])
            label_all.extend([i.item() for i in label_vec])

        return whole_loss, predict_all, label_all, nctid_all

    def bootstrap_test(self, dataloader, sample_num = 20):
        # if validloader is not None:
        # 	best_threshold = self.select_threshold_for_binary(validloader)
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        from HINT.utils import plot_hist
        plt.clf()
        prefix_name = self.save_name.replace('logs/', 'logs/figure/') 
        os.makedirs(prefix_name, exist_ok=True)
        plot_hist(prefix_name, predict_all, label_all)		
        def bootstrap(length, sample_num):
            idx = [i for i in range(length)]
            from random import choices 
            bootstrap_idx = [choices(idx, k = length) for i in range(sample_num)]
            return bootstrap_idx 
        results_lst = []
        bootstrap_idx_lst = bootstrap(len(predict_all), sample_num = sample_num)
        for bootstrap_idx in bootstrap_idx_lst: 
            bootstrap_label = [label_all[idx] for idx in bootstrap_idx]		
            bootstrap_predict = [predict_all[idx] for idx in bootstrap_idx]
            results = self.evaluation(bootstrap_predict, bootstrap_label, threshold = best_threshold)
            results_lst.append(results)
        self.train() 
        if self.num_classes > 0:
            auc = [results[0] for results in results_lst]
            f1score = [results[1] for results in results_lst]
            prauc_score = [results[2] for results in results_lst]
            precision = [results[3] for results in results_lst]
            recall = [results[4] for results in results_lst]
            accuracy = [results[5] for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "PR-AUC   mean: "+str(np.mean(prauc_score))[:6], "std: "+str(np.std(prauc_score))[:6])
            print2file(self.save_name+'.txt',  "F1       mean: "+str(np.mean(f1score))[:6], "std: "+str(np.std(f1score))[:6])
            print2file(self.save_name+'.txt',  "ROC-AUC  mean: "+ str(np.mean(auc))[:6], "std: " + str(np.std(auc))[:6])
            print2file(self.save_name+'.txt',  "Precision mean: "+str(np.mean(precision))[:6], "std: "+str(np.std(precision))[:6])
            print2file(self.save_name+'.txt',  "Recall   mean: "+str(np.mean(recall))[:6], "std: "+str(np.std(recall))[:6])
            print2file(self.save_name+'.txt',  "Accuracy mean: "+str(np.mean(accuracy))[:6], "std: "+str(np.std(accuracy))[:6])
        else:
            mae = [results[0] for results in results_lst]
            mse = [results[1] for results in results_lst]
            rmse = [results[2] for results in results_lst]
            correlation = [results[3] for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "MAE      mean: "+str(np.mean(mae))[:6], "std: "+str(np.std(mae))[:6])
            print2file(self.save_name+'.txt',  "MSE      mean: "+str(np.mean(mse))[:6], "std: "+str(np.std(mse))[:6])
            print2file(self.save_name+'.txt',  "RMSE     mean: "+str(np.mean(rmse))[:6], "std: "+str(np.std(rmse))[:6])
            print2file(self.save_name+'.txt',  "Correlat mean: "+str(np.mean(correlation))[:6], "std: "+str(np.std(correlation))[:6])


        # for nctid, label, predict in zip(nctid_all, label_all, predict_all):
        #     if (predict > 0.5 and label == 0) or (predict < 0.5 and label == 1):
        #         print(nctid, label, str(predict)[:5])

        # nctid2predict = {nctid:predict for nctid, predict in zip(nctid_all, predict_all)} 
        # pickle.dump(nctid2predict, open('results/nctid2predict.pkl', 'wb'))
        return nctid_all, predict_all 

    def ongoing_test(self, dataloader, sample_num = 20):
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader) 
        self.train() 
        return nctid_all, predict_all 
        
    def test(self, dataloader, return_loss = True, validloader=None):
        # if validloader is not None:
        # 	best_threshold = self.select_threshold_for_binary(validloader)
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        # from HINT.utils import plot_hist
        # plt.clf()
        # prefix_name = "./figure/" + self.save_name 
        # plot_hist(prefix_name, predict_all, label_all)
        self.train()
        if return_loss:
            return whole_loss
        else:
            print_num = 5
            results_lst = self.evaluation(predict_all, label_all, threshold = best_threshold)
            if self.num_classes > 0:
                auc, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio = results_lst
                print2file(self.save_name+'.txt', "==================Test==================")
                print2file(self.save_name+'.txt',  "PR-AUC   mean: "+str(prauc_score)[:6])
                print2file(self.save_name+'.txt',  "F1       mean: "+str(f1score)[:6])
                print2file(self.save_name+'.txt',  "ROC-AUC  mean: "+ str(auc)[:6])
                print2file(self.save_name+'.txt',  "Precision mean: "+str(precision)[:6])
                print2file(self.save_name+'.txt',  "Recall   mean: "+str(recall)[:6])
                print2file(self.save_name+'.txt',  "Accuracy mean: "+str(accuracy)[:6])
            else:
                mae, mse, rmse, correlation, p_value = results_lst
                print2file(self.save_name+'.txt', "==================Test==================")
                print2file(self.save_name+'.txt',  "MAE      mean: "+str(mae)[:6])
                print2file(self.save_name+'.txt',  "MSE      mean: "+str(mse)[:6])
                print2file(self.save_name+'.txt',  "RMSE     mean: "+str(rmse)[:6])
                print2file(self.save_name+'.txt',  "Correlat mean: "+str(correlation)[:6])
            return results_lst

    def learn(self, train_loader, valid_loader, test_loader):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        train_loss_record = [] 
        valid_loss = self.test(valid_loader, return_loss=True)
        valid_loss_record = [valid_loss]
        best_valid_loss = valid_loss
        best_model = deepcopy(self)
        for ep in tqdm(range(self.epoch)):
            for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst in train_loader:
                label_vec = label_vec.to(self.device)
                output = self.forward(smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst)#### 32, 1 -> 32, ||  label_vec 32,
                if self.num_classes < 2:
                    output = output.view(-1)  
                    label_vec = label_vec.float()
                loss = self.loss(output, label_vec)
                train_loss_record.append(loss.item())
                opt.zero_grad() 
                loss.backward() 
                opt.step()
            valid_loss = self.test(valid_loader, return_loss=True)
            valid_loss_record.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                best_model = deepcopy(self)

        self.plot_learning_curve(train_loss_record, valid_loss_record)
        self = deepcopy(best_model)
        results_lst = self.test(test_loader, return_loss = False, validloader = valid_loader)

    def plot_learning_curve(self, train_loss_record, valid_loss_record):
        plt.plot(train_loss_record)
        plt.savefig(self.save_name + '_train_loss.jpg')
        plt.clf() 
        plt.plot(valid_loss_record)
        plt.savefig(self.save_name + '_valid_loss.jpg')
        plt.clf() 

    def select_threshold_for_binary(self, validloader):
        _, prediction, label_all, nctid_all = self.generate_predict(validloader)
        best_f1 = 0
        for threshold in prediction:
            float2binary = lambda x:0 if x<threshold else 1
            predict_all = list(map(float2binary, prediction))
            f1score = precision_score(label_all, predict_all, zero_division=0.0)        
            if f1score > best_f1:
                best_f1 = f1score 
                best_threshold = threshold
        return best_threshold 


class MultiModel(Interaction):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, tabular_encoder, text_encoder, mesh_encoder,
                    device, 
                    num_classes,
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0, ):
        super(MultiModel, self).__init__(molecule_encoder = molecule_encoder, 
                                   disease_encoder = disease_encoder, 
                                   protocol_encoder = protocol_encoder,
                                   tabular_encoder = tabular_encoder,
                                   text_encoder = text_encoder,
                                   mesh_encoder = mesh_encoder,
                                   device = device,  
                                   num_classes = num_classes,
                                   global_embed_size = global_embed_size, 
                                   prefix_name = prefix_name, 
                                   highway_num_layer = highway_num_layer,
                                   epoch = epoch,
                                   lr = lr, 
                                   weight_decay = weight_decay, 
                                   ) 
        self.save_name = prefix_name + 'Six_Modal'


        #### risk of disease 
        self.risk_disease_fc = nn.Linear(self.disease_encoder.embedding_size, self.global_embed_size)
        self.risk_disease_higway = Highway(self.global_embed_size, self.highway_num_layer)

        #### augment interaction 
        self.augment_interaction_fc = nn.Linear(self.global_embed_size*2, self.global_embed_size)
        self.augment_interaction_highway = Highway(self.global_embed_size, self.highway_num_layer)

        #### ADMET 
        self.admet_model = []
        for i in range(5):
            admet_fc = nn.Linear(self.molecule_encoder.embedding_size, self.global_embed_size).to(device)
            admet_highway = Highway(self.global_embed_size, self.highway_num_layer).to(device)
            self.admet_model.append(nn.ModuleList([admet_fc, admet_highway])) 
        self.admet_model = nn.ModuleList(self.admet_model)

        #### PK 
        self.pk_fc = nn.Linear(self.global_embed_size*5, self.global_embed_size)
        self.pk_highway = Highway(self.global_embed_size, self.highway_num_layer)

        #### trial node 
        self.trial_fc = nn.Linear(self.global_embed_size*2, self.global_embed_size)
        self.trial_highway = Highway(self.global_embed_size, self.highway_num_layer)

        ## self.pred_nn = nn.Linear(self.global_embed_size, 1)

        self.device = device 
        self = self.to(device)


    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst, if_gnn = False):
        ### encoder for molecule, disease and protocol
        molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed = self.forward_get_six_encoders(smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst)
        ### interaction 
        interaction_embedding = self.forward_encoder_2_interaction(molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed)
        ### risk of disease 
        risk_of_disease_embedding = self.feed_lst_of_module(input_feature = icd_embed, 
                                                            lst_of_module = [self.risk_disease_fc, self.risk_disease_higway])
        ### augment interaction   
        augment_interaction_input = torch.cat([interaction_embedding, risk_of_disease_embedding], 1)
        augment_interaction_embedding = self.feed_lst_of_module(input_feature = augment_interaction_input, 
                                                                lst_of_module = [self.augment_interaction_fc, self.augment_interaction_highway])
        ### admet 
        admet_embedding_lst = []
        for idx in range(5):
            admet_embedding = self.feed_lst_of_module(input_feature = molecule_embed, 
                                                      lst_of_module = self.admet_model[idx])
            admet_embedding_lst.append(admet_embedding)
        ### pk 
        pk_input = torch.cat(admet_embedding_lst, 1)
        pk_embedding = self.feed_lst_of_module(input_feature = pk_input, 
                                               lst_of_module = [self.pk_fc, self.pk_highway])
        ### trial 
        trial_input = torch.cat([pk_embedding, augment_interaction_embedding], 1)
        trial_embedding = self.feed_lst_of_module(input_feature = trial_input, 
                                                  lst_of_module = [self.trial_fc, self.trial_highway])
        output = self.pred_nn(trial_embedding)
        if if_gnn == False:
            return output 
        else:
            embedding_lst = [molecule_embed, icd_embed, protocol_embed, interaction_embedding, risk_of_disease_embedding, \
                             augment_interaction_embedding] + admet_embedding_lst + [pk_embedding, trial_embedding]
            return embedding_lst
            
    def init_pretrain(self, admet_model):
        self.molecule_encoder = admet_model.molecule_encoder


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()
        print(f'Mix Regression and Classification Loss')
        self.loss_cls = nn.BCEWithLogitsLoss()
        self.loss_reg = nn.MSELoss()

    def forward(self, output, label):
        return self.loss_cls(output[:, 1], label[:, 1]) + self.loss_reg(output[:, 0], label[:, 0])

class Multi_2_head(Interaction):
    def __init__(self, molecule_encoder, disease_encoder, protocol_encoder, tabular_encoder, text_encoder, mesh_encoder,
                    device, 
                    num_classes,
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0, ):
        super(Multi_2_head, self).__init__(molecule_encoder = molecule_encoder, 
                                   disease_encoder = disease_encoder, 
                                   protocol_encoder = protocol_encoder,
                                   tabular_encoder = tabular_encoder,
                                   text_encoder = text_encoder,
                                   mesh_encoder = mesh_encoder,
                                   device = device,  
                                   num_classes = num_classes,
                                   global_embed_size = global_embed_size, 
                                   prefix_name = prefix_name, 
                                   highway_num_layer = highway_num_layer,
                                   epoch = epoch,
                                   lr = lr, 
                                   weight_decay = weight_decay, 
                                   ) 
        self.save_name = prefix_name + 'Two_Head'
        
        self.loss = MixLoss()

        #### risk of disease 
        self.risk_disease_fc = nn.Linear(self.disease_encoder.embedding_size, self.global_embed_size)
        self.risk_disease_higway = Highway(self.global_embed_size, self.highway_num_layer)

        #### augment interaction 
        self.augment_interaction_fc = nn.Linear(self.global_embed_size*2, self.global_embed_size)
        self.augment_interaction_highway = Highway(self.global_embed_size, self.highway_num_layer)

        #### ADMET 
        self.admet_model = []
        for i in range(5):
            admet_fc = nn.Linear(self.molecule_encoder.embedding_size, self.global_embed_size).to(device)
            admet_highway = Highway(self.global_embed_size, self.highway_num_layer).to(device)
            self.admet_model.append(nn.ModuleList([admet_fc, admet_highway])) 
        self.admet_model = nn.ModuleList(self.admet_model)

        #### PK 
        self.pk_fc = nn.Linear(self.global_embed_size*5, self.global_embed_size)
        self.pk_highway = Highway(self.global_embed_size, self.highway_num_layer)

        #### trial node 
        self.trial_fc = nn.Linear(self.global_embed_size*2, self.global_embed_size)
        self.trial_highway = Highway(self.global_embed_size, self.highway_num_layer)

        self.pred_cls = nn.Linear(self.global_embed_size, 1)
        self.pred_nn = nn.Linear(self.global_embed_size, 1)

        self.num_classes = 99

        self.device = device 
        self = self.to(device)


    def forward(self, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst, if_gnn = False):
        ### encoder for molecule, disease and protocol
        molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed = self.forward_get_six_encoders(smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst)
        ### interaction 
        interaction_embedding = self.forward_encoder_2_interaction(molecule_embed, icd_embed, protocol_embed, tabular_embed, text_embed, mesh_embed)
        ### risk of disease 
        risk_of_disease_embedding = self.feed_lst_of_module(input_feature = icd_embed, 
                                                            lst_of_module = [self.risk_disease_fc, self.risk_disease_higway])
        ### augment interaction   
        augment_interaction_input = torch.cat([interaction_embedding, risk_of_disease_embedding], 1)
        augment_interaction_embedding = self.feed_lst_of_module(input_feature = augment_interaction_input, 
                                                                lst_of_module = [self.augment_interaction_fc, self.augment_interaction_highway])
        ### admet 
        admet_embedding_lst = []
        for idx in range(5):
            admet_embedding = self.feed_lst_of_module(input_feature = molecule_embed, 
                                                      lst_of_module = self.admet_model[idx])
            admet_embedding_lst.append(admet_embedding)
        ### pk 
        pk_input = torch.cat(admet_embedding_lst, 1)
        pk_embedding = self.feed_lst_of_module(input_feature = pk_input, 
                                               lst_of_module = [self.pk_fc, self.pk_highway])
        ### trial 
        trial_input = torch.cat([pk_embedding, augment_interaction_embedding], 1)
        trial_embedding = self.feed_lst_of_module(input_feature = trial_input, 
                                                  lst_of_module = [self.trial_fc, self.trial_highway])
        output = self.pred_nn(trial_embedding)
        output_cls = torch.sigmoid(self.pred_cls(trial_embedding))
        if if_gnn == False:
            return torch.cat([output, output_cls], 1)
        else:
            embedding_lst = [molecule_embed, icd_embed, protocol_embed, interaction_embedding, risk_of_disease_embedding, \
                             augment_interaction_embedding] + admet_embedding_lst + [pk_embedding, trial_embedding]
            return embedding_lst
            
    def init_pretrain(self, admet_model):
        self.molecule_encoder = admet_model.molecule_encoder

    def evaluation(self, predict_all, label_all, threshold = 0.5):
        import pickle, os
        from sklearn.metrics import roc_curve, precision_recall_curve
        # with open("predict_label.txt", 'w') as fout:
        #     for i,j in zip(predict_all, label_all):
        #         fout.write(str(i)[:4] + '\t' + str(j)[:4]+'\n')
        reg_predict_all = [i[0] for i in predict_all]
        reg_label_all = [i[0] for i in label_all]
        cls_predict_all = [i[1] for i in predict_all]
        cls_label_all = [i[1] for i in label_all]
        if self.num_classes != -1:
            auc_score = roc_auc_score(cls_label_all, cls_predict_all)
            figure_folder = "figure"
            #### ROC-curve 
            fpr, tpr, thresholds = roc_curve(cls_label_all, cls_predict_all, pos_label=1)
            # roc_curve =plt.figure()
            # plt.plot(fpr,tpr,'-',label=self.save_name + ' ROC Curve ')
            # plt.legend(fontsize = 15)
            #plt.savefig(os.path.join(figure_folder,name+"_roc_curve.png"))
            #### PR-curve
            precision, recall, thresholds = precision_recall_curve(cls_label_all, cls_predict_all)
            # plt.plot(recall,precision, label = self.save_name + ' PR Curve')
            # plt.legend(fontsize = 15)
            # plt.savefig(os.path.join(figure_folder,self.save_name + "_pr_curve.png"))
            cls_label_all = [int(i) for i in cls_label_all]
            float2binary = lambda x:0 if x<threshold else 1
            cls_predict_all = list(map(float2binary, cls_predict_all))
            f1score = f1_score(cls_label_all, cls_predict_all)
            prauc_score = average_precision_score(cls_label_all, cls_predict_all)
            # print(predict_all)
            precision = precision_score(cls_label_all, cls_predict_all, zero_division=0.0)
            recall = recall_score(cls_label_all, cls_predict_all)
            accuracy = accuracy_score(cls_label_all, cls_predict_all)
            predict_1_ratio = sum(cls_predict_all) / len(cls_predict_all)
            label_1_ratio = sum(cls_label_all) / len(cls_label_all)
            mae = mean_absolute_error(reg_label_all, reg_predict_all)
            mse = mean_squared_error(reg_label_all, reg_predict_all)
            correlation, p_value = pearsonr(reg_label_all, reg_predict_all)
            rmse = np.sqrt(mse)
            return auc_score, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio, mae, mse, rmse, correlation, p_value

    def generate_predict(self, dataloader):
        '''
        '''
        whole_loss = 0 
        label_all, predict_all, nctid_all = [], [], []
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst in dataloader:
            nctid_all.extend(nctid_lst)
            label_vec = label_vec.to(self.device)
            output = self.forward(smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst)
            if self.num_classes < 2:
                output = output.view(-1)
                label_vec = label_vec.float()
            loss = self.loss(output, label_vec)
            whole_loss += loss.item()
            predict_all.extend([i.detach().numpy() for i in output])
            label_all.extend([i.detach().numpy() for i in label_vec])
        return whole_loss, predict_all, label_all, nctid_all

    def select_threshold_for_binary(self, validloader):
        _, prediction, label_all, nctid_all = self.generate_predict(validloader)
        prediction = [pre[1] for pre in prediction]
        best_f1 = 0
        for threshold in prediction:
            float2binary = lambda x:0 if x<threshold else 1
            predict_all = list(map(float2binary, prediction))
            f1score = precision_score(label_all, predict_all, zero_division=0.0)        
            if f1score > best_f1:
                best_f1 = f1score 
                best_threshold = threshold
        return best_threshold 

    def test(self, dataloader, return_loss = True, validloader=None):
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        self.train()
        if return_loss:
            return whole_loss
        else:
            print_num = 5
            results_lst = self.evaluation(predict_all, label_all, threshold = best_threshold)
            if self.num_classes > 0:
                auc, f1score, prauc_score, precision, recall, accuracy, predict_1_ratio, label_1_ratio, mae, mse, rmse, correlation, p_value = results_lst
                print2file(self.save_name+'.txt', "==================Test==================")
                print2file(self.save_name+'.txt',  "PR-AUC   mean: "+str(prauc_score)[:6])
                print2file(self.save_name+'.txt',  "F1       mean: "+str(f1score)[:6])
                print2file(self.save_name+'.txt',  "ROC-AUC  mean: "+ str(auc)[:6])
                print2file(self.save_name+'.txt',  "Precision mean: "+str(precision)[:6])
                print2file(self.save_name+'.txt',  "Recall   mean: "+str(recall)[:6])
                print2file(self.save_name+'.txt',  "Accuracy mean: "+str(accuracy)[:6])
                print2file(self.save_name+'.txt',  "MAE      mean: "+str(mae)[:6])
                print2file(self.save_name+'.txt',  "MSE      mean: "+str(mse)[:6])
                print2file(self.save_name+'.txt',  "RMSE     mean: "+str(rmse)[:6])
                print2file(self.save_name+'.txt',  "Correlat mean: "+str(correlation)[:6])
            return results_lst

    def bootstrap_test(self, dataloader, sample_num = 20):
        # if validloader is not None:
        # 	best_threshold = self.select_threshold_for_binary(validloader)
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        reg_predict_all = [i[0] for i in predict_all]
        reg_label_all = [i[0] for i in label_all]
        cls_predict_all = [i[1] for i in predict_all]
        cls_label_all = [i[1] for i in label_all]
        from HINT.utils import plot_hist
        plt.clf()
        prefix_name = self.save_name.replace('logs/', 'logs/figure/') 
        os.makedirs(prefix_name, exist_ok=True)
        plot_hist(prefix_name, cls_predict_all, cls_label_all)		
        def bootstrap(length, sample_num):
            idx = [i for i in range(length)]
            from random import choices 
            bootstrap_idx = [choices(idx, k = length) for i in range(sample_num)]
            return bootstrap_idx 
        results_lst = []
        bootstrap_idx_lst = bootstrap(len(predict_all), sample_num = sample_num)
        for bootstrap_idx in bootstrap_idx_lst: 
            bootstrap_label = [label_all[idx] for idx in bootstrap_idx]		
            bootstrap_predict = [predict_all[idx] for idx in bootstrap_idx]
            results = self.evaluation(bootstrap_predict, bootstrap_label, threshold = best_threshold)
            results_lst.append(results)
        self.train() 
        if self.num_classes != -1:
            auc = [results[0] for results in results_lst]
            f1score = [results[1] for results in results_lst]
            prauc_score = [results[2] for results in results_lst]
            precision = [results[3] for results in results_lst]
            recall = [results[4] for results in results_lst]
            accuracy = [results[5] for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "PR-AUC   mean: "+str(np.mean(prauc_score))[:6], "std: "+str(np.std(prauc_score))[:6])
            print2file(self.save_name+'.txt',  "F1       mean: "+str(np.mean(f1score))[:6], "std: "+str(np.std(f1score))[:6])
            print2file(self.save_name+'.txt',  "ROC-AUC  mean: "+ str(np.mean(auc))[:6], "std: " + str(np.std(auc))[:6])
            print2file(self.save_name+'.txt',  "Precision mean: "+str(np.mean(precision))[:6], "std: "+str(np.std(precision))[:6])
            print2file(self.save_name+'.txt',  "Recall   mean: "+str(np.mean(recall))[:6], "std: "+str(np.std(recall))[:6])
            print2file(self.save_name+'.txt',  "Accuracy mean: "+str(np.mean(accuracy))[:6], "std: "+str(np.std(accuracy))[:6])
            mae = [results[8] for results in results_lst]
            mse = [results[9] for results in results_lst]
            rmse = [results[10] for results in results_lst]
            correlation = [results[11] for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "MAE      mean: "+str(np.mean(mae))[:6], "std: "+str(np.std(mae))[:6])
            print2file(self.save_name+'.txt',  "MSE      mean: "+str(np.mean(mse))[:6], "std: "+str(np.std(mse))[:6])
            print2file(self.save_name+'.txt',  "RMSE     mean: "+str(np.mean(rmse))[:6], "std: "+str(np.std(rmse))[:6])
            print2file(self.save_name+'.txt',  "Correlat mean: "+str(np.mean(correlation))[:6], "std: "+str(np.std(correlation))[:6])

        return nctid_all, predict_all 

    def testloader_to_lst(self, dataloader):
        '''
        '''
        nctid_lst, label_lst, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst = [], [], [], [], [], [], []
        for nctid, label, smiles, icdcode, criteria, tabular, text, mesh in dataloader:
            nctid_lst.extend(nctid)
            label_lst.extend([i for i in label])
            smiles_lst2.extend(smiles)
            icdcode_lst3.extend(icdcode)
            criteria_lst.extend(criteria)
            tabular_lst.extend(tabular)
            text_lst.extend(text)
            mesh_lst.extend(mesh)
        length = len(nctid_lst)
        assert length == len(smiles_lst2) and length == len(icdcode_lst3) and length == len(tabular_lst)
        return nctid_lst, label_lst, smiles_lst2, icdcode_lst3, criteria_lst, tabular_lst, text_lst, mesh_lst, length 



class CoverageLoss(nn.Module):
    def __init__(self):
        super(CoverageLoss, self).__init__()

    def forward(self, y_true, y_score):
        """
        Compute coverage percent of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            (mid, width)
        y_score : np.ndarray
            (mid, width)

        Returns
        -------
        float
            Coverage of predictions vs targets.
        """
        mid, width = y_score[:, 0], y_score[:, 1]
        lower_bound = mid - width / 2
        upper_bound = mid + width / 2
        true_lower, true_upper = y_true[:, 0], y_true[:, 1]
        cover_lower = torch.maximum(lower_bound, true_lower)
        cover_upper = torch.minimum(upper_bound, true_upper)
        cover_width = torch.maximum(torch.tensor(0.0), cover_upper - cover_lower)
        union_lower = torch.minimum(lower_bound, true_lower)
        union_upper = torch.maximum(upper_bound, true_upper)
        union_width = torch.maximum(torch.tensor(0.0), union_upper - union_lower)
        return torch.mean(cover_width / union_width)


class Dose(nn.Sequential):
    def __init__(self, molecule_encoder, mesh_encoder,
                    device, 
                    num_classes,
                    global_embed_size, 
                    highway_num_layer,
                    prefix_name, 
                    epoch = 20,
                    lr = 3e-4, 
                    weight_decay = 0, 
                    ):
        super(Dose, self).__init__()
        self.save_name = prefix_name + 'dose'
        self.molecule_encoder = molecule_encoder
        self.mesh_encoder = mesh_encoder
        self.num_classes = num_classes
        self.global_embed_size = global_embed_size 
        self.highway_num_layer = highway_num_layer 
        self.feature_dim = self.molecule_encoder.embedding_size + self.mesh_encoder.embedding_size
        self.epoch = epoch 
        self.lr = lr 
        self.weight_decay = weight_decay 

        self.f = F.relu

        if self.num_classes == 4:
            self.loss = nn.CrossEntropyLoss()
        elif self.num_classes == 2:
            self.loss = CoverageLoss()
        
        ##### NN 
        self.encoder2interaction_fc = nn.Linear(self.feature_dim, self.global_embed_size).to(device)
        self.encoder2interaction_highway = Highway(self.global_embed_size, self.highway_num_layer).to(device)
        self.pred_nn = nn.Linear(self.global_embed_size, num_classes if num_classes > 0 else 2)

        self.device = device 
        self = self.to(device)

    def feed_lst_of_module(self, input_feature, lst_of_module):
        x = input_feature
        for single_module in lst_of_module:
            x = self.f(single_module(x))
        return x

    def forward_get_two_encoders(self, smiles_lst2, mesh_lst):
        molecule_embed = self.molecule_encoder.forward_smiles_lst_lst(smiles_lst2)
        mesh_embed = self.mesh_encoder.forward(mesh_lst)
        return molecule_embed, mesh_embed

    def forward_encoder_2_interaction(self, molecule_embed, mesh_embed):
        encoder_embedding = torch.cat([molecule_embed, mesh_embed], 1)
        h = self.encoder2interaction_fc(encoder_embedding)
        h = self.f(h)
        h = self.encoder2interaction_highway(h)
        interaction_embedding = self.f(h)
        return interaction_embedding 

    def forward(self, smiles_lst2, mesh_lst):
        molecule_embed, mesh_embed = self.forward_get_two_encoders(smiles_lst2, mesh_lst)
        interaction_embedding = self.forward_encoder_2_interaction(molecule_embed, mesh_embed)
        output = self.pred_nn(interaction_embedding)
        return output ### 32, 1

    def concordance_index_for_ranking(self, y_true, y_pred):
        """
        Compute the concordance index (C-index) for ranking tasks.

        Parameters:
        - y_true: Array of actual grades/labels.
        - y_pred: Array of predicted scores (higher means higher predicted grade).

        Returns:
        - c_index: The concordance index.
        """
        n = len(y_true)
        assert len(y_pred) == n
        
        comparable_pairs = 0
        concordant_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:
                    comparable_pairs += 1
                    if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                    (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                        concordant_pairs += 1
        
        return concordant_pairs / comparable_pairs if comparable_pairs > 0 else 0


    def evaluation(self, predict_all, label_all, threshold = 0.5):
        import pickle, os
        from sklearn.metrics import roc_curve, precision_recall_curve
        with open("predict_label.txt", 'w') as fout:
            for i,j in zip(predict_all, label_all):
                fout.write(str(i)[:4] + '\t' + str(j)[:4]+'\n')
        if self.num_classes == 4:
            # predict_all_x = np.max(predict_all, axis=1, keepdims=True)
            # e_x = np.exp(predict_all - predict_all_x)
            # predict_all = e_x / e_x.sum(axis=1, keepdims=True)
            predict_all = np.array(predict_all)
            label_all = np.array(label_all)
            predict_all_x = np.max(predict_all, axis=1, keepdims=True)
            e_x = np.exp(predict_all - predict_all_x)
            predict_all = e_x / e_x.sum(axis=1, keepdims=True)
            auc_score = roc_auc_score(label_all, predict_all, average="macro", multi_class="ovr")
            label_all = [int(i) for i in label_all]
            float2cls = lambda x:np.argmax(x)
            predict_cls = list(map(float2cls, predict_all))
            f1score = f1_score(label_all, predict_cls, average="macro")
            precision = precision_score(label_all, predict_cls, average="macro", zero_division=0.0)
            recall = recall_score(label_all, predict_cls, average="macro")
            accuracy = accuracy_score(label_all, predict_cls)

            cindex = self.concordance_index_for_ranking(label_all, predict_cls)

            encoder = OneHotEncoder(sparse_output=False)
            one_hot_labels = encoder.fit_transform(np.array(label_all).reshape(-1, 1))
            # print("One-hot encoded labels:")
            # print(one_hot_labels)
            pr_aucs = []
            for i in range(predict_all.shape[1]):
                prec, recal, _ = precision_recall_curve(one_hot_labels[:, i],predict_all[:, i])
                pr_auc = auc(recal, prec)
                pr_aucs.append(pr_auc)
            prauc_score = np.mean(pr_aucs)
            
            predict_1_ratio = 0#sum(label_all) / len(predict_cls)  # no sense
            label_1_ratio = 0# sum(label_all) / len(label_cls) # no sense

            return auc_score, f1score, prauc_score, precision, recall, accuracy, cindex, predict_1_ratio, label_1_ratio 

        elif self.num_classes == 2:
            predict_all = np.array(predict_all)
            label_all  = np.array(label_all)
            mid, width = predict_all[:, 0], predict_all[:, 1]
            lower_bound = mid - width / 2
            upper_bound = mid + width / 2
            lower_bound = mid - width / 2
            upper_bound = mid + width / 2
            true_lower, true_upper = label_all[:, 0], label_all[:, 1]
            cover_lower = np.maximum(lower_bound, true_lower)
            cover_upper = np.minimum(upper_bound, true_upper)
            cover_width = np.maximum(0, cover_upper - cover_lower)
            union_lower = np.minimum(lower_bound, true_lower)
            union_upper = np.maximum(upper_bound, true_upper)
            union_width = np.maximum(0, union_upper - union_lower)
            return np.mean(cover_width / union_width)

        elif self.num_classes == 0:
            mae = mean_absolute_error(label_all, predict_all)
            mse = mean_squared_error(label_all, predict_all)
            correlation, p_value = pearsonr(label_all, predict_all)
            rmse = np.sqrt(mse)
            return mae, mse, rmse, correlation, p_value
        
    def testloader_to_lst(self, dataloader):
        '''
        '''
        nctid_lst, label_lst, smiles_lst2, mesh_lst = [], [], [], []
        for nctid, label, smiles, mesh in dataloader:
            nctid_lst.extend(nctid)
            label_lst.extend([i.item() for i in label])
            smiles_lst2.extend(smiles)
            mesh_lst.extend(mesh)
        length = len(nctid_lst)
        assert length == len(smiles_lst2) 
        return nctid_lst, label_lst, smiles_lst2, mesh_lst, length 

    def generate_predict(self, dataloader):
        '''
        '''
        whole_loss = 0 
        label_all, predict_all, nctid_all = [], [], []
        for nctid_lst, label_vec, smiles_lst2, mesh_lst in dataloader:
            nctid_all.extend(nctid_lst)
            label_vec = label_vec.to(self.device)
            output = self.forward(smiles_lst2, mesh_lst)
            if self.num_classes < 2:
                output = output.view(-1)  
                label_vec = label_vec.float()
            loss = self.loss(output, label_vec)
            whole_loss += loss.item()
            if self.num_classes == 2:
                predict_all.extend([i.detach().numpy() for i in output])
                label_all.extend([i.detach().numpy() for i in label_vec])
            else:
                output = torch.softmax(output, dim = 1)
                predict_all.extend([i.detach().numpy() for i in output])
                label_all.extend([i.detach().numpy() for i in label_vec])

        return whole_loss, predict_all, label_all, nctid_all

    def bootstrap_test(self, dataloader, sample_num = 20):
        # if validloader is not None:
        # 	best_threshold = self.select_threshold_for_binary(validloader)
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        from HINT.utils import plot_hist
        plt.clf()
        prefix_name = self.save_name.replace('logs/', 'logs/figure/') 
        os.makedirs(prefix_name, exist_ok=True)
        if self.num_classes == 4:
            plot_hist(prefix_name, predict_all, label_all)		
            
        def bootstrap(length, sample_num):
            idx = [i for i in range(length)]
            from random import choices 
            bootstrap_idx = [choices(idx, k = length) for i in range(sample_num)]
            return bootstrap_idx 
        results_lst = []
        bootstrap_idx_lst = bootstrap(len(predict_all), sample_num = sample_num)
        for bootstrap_idx in bootstrap_idx_lst: 
            bootstrap_label = [label_all[idx] for idx in bootstrap_idx]		
            bootstrap_predict = [predict_all[idx] for idx in bootstrap_idx]
            results = self.evaluation(bootstrap_predict, bootstrap_label, threshold = best_threshold)
            results_lst.append(results)
        self.train() 
        if self.num_classes == 4:
            auc = [results[0] for results in results_lst]
            f1score = [results[1] for results in results_lst]
            prauc_score = [results[2] for results in results_lst]
            precision = [results[3] for results in results_lst]
            recall = [results[4] for results in results_lst]
            accuracy = [results[5] for results in results_lst]
            cindex = [results[6] for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "PR-AUC   mean: "+str(np.mean(prauc_score))[:6], "std: "+str(np.std(prauc_score))[:6])
            print2file(self.save_name+'.txt',  "F1       mean: "+str(np.mean(f1score))[:6], "std: "+str(np.std(f1score))[:6])
            print2file(self.save_name+'.txt',  "ROC-AUC  mean: "+ str(np.mean(auc))[:6], "std: " + str(np.std(auc))[:6])
            print2file(self.save_name+'.txt',  "Precision mean: "+str(np.mean(precision))[:6], "std: "+str(np.std(precision))[:6])
            print2file(self.save_name+'.txt',  "Recall   mean: "+str(np.mean(recall))[:6], "std: "+str(np.std(recall))[:6])
            print2file(self.save_name+'.txt',  "Accuracy mean: "+str(np.mean(accuracy))[:6], "std: "+str(np.std(accuracy))[:6])
            print2file(self.save_name+'.txt',  "C-index  mean: "+str(np.mean(cindex))[:6], "std: "+str(np.std(cindex))[:6])
        elif self.num_classes == 2:
            coverage = [results for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "Coverage mean: "+str(np.mean(coverage))[:6], "std: "+str(np.std(coverage))[:6])
        else:
            mae = [results[0] for results in results_lst]
            mse = [results[1] for results in results_lst]
            rmse = [results[2] for results in results_lst]
            correlation = [results[3] for results in results_lst]
            print2file(self.save_name+'.txt', "==================Bootstrap Test==================")
            print2file(self.save_name+'.txt',  "MAE      mean: "+str(np.mean(mae))[:6], "std: "+str(np.std(mae))[:6])
            print2file(self.save_name+'.txt',  "MSE      mean: "+str(np.mean(mse))[:6], "std: "+str(np.std(mse))[:6])
            print2file(self.save_name+'.txt',  "RMSE     mean: "+str(np.mean(rmse))[:6], "std: "+str(np.std(rmse))[:6])
            print2file(self.save_name+'.txt',  "Correlat mean: "+str(np.mean(correlation))[:6], "std: "+str(np.std(correlation))[:6])


        # for nctid, label, predict in zip(nctid_all, label_all, predict_all):
        #     if (predict > 0.5 and label == 0) or (predict < 0.5 and label == 1):
        #         print(nctid, label, str(predict)[:5])

        # nctid2predict = {nctid:predict for nctid, predict in zip(nctid_all, predict_all)} 
        # pickle.dump(nctid2predict, open('results/nctid2predict.pkl', 'wb'))
        return nctid_all, predict_all 

    def ongoing_test(self, dataloader, sample_num = 20):
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader) 
        self.train() 
        return nctid_all, predict_all 
        
    def test(self, dataloader, return_loss = True, validloader=None):
        # if validloader is not None:
        # 	best_threshold = self.select_threshold_for_binary(validloader)
        self.eval()
        best_threshold = 0.5 
        whole_loss, predict_all, label_all, nctid_all = self.generate_predict(dataloader)
        # from HINT.utils import plot_hist
        # plt.clf()
        # prefix_name = "./figure/" + self.save_name 
        # plot_hist(prefix_name, predict_all, label_all)
        self.train()
        if return_loss:
            return whole_loss
        else:
            print_num = 5
            results_lst = self.evaluation(predict_all, label_all, threshold = best_threshold)
            if self.num_classes == 4:
                auc, f1score, prauc_score, precision, recall, accuracy, cindex, predict_1_ratio, label_1_ratio = results_lst
                print2file(self.save_name+'.txt', "==================Test==================")
                print2file(self.save_name+'.txt',  "PR-AUC   mean: "+str(prauc_score)[:6])
                print2file(self.save_name+'.txt',  "F1       mean: "+str(f1score)[:6])
                print2file(self.save_name+'.txt',  "ROC-AUC  mean: "+ str(auc)[:6])
                print2file(self.save_name+'.txt',  "Precision mean: "+str(precision)[:6])
                print2file(self.save_name+'.txt',  "Recall   mean: "+str(recall)[:6])
                print2file(self.save_name+'.txt',  "Accuracy mean: "+str(accuracy)[:6])
                print2file(self.save_name+'.txt',  "C-index  mean: "+str(cindex)[:6])
            elif self.num_classes == 2:
                coverage = results_lst
                print2file(self.save_name+'.txt', "==================Test==================")
                print2file(self.save_name+'.txt',  "Coverage mean: "+str(coverage)[:6])
            else:
                mae, mse, rmse, correlation, p_value = results_lst
                print2file(self.save_name+'.txt', "==================Test==================")
                print2file(self.save_name+'.txt',  "MAE      mean: "+str(mae)[:6])
                print2file(self.save_name+'.txt',  "MSE      mean: "+str(mse)[:6])
                print2file(self.save_name+'.txt',  "RMSE     mean: "+str(rmse)[:6])
                print2file(self.save_name+'.txt',  "Correlat mean: "+str(correlation)[:6])
            return results_lst

    def learn(self, train_loader, valid_loader, test_loader):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        train_loss_record = [] 
        valid_loss = self.test(valid_loader, return_loss=True)
        valid_loss_record = [valid_loss]
        best_valid_loss = valid_loss
        best_model = deepcopy(self)
        for ep in tqdm(range(self.epoch)):
            for nctid_lst, label_vec, smiles_lst2, mesh_lst in train_loader:
                label_vec = label_vec.to(self.device)
                output = self.forward(smiles_lst2, mesh_lst)
                if self.num_classes < 2:
                    output = output.view(-1) 
                    label_vec = label_vec.float() 
                loss = self.loss(output, label_vec)
                train_loss_record.append(loss.item())
                opt.zero_grad() 
                loss.backward() 
                opt.step()
            valid_loss = self.test(valid_loader, return_loss=True)
            valid_loss_record.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss 
                best_model = deepcopy(self)

        self.plot_learning_curve(train_loss_record, valid_loss_record)
        self = deepcopy(best_model)
        results_lst = self.test(test_loader, return_loss = False, validloader = valid_loader)\

    def plot_learning_curve(self, train_loss_record, valid_loss_record):
        plt.plot(train_loss_record)
        plt.savefig(self.save_name + '_train_loss.jpg')
        plt.clf() 
        plt.plot(valid_loss_record)
        plt.savefig(self.save_name + '_valid_loss.jpg')
        plt.clf() 

    def init_pretrain(self, admet_model):
        self.molecule_encoder = admet_model.molecule_encoder


