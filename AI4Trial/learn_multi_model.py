## 1. import 
import torch, os, sys
torch.manual_seed(0) 
sys.path.append('.')
sys.path.append('..')
from dataset.dataset import *
from dataset.utils import *
from models.molecule_encode import MPNN, ADMET 
from models.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from models.protocol_encode import Protocol_Embedding
from models.tabular_encode import DANet
from models.text_encode import Text_Embedding
from models.mesh_encode import Mesh_Embedding
from models.model_multi import MultiModel, Dose, Multi_2_head
device = torch.device("cpu")


import argparse

parser = argparse.ArgumentParser(description='HINT')
parser.add_argument('--base_name', type=str, default='mortality_rate')
parser.add_argument('--phase', type=str, default="", help='Phase 1, Phase 2, Phase 3, Phase 4')
parser.add_argument('--exp', type=str, default='')
args = parser.parse_args()

print(args.exp)

base_name = 'logs/' + args.base_name + '/' + args.exp + (args.phase.replace(' ', '_') if args.phase else 'all')
if not os.path.exists('logs/' + args.base_name + '/' ):
	os.makedirs('logs/' + args.base_name + '/' )

def get_data(datasetname, phase):
    if datasetname == 'mortality_rate':
        return mortality_rate(phase)
    elif datasetname == 'serious_adverse_rate':
        return serious_adverse_rate(phase)
    elif datasetname == 'patient_dropout_rate':
        return patient_dropout_rate(phase)
    elif datasetname == 'duration':
        return duration(phase)
    elif datasetname == 'outcome':
        return outcome(phase)
    elif datasetname == 'failure_reason':
        return failure_reason(phase) 
    elif datasetname == 'serious_adverse_rate_yn':
        return serious_adverse_rate_yn(phase)
    elif datasetname == 'patient_dropout_rate_yn':
        return patient_dropout_rate_yn(phase)
    elif datasetname == 'mortality_rate_yn':
        return mortality_rate_yn(phase)
    elif datasetname == 'dose':
        return dose(phase)
    elif datasetname == 'dose_cls':
        return dose_cls(phase)


train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = get_data(args.base_name, args.phase if len(args.phase) > 3 else None)
print("num_classes: ", num_classes, 'tabular_input_dim: ', tabular_input_dim)

def get_mpnn_model(device):
	mpnn_model = MPNN(mpnn_hidden_size = 50, mpnn_depth=3, device = device)
	admet_model_path = "save_model/admet_model.ckpt"
	if not os.path.exists(admet_model_path):
		admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
		admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
		admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
		admet_model = ADMET(molecule_encoder = mpnn_model, 
							highway_num=2, 
							device = device, 
							epoch=3, 
							lr=5e-4, 
							weight_decay=0, 
							save_name = 'admet_')
		admet_model.train(admet_trainloader_lst, admet_testloader_lst)
		torch.save(admet_model, admet_model_path)
	else:
		admet_model = torch.load(admet_model_path)
		admet_model = admet_model.to(device)
		admet_model.set_device(device)
	return mpnn_model, admet_model

hint_model_path = base_name + "save_model" + ".ckpt"

if 'dose' not in base_name and 'two' not in args.exp:
	# For classifier and regression model
	icdcode2ancestor_dict = build_icdcode2ancestor_dict()
	gram_model = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
	protocol_model = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)
	tabular_model = DANet(input_dim = tabular_input_dim, output_dim = 50, layer_num=3, base_outdim = 50, device = device)
	text_model = Text_Embedding(output_dim = 50, highway_num=3, device = device)
	mesh_model = Mesh_Embedding(output_dim = 50, highway_num=3, device = device)
	mpnn_model, admet_model = get_mpnn_model(device)

	model = MultiModel(molecule_encoder = mpnn_model, 
			 disease_encoder = gram_model, 
			 protocol_encoder = protocol_model,
			 tabular_encoder = tabular_model,
			 text_encoder = text_model,
			 mesh_encoder = mesh_model,
			 device = device, 
			 num_classes = num_classes,
			 global_embed_size = 50, 
			 highway_num_layer = 2,
			 prefix_name = base_name, 
			#  gnn_hidden_size = 50,  
			 epoch = 20,
			 lr = 1e-3, 
			 weight_decay = 0, 
			)
	model.init_pretrain(admet_model)
	# transfer learning Setting
	if 'transfer' in args.exp:
		model = torch.load(f'logs/{args.base_name}/transferallsave_model.ckpt')
		print(f"load model from logs/{args.base_name}/transferallsave_model.ckpt")

	model.learn(train_loader, valid_loader, test_loader)
	model.bootstrap_test(test_loader)
	torch.save(model, hint_model_path)

elif 'dose' in base_name:
	# For Dose model
	mesh_model = Mesh_Embedding(output_dim = 50, highway_num=3, device = device)
	mpnn_model, admet_model = get_mpnn_model(device)
	model = Dose(molecule_encoder=mpnn_model, mesh_encoder=mesh_model, device=device, 
				num_classes=num_classes, global_embed_size=50, 
				highway_num_layer=2, prefix_name=base_name, 
				epoch=40, lr=1e-3, weight_decay=0)

	model.init_pretrain(admet_model)
	model.learn(train_loader, valid_loader, test_loader)
	model.bootstrap_test(test_loader)
	torch.save(model, hint_model_path)

elif 'two' in args.exp:
	# For Two-Head model (classifier + regression)
	icdcode2ancestor_dict = build_icdcode2ancestor_dict()
	gram_model = GRAM(embedding_dim = 50, icdcode2ancestor = icdcode2ancestor_dict, device = device)
	protocol_model = Protocol_Embedding(output_dim = 50, highway_num=3, device = device)
	tabular_model = DANet(input_dim = tabular_input_dim, output_dim = 50, layer_num=3, base_outdim = 50, device = device)
	text_model = Text_Embedding(output_dim = 50, highway_num=3, device = device)
	mesh_model = Mesh_Embedding(output_dim = 50, highway_num=3, device = device)
	mpnn_model, admet_model = get_mpnn_model(device)

	model = Multi_2_head(molecule_encoder = mpnn_model, 
			 disease_encoder = gram_model, 
			 protocol_encoder = protocol_model,
			 tabular_encoder = tabular_model,
			 text_encoder = text_model,
			 mesh_encoder = mesh_model,
			 device = device, 
			 num_classes = num_classes,
			 global_embed_size = 50, 
			 highway_num_layer = 2,
			 prefix_name = base_name, 
			#  gnn_hidden_size = 50,  
			 epoch = 1,
			 lr = 1e-3, 
			 weight_decay = 0, 
			)
	model.init_pretrain(admet_model)
	model.learn(train_loader, valid_loader, test_loader)
	model.bootstrap_test(test_loader)
	torch.save(model, hint_model_path)
else:
	# For testing
	model = torch.load(hint_model_path)
	model.bootstrap_test(test_loader)














