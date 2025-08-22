import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model.uv_model_mc import SinglePropKANModel
from dataset.dataset import MolDataset


def predict_by_smiles_list(SMILES_list, with_std=False, with_fea=False, 
                           model_pth = './model_pth/20250721.pth', 
                           pred_num_for_std=30, batch_size=256):
    dataset = MolDataset(SMILES_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_pth, map_location=torch.device(device)) 
    c_num = 1
    for c in model.keys():
        if 'core_layers' in c:
            c_num = max(int(c.split('.')[1])+1 , c_num)
    kan_in_dim = model['out_kan.act_fun.0.grid'].shape[0]
    
    model = SinglePropKANModel(dataset[0][0].x.shape[1], dropout=0.1, core_layer_num=c_num, kan_in_dim=kan_in_dim).to(device)

    checkpoint = torch.load(model_pth, map_location=device)
    model.load_state_dict(checkpoint)

    result_dict = {'model_pth': model_pth}

    if with_fea:
        preds, feas = predict(dataloader, model, with_fea=True)
        result_dict['preds'] = preds
        result_dict['feas'] = feas

    if with_std:
        pred_means, pred_stds = predict_with_dropout_std(dataloader, model, pred_num=pred_num_for_std)
        result_dict['pred_means'] = pred_means
        result_dict['pred_stds']  = pred_stds
        return result_dict
    else:
        preds = predict(dataloader, model)
        if 'pred' not in result_dict.keys():
            result_dict['preds'] = preds
        return result_dict
    

def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            
def predict(dataloader_to_pred, trained_model, device=None, use_dropout=False, with_fea=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model.eval()
    if use_dropout:
        enable_dropout(trained_model)

    with torch.no_grad():
        valid_preds = []
        feas_list = []
        for idx,(data_x,data_y) in enumerate(dataloader_to_pred):
            if with_fea:
                preds, feas = trained_model(data_x.to(device), return_fea=True)
                preds = preds.to('cpu')
                feas = feas.to('cpu')
                feas_list.append(feas)
            else:
                preds = trained_model(data_x.to(device)).to('cpu')
            valid_preds.append(preds)
    valid_preds = torch.concatenate(valid_preds).detach().numpy()
    
    
    if with_fea:
        feas_list = torch.concatenate(feas_list).detach().numpy()
        return valid_preds, feas_list
    else:
        return valid_preds


def predict_with_dropout_std(dataloader_to_pred, trained_model, device=None, pred_num=50):
    all_preds = []
    for i in range(pred_num):
        all_preds.append(predict(dataloader_to_pred, trained_model, device=device, use_dropout=True))
    all_preds = np.array(all_preds)
    pred_mean = all_preds.mean(axis=0)
    pred_std  = all_preds.std(axis=0)
    return pred_mean, pred_std

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

if __name__ == '__main__':
    SMILES_list = [ 'CN(N=C1)C=C1CC2(CC(F)(F)C2)NC(CC3=CC(C=C(Br)C=N4)=C4C=C3)C5=C(O)C=CC=N5', 
                    'Nc1[nH]c(=O)nc[c:1]1[N:1]1CSC[C@H]1C(=O)O']
    res = predict_by_smiles_list(SMILES_list)
    model_pth = res['model_pth']
    preds = res['preds']
    print('preds CFs:', preds)

    