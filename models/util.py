from .SITHConClassifier import SITHConClassifier
import torch
from copy import deepcopy

def get_model(config):
    model_type = config['model']['type'].lower()
    model_params = deepcopy(config['model'])

    if model_type == "sithcon":
        ttype = torch.cuda.FloatTensor if config['device'] == 'cuda' else torch.FloatTensor
        
        for layer in model_params['layer_params']:
            layer['ttype'] = ttype
        model = SITHConClassifier(**model_params, collate=config.get('collate', 'batch'))
    else:
        raise "Model not recognized."
    
    return model.to(config['device'])