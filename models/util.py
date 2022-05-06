from .SITHConClassifier import SITHConClassifier
from .DeepLogPolarClassifier import DeepLogPolarClassifier
from .SingleLPClassifier import SingleLPClassifier
import torch
from copy import deepcopy

def get_model(config):
    model_type = config['model']['type'].lower()
    model_params = deepcopy(config['model'])
    collate = config.get('collate', 'batch')

    if model_type == "sithcon":
        ttype = torch.cuda.FloatTensor if config['device'] == 'cuda' else torch.FloatTensor
        
        for layer in model_params['layer_params']:
            layer['ttype'] = ttype
        model = SITHConClassifier(**model_params, collate=collate)
    elif model_type == "logpolar":
        model = DeepLogPolarClassifier(**model_params, collate=collate, device=config['device'])
    elif model_type == "single_lp":
        model = SingleLPClassifier(**model_params, collate=collate, device=config['device'])
    else:
        raise "Model not recognized."
    
    return model.to(config['device'])