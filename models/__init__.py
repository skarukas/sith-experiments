from .SITHConClassifier import SITHConClassifier

def get_model(model_specs):
    model_type = model_specs['type'].lower()
    
    if model_type == "sithcon":
        model = SITHConClassifier(model_specs)
    else:
        raise "Model not recognized."
    
    return model