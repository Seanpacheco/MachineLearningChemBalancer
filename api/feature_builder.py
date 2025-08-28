import numpy as np
import pandas as pd
from chemical_multipliers import get_chemical_multiplier

dual_effect_chems = {"Sodium Carbonate", "Muriatic Acid"}

param_map = {
    'ph': 'ph',
    'alkalinity': 'alkalinity', 
    'chlorine': 'chlorine',
    'calcium_hardness': 'calcium_hardness',
    'cyanuric_acid': 'cyanuric_acid'
}

def prepare_features(data_point, param, target, chemical, ohe):
    param_col = param_map.get(param.lower())
    if not param_col:
        raise ValueError(f"Unknown parameter: {param}")
    
    # Extract base readings
    current_val = data_point[param_col]
    pool_volume = data_point['pool_volume']
    
    ppm_change = target - current_val  
    volume_scaled_change = (pool_volume * abs(ppm_change)) / 120000  
    
    chemical_multiplier = get_chemical_multiplier(chemical, param)
    
    current_x_volume_x_target = current_val * pool_volume * target
    current_x_volume_x_delta = current_val * pool_volume * abs(ppm_change)
    volume_x_delta = pool_volume * abs(ppm_change)
    

    change_ratio = abs(ppm_change) / max(abs(current_val), 0.1)  # % change needed
    concentration_factor = current_val * pool_volume  # Total current chemical in pool
    

    log_pool_volume = np.log1p(pool_volume)
    is_dual_effect = 1 if chemical.strip().lower() in {c.lower() for c in dual_effect_chems} else 0
    

    ph_val = data_point['ph']
    alk_val = data_point['alkalinity']
    buffer_strength = ph_val * alk_val / 100
    

    numeric_features = [
        current_val,                   
        target,                        
        ppm_change,                   
        pool_volume,                   
        log_pool_volume,               
        volume_scaled_change,         
        chemical_multiplier,         
        current_x_volume_x_target,    
        current_x_volume_x_delta,     
        volume_x_delta,              
        change_ratio,                
        concentration_factor,      
        buffer_strength,              
        is_dual_effect,            
        data_point['ph'],               
        data_point['alkalinity'],
        data_point['chlorine'], 
        data_point['calcium_hardness'],
        data_point['cyanuric_acid']
    ]
    

    chem_param = f"{chemical}_{param}"
    try:
        cat_df = pd.DataFrame([[chemical, param, chem_param]], 
                             columns=['chemical', 'parameter', 'chem_param'])
        cat_array = ohe.transform(cat_df)[0]
        feature_vector = np.hstack([numeric_features, cat_array])
        return feature_vector
    except Exception as e:
        print(f"Error in prepare_features: {e}")
        raise
