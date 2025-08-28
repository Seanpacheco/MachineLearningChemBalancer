from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from feature_builder import prepare_features
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('model/chem_balancer_ml_model.pkl')
ohe = joblib.load('model/chem_balancer_encoder.pkl')

try:
    use_log_transform = joblib.load('model/use_log_transform.pkl')
except:
    use_log_transform = False

specialized_models = {}
specialized_chemicals = ['Muriatic_Acid', 'Calcium_Chloride_77pct', 'Sodium_Bicarbonate']

for chem in specialized_chemicals:
    model_path = f'model/specialized_{chem}.pkl'
    if os.path.exists(model_path):
        specialized_models[chem.replace('_', ' ').replace('77pct', '(77%)')] = joblib.load(model_path)

print(f"Loaded {len(specialized_models)} specialized models")

chemical_units = {
    "Sodium Bicarbonate": "oz.",
    "Muriatic Acid": "fl oz.",
    "Chlorine Gas": "oz.",
    "Calcium Hypochlorite 67%": "oz.",
    "Calcium Hypochlorite 75%": "oz.",
    "Sodium Hypochlorite 12%": "fl oz.",
    "Lithium Hypochlorite 35%": "oz.",
    "Trichlor 90%": "oz.",
    "Dichlor 56%": "oz.",
    "Dichlor 62%": "oz.",
    "Calcium Chloride (77%)": "oz.",
    "Cyanuric Acid": "oz.",
    "Sodium Carbonate": "oz.",
    "Sodium Hydroxide": "fl oz.",
    "Sodium Thiosulfate": "oz."
}

raise_chemicals = {
    'alkalinity': 'Sodium Bicarbonate',
    'ph': ['Sodium Carbonate', 'Sodium Hydroxide'],
    'chlorine': None,
    'calcium_hardness': 'Calcium Chloride (77%)',
    'cyanuric_acid': 'Cyanuric Acid'
}

lower_chemicals = {
    'alkalinity': 'Muriatic Acid',
    'ph': 'Muriatic Acid',
    'chlorine': 'Sodium Thiosulfate',
    'calcium_hardness': None,
    'cyanuric_acid': None
}

chlorine_raise_options = [
    "Chlorine Gas", "Calcium Hypochlorite 67%", "Calcium Hypochlorite 75%",
    "Sodium Hypochlorite 12%", "Lithium Hypochlorite 35%", "Trichlor 90%",
    "Dichlor 56%", "Dichlor 62%"
]

def convert_to_appropriate_units(dosage, unit):
    """Convert dosages to more appropriate units when too large"""
    if unit == "fl oz.":
        if dosage >= 128:
            return round(dosage / 128, 2), "gallons"
        elif dosage >= 32:
            return round(dosage / 32, 2), "quarts"
        elif dosage >= 16:
            return round(dosage / 16, 2), "pints"
        return round(dosage, 2), unit
    elif unit == "oz.":
        if dosage >= 16:
            return round(dosage / 16, 2), "lbs"
        return round(dosage, 2), unit
    else:
        return round(dosage, 2), unit

def apply_pool_chemistry_priority(targets, current_values):
    """Apply pool chemistry priority: alkalinity first, then pH"""
    ph_needs_acid = 'ph' in targets and targets['ph'] < current_values['ph']
    alk_needs_acid = 'alkalinity' in targets and targets['alkalinity'] < current_values['alkalinity']
    
    if ph_needs_acid and alk_needs_acid:
        modified_targets = targets.copy()
        del modified_targets['ph']
        return modified_targets, [{'parameter': 'ph', 'reason': 'pH will be affected by alkalinity adjustment'}]
    
    return targets, []


def multi_stage_prediction(data, param, target, chemical):
    """Enhanced prediction using main model + specialized models for high dosages"""

    X = prepare_features(data, param, target, chemical, ohe)

    if use_log_transform:
        log_pred = model.predict([X])[0]
        base_pred = np.expm1(log_pred)
    else:
        base_pred = model.predict([X])[0]
    
    base_pred = max(base_pred, 0.01)
    

    if chemical in specialized_models and base_pred > 500:
        try:
            if use_log_transform:
                sp_log_pred = specialized_models[chemical].predict([X])[0]
                sp_pred = np.expm1(sp_log_pred)
            else:
                sp_pred = specialized_models[chemical].predict([X])[0]
            

            final_pred = 0.5 * base_pred + 0.5 * sp_pred
            return final_pred, "specialized_blend"
        except Exception as e:
            print(f"Error using specialized model for {chemical}: {e}")
    

    return float(base_pred), "main_model"

@app.route('/api/predict_dosage', methods=['POST'])
def predict_dosage():
    data = request.get_json()
    
    required = [
        'ph', 'alkalinity', 'chlorine', 'calcium_hardness', 'cyanuric_acid',
        'pool_volume', 'targets', 'available_chemicals'
    ]
    
    for f in required:
        if f not in data:
            return jsonify({'error': f'Missing required field: {f}'}), 400
    
    data['targets'], priority_skips = apply_pool_chemistry_priority(data['targets'], data)
    available_chems = set(data['available_chemicals'])
    
    recs = []
    skipped = priority_skips
    
    for param, target in data['targets'].items():
        current = data.get(param)
        if current is None:
            skipped.append({'parameter': param, 'reason': 'Missing current value in request'})
            continue
        
        if abs(target - current) < 1e-2:
            skipped.append({'parameter': param, 'reason': 'Already in target range'})
            continue
        
        needs_raise = target > current
        needs_lower = target < current
        chosen_chem = None
        
        if param == 'chlorine':
            if needs_raise:
                chosen_chem = next((chem for chem in chlorine_raise_options
                                  if chem in available_chems), None)
                if not chosen_chem:
                    skipped.append({'parameter': param, 'reason': 'No chlorine-raising chemical available'})
                    continue
            elif needs_lower:
                if 'Sodium Thiosulfate' not in available_chems:
                    skipped.append({'parameter': param, 'reason': 'Sodium Thiosulfate required but unavailable'})
                    continue
                chosen_chem = 'Sodium Thiosulfate'
        else:
            candidate_chems = raise_chemicals.get(param) if needs_raise else lower_chemicals.get(param)
            if not candidate_chems:
                skipped.append({
                    'parameter': param,
                    'reason': f'No chemical available to {"raise" if needs_raise else "lower"} {param}'
                })
                continue
            
            if isinstance(candidate_chems, list):
                chosen_chem = next((chem for chem in candidate_chems if chem in available_chems), None)
            else:
                chosen_chem = candidate_chems if candidate_chems in available_chems else None
            
            if not chosen_chem:
                skipped.append({
                    'parameter': param,
                    'reason': f'Required chemical(s) not available to {"raise" if needs_raise else "lower"} {param}'
                })
                continue
        

        pred_dosage, model_type = multi_stage_prediction(data, param, target, chosen_chem)
        

        original_unit = chemical_units.get(chosen_chem, 'units')
        converted_dosage, converted_unit = convert_to_appropriate_units(pred_dosage, original_unit)
        
        recs.append({
            'parameter': param,
            'chemical': chosen_chem,
            'dosage': converted_dosage,
            'unit': converted_unit,
            'original_dosage': round(pred_dosage, 2),
            'original_unit': original_unit,
            'model_type': model_type 
        })
    
    if not recs:
        return jsonify({
            'message': 'No parameters need adjustment.',
            'skipped': skipped
        }), 200
    
    return jsonify({
        'recommendations': recs,
        'skipped': skipped
    })

@app.route('/supported_adjustments', methods=['GET'])
def supported_adjustments():
    cats = ohe.categories_
    return jsonify({
        'parameters': list(cats[1]),
        'chemicals': list(cats[0]),
        'specialized_models': list(specialized_models.keys())
    })

if __name__ == '__main__':
    app.run(debug=True)
