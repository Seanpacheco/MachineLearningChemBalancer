

CHEMICAL_MULTIPLIERS = {
    # Alkalinity adjustments
    'Muriatic Acid': {'alkalinity': 0.96, 'ph': 0.96},
    'Sodium Bicarbonate': {'alkalinity': 1.68},
    'Sodium Carbonate': {'alkalinity': 1.06, 'ph': 1.06},
    
    # Chlorine adjustments
    'Chlorine Gas': {'chlorine': 1.3/16},  # 1.3 oz per 10k gal, convert to lbs
    'Calcium Hypochlorite 65%': {'chlorine': 1.55},
    'Calcium Hypochlorite 67%': {'chlorine': 2.0/16},
    'Calcium Hypochlorite 75%': {'chlorine': 2.0/16}, 
    'Sodium Hypochlorite 12%': {'chlorine': 1.0},  # 1 gallon per lb equivalent
    'Lithium Hypochlorite 35%': {'chlorine': 4.0/16},
    'Trichlor 90%': {'chlorine': 1.09},
    'Dichlor 56%': {'chlorine': 1.81},
    'Dichlor 62%': {'chlorine': 1.81},
    
    # Other chemicals
    'Calcium Chloride (77%)': {'calcium_hardness': 1.44},
    'Cyanuric Acid': {'cyanuric_acid': 1.0},
    'Sodium Thiosulfate': {'chlorine': 1.0/16},  # 1 oz per 10k gal
    'Sodium Hydroxide': {'ph': 1.0, 'alkalinity': 1.0},
}

def get_chemical_multiplier(chemical, parameter):
    """Get the chemical multiplier from industry standards"""
    return CHEMICAL_MULTIPLIERS.get(chemical, {}).get(parameter, 1.0)
