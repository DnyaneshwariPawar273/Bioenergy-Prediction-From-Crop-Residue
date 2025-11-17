import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

from data_preprocessing import build_preprocessor, NUMERIC_COLS, CATEGORICAL_COLS
from default_values import crop_defaults
from feature_engineering import compute_residue_quantity

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')


def _prepare_input_df(payload: dict) -> pd.DataFrame:
    # Expected keys: Crop_Type, Yield_t_ha, Residue_to_Product_Ratio, Residue_Quantity_t, Moisture_%, Calorific_Value_MJ_kg
    # Accept missing optional keys; they will be filled by preprocessor via defaults and calculators
    data = {**{k: np.nan for k in (CATEGORICAL_COLS + NUMERIC_COLS)}, **payload}
    df = pd.DataFrame([data])
    return df


def predict_one(input_payload: dict, model_bundle_path: str = MODEL_PATH) -> dict:
    bundle = joblib.load(model_bundle_path)
    model = bundle['model']
    preprocessor = bundle['preprocessor']

    df = _prepare_input_df(input_payload)
    X_proc = preprocessor.transform(df)
    y_mj = float(model.predict(X_proc)[0])
    y_kwh = y_mj / 3.6

    # Determine what defaults were applied for transparency
    applied = {}
    crop = str(input_payload.get('Crop_Type')).strip()
    if crop in crop_defaults:
        d = crop_defaults[crop]
        applied['RPR'] = input_payload.get('Residue_to_Product_Ratio', d['RPR'])
        applied['Moisture_%'] = input_payload.get('Moisture_%', d['Moisture'])
        applied['Calorific_Value_MJ_kg'] = input_payload.get('Calorific_Value_MJ_kg', d['CV'])

    return {
        'Predicted_Energy_MJ': y_mj,
        'Predicted_Energy_kWh': y_kwh,
        'defaults_used': applied,
    }


def main():
    parser = argparse.ArgumentParser(description='Predict bioenergy potential from inputs')
    parser.add_argument('--payload', type=str, help='JSON string with input fields')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model.pkl')
    parser.add_argument('--crop', type=str, help='Crop type (e.g., Rice, Wheat, Maize, Sugarcane)')
    parser.add_argument('--yield_t_ha', type=float, help='Yield (t/ha)')
    parser.add_argument('--rpr', type=float, help='Residue-to-Product Ratio (optional)')
    parser.add_argument('--residue_t', type=float, help='Residue quantity (t) (optional)')
    parser.add_argument('--moisture', type=float, help='Moisture % (optional)')
    parser.add_argument('--cv', type=float, help='Calorific Value (MJ/kg) (optional)')
    args = parser.parse_args()

    if args.payload:
        payload = json.loads(args.payload)
    else:
        payload = {}
        if args.crop is not None:
            payload['Crop_Type'] = args.crop
        if args.yield_t_ha is not None:
            payload['Yield_t_ha'] = args.yield_t_ha
        if args.rpr is not None:
            payload['Residue_to_Product_Ratio'] = args.rpr
        if args.residue_t is not None:
            payload['Residue_Quantity_t'] = args.residue_t
        if args.moisture is not None:
            payload['Moisture_%'] = args.moisture
        if args.cv is not None:
            payload['Calorific_Value_MJ_kg'] = args.cv

    if 'Crop_Type' not in payload or 'Yield_t_ha' not in payload:
        raise SystemExit('Crop_Type and Yield_t_ha are required (pass via --crop and --yield_t_ha or --payload).')

    result = predict_one(payload, args.model)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
