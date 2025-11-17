import numpy as np
import pandas as pd

REQUIRED_COLS = [
    'Yield_t_ha',
    'Residue_to_Product_Ratio',
    'Residue_Quantity_t',
    'Moisture_%',
    'Calorific_Value_MJ_kg',
]


def compute_residue_quantity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Residue_Quantity_t' not in df.columns:
        df['Residue_Quantity_t'] = np.nan
    mask = df['Residue_Quantity_t'].isna() & df['Yield_t_ha'].notna() & df['Residue_to_Product_Ratio'].notna()
    df.loc[mask, 'Residue_Quantity_t'] = df.loc[mask, 'Yield_t_ha'] * df.loc[mask, 'Residue_to_Product_Ratio']
    return df


def compute_dry_residue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dry = df.get('Residue_Quantity_t') * (1.0 - df.get('Moisture_%') / 100.0)
    df['Dry_Residue_t'] = dry
    return df


def compute_energy_mj(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Dry_Residue_t' not in df.columns:
        df = compute_dry_residue(df)
    # Energy (MJ) = Dry Residue (t) * 1000 kg/t * Calorific Value (MJ/kg)
    df['Energy_MJ_computed'] = df['Dry_Residue_t'] * 1000.0 * df['Calorific_Value_MJ_kg']
    return df
