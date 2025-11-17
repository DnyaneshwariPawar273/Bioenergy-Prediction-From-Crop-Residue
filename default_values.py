crop_defaults = {
    'Rice': {'RPR': 1.5, 'Moisture': 12.0, 'CV': 14.5},
    'Wheat': {'RPR': 1.3, 'Moisture': 10.0, 'CV': 15.2},
    'Sugarcane': {'RPR': 0.3, 'Moisture': 50.0, 'CV': 17.0},
    'Maize': {'RPR': 1.8, 'Moisture': 15.0, 'CV': 16.5},
    # Extendable: add more crops as needed
}

KNOWN_CROPS = list(crop_defaults.keys())


def get_defaults(crop_type: str):
    """Return default RPR, Moisture, CV for a crop. Raises KeyError if unknown."""
    if crop_type not in crop_defaults:
        raise KeyError(f"Unknown crop type '{crop_type}'. Known: {', '.join(KNOWN_CROPS)}")
    return crop_defaults[crop_type].copy()
