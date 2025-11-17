from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd

from default_values import crop_defaults, KNOWN_CROPS
from predict_energy import predict_one, MODEL_PATH

app = Flask(__name__)

# ---------- Ensure Model Exists ----------
if not os.path.exists(MODEL_PATH):
    print("\n⚠ Model NOT found — run train_model.py first\n")


# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    user_inputs_used = None
    defaults_used = None

    if request.method == "POST":
        try:
            crop = request.form.get("crop")
            yield_val = float(request.form.get("yield_val"))

            payload = {
                "Crop_Type": crop,
                "Yield_t_ha": yield_val
            }

            # ---------- OPTIONAL USER INPUTS ----------
            if request.form.get("use_opt") == "on":

                rpr = request.form.get("rpr")
                if rpr:
                    payload["Residue_to_Product_Ratio"] = float(rpr)

                residue_t = request.form.get("residue_t")
                if residue_t:
                    payload["Residue_Quantity_t"] = float(residue_t)

                moisture = request.form.get("moisture")
                if moisture:
                    payload["Moisture_%"] = float(moisture)

                cv = request.form.get("cv")
                if cv:
                    payload["Calorific_Value_MJ_kg"] = float(cv)

            # ---------- RUN PREDICTION ----------
            result = predict_one(payload, MODEL_PATH)

            prediction = {
                "text": f"Crop: {crop}\n"
                        f"Predicted Bioenergy: {result['Predicted_Energy_MJ']:,.2f} MJ\n"
                        f"Equivalent Electrical Energy: {result['Predicted_Energy_kWh']:,.2f} kWh"
            }

            # ---------- SMART LOGIC ----------
            user_inputs_used = {}
            defaults_used = {}

            # User manually provided → store under user_inputs_used
            # Otherwise → read the default that model used

            if "Residue_to_Product_Ratio" in payload:
                user_inputs_used["RPR"] = payload["Residue_to_Product_Ratio"]
            else:
                defaults_used["RPR"] = result["defaults_used"]["RPR"]

            if "Moisture_%" in payload:
                user_inputs_used["Moisture_%"] = payload["Moisture_%"]
            else:
                defaults_used["Moisture_%"] = result["defaults_used"]["Moisture_%"]

            if "Calorific_Value_MJ_kg" in payload:
                user_inputs_used["Calorific_Value_MJ_kg"] = payload["Calorific_Value_MJ_kg"]
            else:
                defaults_used["Calorific_Value_MJ_kg"] = result["defaults_used"]["Calorific_Value_MJ_kg"]

            # If user entered ALL optional values → do not show defaults
            if len(defaults_used) == 0:
                defaults_used = None

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        crops=KNOWN_CROPS,
        crop_defaults=crop_defaults,
        prediction=prediction,
        defaults_used=defaults_used,
        user_inputs_used=user_inputs_used,
        error=error
    )


if __name__ == "__main__":
    print("\n🌍 Running Flask App → http://127.0.0.1:5000/\n")
    app.run(debug=True)
