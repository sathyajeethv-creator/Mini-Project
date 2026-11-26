import streamlit as st, pandas as pd, joblib, json, os

st.set_page_config(page_title="Delivery Delay Predictor", page_icon="📦")

def find_file(ext):
    for root, _, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for f in files:
            name = f.replace(" ", "").lower()
            if name.endswith(ext) and ".git" not in root:
                if (ext == ".joblib" and "model" in name) or (ext == ".json" and "meta" in name):
                    return os.path.join(root, f)
    return None

@st.cache_resource
def load():
    mp, jp = find_file(".joblib"), find_file(".json")
    return joblib.load(mp), json.load(open(jp))

st.title("📦 Delivery Delay Predictor")
st.write("Enter order details to predict whether the delivery will be late.")

model, meta = load()

distance = st.number_input("Distance (km)", min_value=0.0,
                           max_value=float(meta["num_ranges"]["distance_km"][1]),
                           value=float(meta["num_defaults"]["distance_km"]))
promised = st.number_input("Promised delivery days", min_value=1, max_value=30,
                           value=int(meta["num_defaults"]["promised_delivery_days"]))
processing = st.number_input("Warehouse processing hours", min_value=0.0,
                             max_value=float(meta["num_ranges"]["warehouse_processing_hours"][1]),
                             value=float(meta["num_defaults"]["warehouse_processing_hours"]))

weather_opts = meta["cat_options"]["weather_condition"]
weather = st.selectbox("Weather condition", weather_opts,
                       index=weather_opts.index(meta["cat_defaults"]["weather_condition"]))
prio_opts = meta["cat_options"]["order_priority"]
priority = st.selectbox("Order priority", prio_opts,
                        index=prio_opts.index(meta["cat_defaults"]["order_priority"]))

if st.button("Predict"):
    row = {**meta["num_defaults"], **meta["cat_defaults"]}
    row.update({"distance_km": distance, "promised_delivery_days": promised,
                "warehouse_processing_hours": processing,
                "weather_condition": weather, "order_priority": priority})
    df_row = pd.DataFrame([row])[meta["columns"]]

    pred = model.predict(df_row)[0]
    prob = model.predict_proba(df_row)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely DELAYED (delay probability {prob:.0%})")
    else:
        st.success(f"✅ Likely ON TIME (delay probability {prob:.0%})")
    st.progress(float(prob))
