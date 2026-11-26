import streamlit as st, pandas as pd, joblib, json, os

st.set_page_config(page_title="Delivery Delay Predictor", page_icon="📦")

def find_file(ext):
    # search the whole repo for a file ending in the extension
    for root, _, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for f in files:
            if f.replace(" ", "").lower().endswith(ext) and "venv" not in root:
                if (ext == ".joblib" and "model" in f.lower()) or (ext == ".json" and "meta" in f.lower()):
                    return os.path.join(root, f)
    return None

@st.cache_resource
def load():
    mp, jp = find_file(".joblib"), find_file(".json")
    if mp is None or jp is None:
        st.error("Files not found. The server sees:")
        st.write([os.path.join(r, f) for r, _, fs in os.walk(".") for f in fs if ".git" not in r])
        st.stop()
    return joblib.load(mp), json.load(open(jp))

model, meta = load()
