import streamlit as st
import pickle
import pandas as pd
import base64

# Configurar página
st.set_page_config(page_title="Clasificador CarrRisk", layout="centered")

# Cargar y mostrar logo (opcional)
def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Si tienes un logo, pon su nombre aquí, si no, comenta esta línea
# image_base64 = load_image_base64("logo_app.png")
# st.markdown(f'<div style="text-align:right"><img src="data:image/png;base64,{image_base64}" width="120"></div>', unsafe_allow_html=True)

# Cargar modelos entrenados
with open("modelo-clas-tree-knn-nn.pkl", "rb") as f:
    model_tree, model_knn, model_nn, le_risk, variables, scaler = pickle.load(f)

st.title("Clasificador CarrRisk")
st.sidebar.header("Datos del usuario")

def user_input():
    edad = st.sidebar.slider("Edad", 18, 70, 30)
    cartype = st.sidebar.selectbox("Tipo de vehículo", ["combi", "sport", "family", "minivan"])
    modelo = st.sidebar.selectbox("Modelo de clasificación", ["DT", "Knn", "NN"])
    
    data = pd.DataFrame({
        "age": [edad],
        "cartype": [cartype]
    })
    
    # One-hot encoding y reindexar
    data = pd.get_dummies(data, columns=["cartype"], drop_first=False)
    data = data.reindex(columns=variables, fill_value=0)
    return data, modelo

df, modelo = user_input()

st.subheader("Datos ingresados")
st.write(df)

st.subheader(f"Modelo seleccionado: {modelo}")

if st.button("Predecir"):
    df[["age"]] = scaler.transform(df[["age"]])
    if modelo == "DT":
        pred = model_tree.predict(df)
    elif modelo == "Knn":
        pred = model_knn.predict(df)
    else:
        pred = model_nn.predict(df)

    resultado = le_risk.inverse_transform(pred)
    riesgo = "Alto Riesgo" if resultado[0] == "high" else "Bajo Riesgo"
    st.success(f"Predicción: {riesgo}")
