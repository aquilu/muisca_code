import streamlit as st
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Configuraciones iniciales
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Configuración de la página
st.set_page_config(page_title="Muisca - Asistente de IA", page_icon="📊")

# Título y descripción
st.title("Muisca - Asistente de IA para Análisis de Datos")
st.image("https://d1b4gd4m8561gs.cloudfront.net/sites/default/files/inline-images/brc-principal_1.png", width=400)

# Función para realizar EDA
def realizar_eda(df):
    st.write("### Análisis Exploratorio de Datos (EDA)")
    st.write("#### Descripción Estadística")
    st.write(df.describe())
    st.write("#### Visualizaciones")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        st.pyplot(plt)
        plt.figure()
        sns.boxplot(y=df[col])
        st.pyplot(plt)

# Carga de archivo
uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV aquí", type=["csv"])
df = None
file_processed = False  # Definir file_processed aquí

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        file_processed = True
        st.sidebar.success("Archivo CSV cargado correctamente.")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el archivo: {e}")

# Mostrar datos (opcional)
if file_processed and df is not None:
    st.markdown("## Vista Previa de los Datos")
    st.dataframe(df.head())
# Caja de texto para preguntas
user_question = st.sidebar.text_area("Escribe tu pregunta")

# Botón de envío
submit_button = st.sidebar.button("Enviar Pregunta")

# Asegúrate de que 'messages' esté en session_state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Añade la pregunta del usuario y la respuesta del asistente a 'messages'
def add_to_conversation(role, content):
    st.session_state['messages'].append({"role": role, "content": content})

# Manejo de la solicitud del usuario
if submit_button and user_question and df is not None:
    # Añadir pregunta y respuesta al historial
    add_to_conversation("user", user_question)
    add_to_conversation("system", "CSV data is already loaded for analysis.")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state['messages'],
        temperature=0.5,
        max_tokens=1000
    )
    response_content = response.choices[0].message.content
    add_to_conversation("assistant", response_content)

    # Mostrar la respuesta
    st.markdown("### Respuesta del Asistente")
    st.info(response_content)

    # Intentar ejecutar código relevante
    try:
        # Ejecutar solo partes del código que son seguras y conocidas
        if "df.describe()" in response_content:
            st.write("#### Estadísticas Descriptivas Básicas")
            st.write(df.describe())

        if "df.mean()" in response_content:
            st.write("#### Media de las Columnas")
            st.write(df.mean())

        if "df.median()" in response_content:
            st.write("#### Mediana de las Columnas")
            st.write(df.median())

        if "df.std()" in response_content:
            st.write("#### Desviación Estándar de las Columnas")
            st.write(df.std())

        if "df.corr()" in response_content:
            st.write("#### Matriz de Correlación")
            st.write(df.corr())

    except Exception as e:
        st.error(f"Error al ejecutar el código: {e}")

# Mostrar datos (opcional)
if file_processed and df is not None:
    st.markdown("## Vista Previa de los Datos")
    st.dataframe(df.head())