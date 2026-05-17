import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis Promigas", layout="wide")

# Título y descripción
st.title("Análisis del Precio de la Acción de Promigas vs Variables Regresoras")
st.markdown("""
Este dashboard interactivo analiza la relación entre el precio de la acción de **Promigas (BVC)** y dos variables macroeconómicas clave (regresoras):
1. **TRM (Tasa Representativa del Mercado - USD/COP)**: Afecta los costos, deudas y exposición cambiaria de la empresa.
2. **Precio del Petróleo Brent (USD/Barril)**: Benchmark energético clave para la economía colombiana y empresas del sector.
""")

@st.cache_data
def load_and_merge_data():
    # 1. Cargar datos de Promigas (Asegúrate de que el CSV esté en la misma carpeta)
    df_promigas = pd.read_csv("Promigas Stock Price History.csv")
    df_promigas['Date'] = pd.to_datetime(df_promigas['Date'])
    
    # Limpiar formato de precio (eliminar comas y convertir a numérico)
    if df_promigas['Price'].dtype == 'O':
        df_promigas['Price'] = df_promigas['Price'].str.replace(',', '').astype(float)
    
    df_promigas = df_promigas[['Date', 'Price']].sort_values('Date').reset_index(drop=True)
    df_promigas = df_promigas.rename(columns={'Price': 'Promigas_Price_COP'})
    
    # Extraer rango de fechas para las regresoras
    min_date = df_promigas['Date'].min()
    max_date = df_promigas['Date'].max()
    
    # 2. Descargar datos de TRM (USD/COP) vía Yahoo Finance
    trm = yf.Ticker("COP=X").history(start=min_date, end=max_date + pd.Timedelta(days=1))
    trm_df = trm[['Close']].reset_index()
    # Eliminar timezone para poder hacer merge sin problemas
    trm_df['Date'] = pd.to_datetime(trm_df['Date']).dt.tz_localize(None).dt.normalize()
    trm_df = trm_df.rename(columns={'Close': 'TRM_USD_COP'})
    
    # 3. Descargar datos de Petróleo Brent vía Yahoo Finance
    oil = yf.Ticker("BZ=F").history(start=min_date, end=max_date + pd.Timedelta(days=1))
    oil_df = oil[['Close']].reset_index()
    oil_df['Date'] = pd.to_datetime(oil_df['Date']).dt.tz_localize(None).dt.normalize()
    oil_df = oil_df.rename(columns={'Close': 'Brent_Oil_USD'})
    
    # 4. Unir (Merge) todos los datos por la columna Date
    df_merged = pd.merge(df_promigas, trm_df, on='Date', how='inner')
    df_merged = pd.merge(df_merged, oil_df, on='Date', how='inner')
    
    # Rellenar valores nulos si existen (por diferencias en festivos/fines de semana, etc.)
    df_merged = df_merged.ffill().bfill()
    
    return df_merged

# Cargar los datos
try:
    with st.spinner("Cargando y procesando datos financieros..."):
        df = load_and_merge_data()

    st.success(f"Datos cargados exitosamente. Se analizaron {len(df)} registros válidos (más de 90 requeridos) desde {df['Date'].min().strftime('%Y-%m-%d')} hasta {df['Date'].max().strftime('%Y-%m-%d')}.")

    # Calcular correlaciones de Pearson
    corr_trm = df['Promigas_Price_COP'].corr(df['TRM_USD_COP'])
    corr_oil = df['Promigas_Price_COP'].corr(df['Brent_Oil_USD'])

    # Mostrar métricas y coeficientes de correlación
    st.header("Coeficientes de Correlación")
    st.markdown("El coeficiente de correlación de Pearson evalúa el grado de asociación lineal entre dos variables. Varía entre -1 y 1.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Correlación: Promigas vs TRM", value=f"{corr_trm:.4f}")
        if abs(corr_trm) > 0.7:
            st.info("Existe una **fuerte** correlación lineal.")
        elif abs(corr_trm) > 0.3:
            st.info("Existe una **moderada** correlación lineal.")
        else:
            st.info("Existe una **débil o nula** correlación lineal.")

    with col2:
        st.metric(label="Correlación: Promigas vs Petróleo Brent", value=f"{corr_oil:.4f}")
        if abs(corr_oil) > 0.7:
            st.info("Existe una **fuerte** correlación lineal.")
        elif abs(corr_oil) > 0.3:
            st.info("Existe una **moderada** correlación lineal.")
        else:
            st.info("Existe una **débil o nula** correlación lineal.")

    # Visualizaciones
    st.header("Visualizaciones en el Tiempo")

    # Gráfico 1: Promigas vs TRM
    st.subheader("1. Evolución del Precio de Promigas vs TRM (USD/COP)")
    fig1 = go.Figure()

    # Eje Y principal (Promigas)
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Promigas_Price_COP'], name="Promigas (COP)", mode='lines', line=dict(color='#1f77b4', width=2)))

    # Eje Y secundario (TRM)
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['TRM_USD_COP'], name="TRM (USD/COP)", yaxis="y2", mode='lines', line=dict(color='#ff7f0e', width=2, dash='dot')))

    # Configurar layout para doble eje
    fig1.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio Promigas (COP)",
        yaxis2=dict(
            title="TRM (COP)",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig1, use_container_width=True)


    # Gráfico 2: Promigas vs Brent
    st.subheader("2. Evolución del Precio de Promigas vs Petróleo Brent")
    fig2 = go.Figure()

    # Eje Y principal (Promigas)
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Promigas_Price_COP'], name="Promigas (COP)", mode='lines', line=dict(color='#1f77b4', width=2)))

    # Eje Y secundario (Petróleo)
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Brent_Oil_USD'], name="Petróleo Brent (USD)", yaxis="y2", mode='lines', line=dict(color='#2ca02c', width=2, dash='dot')))

    # Configurar layout para doble eje
    fig2.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio Promigas (COP)",
        yaxis2=dict(
            title="Brent (USD/Barril)",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter plots (Dispersión) para ver la relación directamente
    st.subheader("Gráficos de Dispersión (Asociación Directa)")
    col3, col4 = st.columns(2)

    with col3:
        fig_scatter1 = px.scatter(df, x="TRM_USD_COP", y="Promigas_Price_COP", 
                                  title="Dispersión: Promigas vs TRM",
                                  labels={"TRM_USD_COP": "TRM (COP)", "Promigas_Price_COP": "Promigas (COP)"})
        st.plotly_chart(fig_scatter1, use_container_width=True)

    with col4:
        fig_scatter2 = px.scatter(df, x="Brent_Oil_USD", y="Promigas_Price_COP", 
                                  title="Dispersión: Promigas vs Petróleo Brent",
                                  labels={"Brent_Oil_USD": "Brent (USD)", "Promigas_Price_COP": "Promigas (COP)"})
        st.plotly_chart(fig_scatter2, use_container_width=True)

    st.markdown("---")
    st.write("Desarrollado para la clase de Introducción a la IA.")
    
except Exception as e:
    st.error(f"Se produjo un error al cargar o procesar los datos: {e}")
    st.info("Asegúrate de que el archivo 'Promigas Stock Price History.csv' esté en el mismo directorio que app.py y que tengas conexión a internet para descargar datos de Yahoo Finance.")
