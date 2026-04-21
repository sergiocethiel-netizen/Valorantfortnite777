import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import quant_engine as qe

# ==========================================================
# 1. ESTÉTICA BLOOMBERG TERMINAL (CSS Injection)
# ==========================================================
st.set_page_config(page_title="Quantitative Terminal", layout="wide", initial_sidebar_state="expanded")

bloomberg_css = """
<style>
/* Forzar tema ultra-oscuro puro (Bloomberg) */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
}
[data-testid="stSidebar"] {
    background-color: #0b0c10;
    border-right: 1px solid #1f2833;
}
/* Estilizar textos y métricas con neón y fuentes monoespaciadas */
h1, h2, h3, h4, h5, h6, span, p, .stMarkdown {
    color: #c5c6c7;
    font-family: 'Courier New', Courier, monospace !important;
}
[data-testid="stMetricValue"] {
    color: #66fcf1;
    font-weight: bold;
    font-size: 2rem;
}
[data-testid="stMetricDelta"] {
    color: #ff0055;
}
/* Personalización de inputs */
.stTextInput > div > div > input {
    background-color: #1a1a2e;
    color: #45f3ff;
}
</style>
"""
st.markdown(bloomberg_css, unsafe_allow_html=True)

# ==========================================================
# 2. ESTRUCTURA DE LA APP (UI)
# ==========================================================
st.title("⚡ BLOOMBERG QUANT TERMINAL (Markowitz & Tech)")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("🛠️ Configuración de Escenario")
uploaded_file = st.sidebar.file_uploader("Cargar Base de Datos (.xlsx, .csv)", type=['xlsx', 'csv'])
risk_free_rate = st.sidebar.number_input("Tasa Libre de Riesgo (Ej. 0.02 = 2%)", value=0.02, format="%.4f")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=100000)

var_min = st.sidebar.number_input("VaR Min (Límite Conservador en %)", value=-2.0, format="%.2f")
var_max = st.sidebar.number_input("VaR Max (Pérdida en %)", value=2.0, format="%.2f")

if uploaded_file is not None:
    st.success("Base de datos conectada al Engine.")
    
    # --------------------------------------------------
    # FASE 1: PROCESAMIENTO Y LIMPIEZA
    # --------------------------------------------------
    with st.spinner('Compilando álgebra matricial y aislando Benchmark...'):
        try:
            df_precios, df_mercado, nombre_benchmark = qe.load_and_parse_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error Estructural: {e}")
            st.stop()
            
    # --------------------------------------------------
    # FASE 2: MUESTRA ESTADÍSTICA
    # --------------------------------------------------
    st.subheader("1. Estadísticas Descriptivas (Detección Algorítmica)")
    retornos_diarios, betas, ret_anual, covarianzas = qe.calculate_returns_and_beta(df_precios, df_mercado, risk_free_rate)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Universo Seleccionado", f"{len(df_precios.columns)} Activos")
    col2.metric("Benchmark Activo", nombre_benchmark)
    col3.metric("Datos Analizados", f"{len(df_precios)} Periodos")
    
    # Matriz de Correlación (Heatmap Interactivo)
    st.markdown("##### 🌋 Matriz de Correlación de Pearson")
    corr_matrix = retornos_diarios.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    fig_corr.update_layout(paper_bgcolor="black", plot_bgcolor="black", font={'color':'white'})
    st.plotly_chart(fig_corr, use_container_width=True)

    # --------------------------------------------------
    # FASE 3: OPTIMIZACIÓN MARKOWITZ & MONTECARLO
    # --------------------------------------------------
    st.markdown("---")
    st.subheader(f"2. Frontera Eficiente (Simulación: 50,000 Universos Paralelos)")
    st.markdown(f"> *Constricciones aplicadas en Backend: $\Sigma w_i = 1$ (Ventas cortas bloqueadas) | VaR Forzado estrictamente entre {var_min}% y {var_max}%*")
    
    with st.spinner("Computando escenarios vectoriales de Montecarlo..."):
        # El motor lanza la simulación
        resultados_mc, optimos, hay_valido = qe.monte_carlo_frontier(ret_anual, covarianzas, retornos_diarios, 50000, risk_free_rate, var_min, var_max)
        
    if not hay_valido:
        st.warning("⚠️ No se encontró ningún portafolio combinando estos activos capaz de reducir el VaR estructural dentro de los estrictos límites matemáticos impuestos (entre +/- 2%). Se renderiza la frontera completa natural para propósitos exploratorios.")
        
    # Render Ploting Frontera
    fig_front = go.Figure()
    # Puntos MC
    fig_front.add_trace(go.Scatter(
        x=resultados_mc[1, :], 
        y=resultados_mc[0, :], 
        mode='markers',
        marker=dict(size=4, color=resultados_mc[2, :], colorscale='Viridis', showscale=True, colorbar=dict(title='Sharpe Ratio')),
        name='Portafolios Simulados'
    ))
    
    # Estrella Max Sharpe Portafolio
    ms_x = optimos['max_sharpe']['volatility']
    ms_y = optimos['max_sharpe']['return']
    fig_front.add_trace(go.Scatter(
        x=[ms_x], y=[ms_y], mode='markers', marker=dict(color='yellow', size=15, symbol='star'), name='Max Sharpe Portfolio'
    ))
    
    # Estrella Min Volatility Portafolio
    mv_x = optimos['min_vol']['volatility']
    mv_y = optimos['min_vol']['return']
    fig_front.add_trace(go.Scatter(
        x=[mv_x], y=[mv_y], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Min Variance Portfolio'
    ))
    
    # Capital Market Line (CML)
    # y = rf + (Sharpe) * x
    sharpe_optimo = optimos['max_sharpe']['sharpe']
    vols_cml = np.linspace(0, max(resultados_mc[1, :]) * 1.1, 100)
    ret_cml = risk_free_rate + sharpe_optimo * vols_cml
    fig_front.add_trace(go.Scatter(x=vols_cml, y=ret_cml, mode='lines', line=dict(color='cyan', dash='dash', width=2), name='Capital Market Line (CML)'))
    
    fig_front.update_layout(
        title="Frontera Eficiente Markowitz & CML Tangente",
        xaxis_title="Riesgo Sistémico (Volatilidad Anualizada)",
        yaxis_title="Rendimiento Esperado Anualizado",
        paper_bgcolor="black", plot_bgcolor="#0c0c0c", font={'color':'#45f3ff'}
    )
    st.plotly_chart(fig_front, use_container_width=True)
    
    # --------------------------------------------------
    # FASE 4: MÉTRICAS DEL PORTAFOLIO GANADOR Y VaR
    # --------------------------------------------------
    st.subheader("3. Desglose Estratégico (Portafolio Máximo Sharpe)")
    
    pesos_df = pd.DataFrame({'Activo': df_precios.columns, 'Asignación (%)': optimos['max_sharpe']['pesos'] * 100})
    pesos_df = pesos_df.sort_values(by='Asignación (%)', ascending=False)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pesos_df.style.format({'Asignación (%)': '{:.2f}%'}))
    with c2:
        # Calcular matriz VaR en el backend
        matriz_var = qe.calcular_matriz_var(optimos['max_sharpe']['pesos'], retornos_diarios, capital_inicial)
        
        st.markdown("#### 🚨 Análisis Cuantitativo de Riesgo Extremo (Value At Risk)")
        
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric(label="VaR Diario (95%)", value=f"${matriz_var['95%']['Diario']['Valor']:,.2f}", delta=f"{matriz_var['95%']['Diario']['Pct']:.2f}%")
        col_v2.metric(label="VaR Mensual (95%)", value=f"${matriz_var['95%']['Mensual']['Valor']:,.2f}", delta=f"{matriz_var['95%']['Mensual']['Pct']:.2f}%")
        col_v3.metric(label="VaR Anual (95%)", value=f"${matriz_var['95%']['Anual']['Valor']:,.2f}", delta=f"{matriz_var['95%']['Anual']['Pct']:.2f}%")
        
        col_v4, col_v5, col_v6 = st.columns(3)
        col_v4.metric(label="VaR Diario (99%)", value=f"${matriz_var['99%']['Diario']['Valor']:,.2f}", delta=f"{matriz_var['99%']['Diario']['Pct']:.2f}%")
        col_v5.metric(label="VaR Mensual (99%)", value=f"${matriz_var['99%']['Mensual']['Valor']:,.2f}", delta=f"{matriz_var['99%']['Mensual']['Pct']:.2f}%")
        col_v6.metric(label="VaR Anual (99%)", value=f"${matriz_var['99%']['Anual']['Valor']:,.2f}", delta=f"{matriz_var['99%']['Anual']['Pct']:.2f}%")

    # --------------------------------------------------
    # FASE 5: ANÁLISIS TÉCNICO BLOOMBERG
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("4. Technical Desk (Candlestick & Indicadores Avanzados)")
    
    activo_seleccionado = st.selectbox("Seleccionar Activo para Backtesting Técnico:", df_precios.columns)
    
    # Backend procesa los datos técnicos y clusters
    df_tech, ts_sop, ts_res = qe.technical_indicators(df_precios[activo_seleccionado])
    
    # Crear Subplots con MACD y RSI
    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                               subplot_titles=('Movimiento de Precios (SMAs y Canales)', 'MACD (Línea de Señal & Divergencia)', 'RSI (Indicador de Fuerza Relativa)'))
    
    # Para formar velas asumiremos que las series semanales o simples representan "Cierre"
    # Para engañar la visualización sin High/Low usaremos proxy (Scatter) o Velas simuladas si el cierre es igual
    # O simplemente un Mountain Chart estilo Area
    fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Close'], mode='lines', name='Price Line', line=dict(color='#00ff00', width=2)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['SMA_50'], line=dict(color='yellow', width=1.5), name='SMA 50'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['SMA_100'], line=dict(color='magenta', width=1.5), name='SMA 100'), row=1, col=1)
    
    # Trazar Soportes Automáticos
    for sl in ts_sop:
        fig_tech.add_hline(y=sl, line_dash="dot", line_color="lime", annotation_text="Support", row=1, col=1)
    for rl in ts_res:
        fig_tech.add_hline(y=rl, line_dash="dot", line_color="salmon", annotation_text="Resistance", row=1, col=1)
        
    # MACD Plot
    fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='cyan', width=1.5), name='MACD'), row=2, col=1)
    fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], line=dict(color='red', width=1.5, dash='dash'), name='MACD Signal'), row=2, col=1)
    fig_tech.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_Histogram'], name='MACD Hist', marker_color='gray'), row=2, col=1)
    
    # RSI Plot
    fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='orange', width=2), name='RSI'), row=3, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig_tech.update_layout(height=800, paper_bgcolor="#000000", plot_bgcolor="#050505", font={'color':'white'}, showlegend=True, hovermode="x unified")
    fig_tech.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1e1e1e')
    fig_tech.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1e1e1e')
    
    st.plotly_chart(fig_tech, use_container_width=True)
    
else:
    st.info("← Por favor carga tu base de datos en el panel lateral para encender el Terminal.")
