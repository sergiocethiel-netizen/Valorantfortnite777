import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# ==========================================
# 0. CONFIGURACIÓN INICIAL (DARK MODE ABSOLUTO)
# ==========================================
st.set_page_config(page_title="Neon Quant Terminal", layout="wide", initial_sidebar_state="expanded")

neon_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* Reset de fondos Streamlit a Negros Puros */
.stApp { background-color: #080808 !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] { background-color: #0c0c0c !important; border-right: 1px solid #222 !important; }

/* Redondear selectores nativos y sliders */
.stSlider > div[data-baseweb="slider"] { padding-top: 10px; }
div[data-baseweb="select"] > div { background-color: #121212 !important; border-radius: 10px !important; border-color: #333 !important; color:#fff !important;}
label { color: #8b5cf6 !important; font-weight: 600 !important; font-size: 14px !important; }

/* Custom Text Blocks */
.neon-title { font-size: 24px; font-weight: 700; background: -webkit-linear-gradient(45deg, #00f0ff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px;}
.sub-header { color: #9ca3af; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;}

/* Tabs Streamlit Hacks */
.stTabs [data-baseweb="tab-list"] { gap: 20px; background-color: transparent; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 600; background-color: #111; border-radius: 8px 8px 0px 0px; padding: 10px 20px; border-top: 1px solid #333; border-left: 1px solid #333; border-right: 1px solid #333;}
.stTabs [aria-selected="true"] { background: linear-gradient(180deg, #1f1f1f, #121212) !important; color: #00f0ff; border-top: 2px solid #00f0ff !important; box-shadow: 0px -5px 15px rgba(0,240,255,0.15);}

/* Cajas Custom (Reemplazo local de columns) */
.glass-box {
    background-color: #121212; border-radius: 15px; border: 1px solid #1f1f1f;
    box-shadow: 0 5px 20px rgba(0,0,0,0.8), inset 0 0 10px rgba(138, 43, 226, 0.05); padding: 15px;
    margin-bottom: 15px; transition: transform 0.2s, box-shadow 0.2s;
}
.glass-box:hover { box-shadow: 0 5px 25px rgba(0, 240, 255, 0.15), inset 0 0 10px rgba(0, 240, 255, 0.1); border-color: rgba(0, 240, 255, 0.3); }

/* Hack para tablas HTML limpios */
.custom-table { width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; background-color: #121212; border-radius:15px; overflow:hidden;}
.custom-table th { padding: 12px; border-bottom: 1px solid #333; color: #a855f7; font-weight: 600; background-color: #0c0c0c;}
.custom-table td { padding: 12px; border-bottom: 1px solid #222; color: #e2e8f0; }

#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
"""
st.markdown(neon_css, unsafe_allow_html=True)

COLOR_BG = "rgba(0,0,0,0)"
C_CYAN = "#00f0ff"
C_PURP = "#8b5cf6"
C_PINK = "#ff00ff"
C_TEXT = "#e2e8f0"

def style_plotly(fig):
    fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font=dict(color=C_TEXT, family="Inter"), margin=dict(r=10, l=10, t=30, b=10))
    return fig

# ==========================================
# 1. MOTOR QUANTITATIVO (BACKEND)
# ==========================================
@st.cache_data(ttl=86400)
def load_market_data(bench_ticker="^IXIC"):
    # 12 Activos: Los originales 10 + GLD + SGOV (ahora sin IBTA, entra TTD)
    assets = ["VSAT", "PTEN", "HL", "CDE", "ICHR", "WMT", "BAX", "AVGO", "JPM", "GLD", "SGOV"]
    tkrs = assets + [bench_ticker]
    end = datetime.today()
    start = end - timedelta(days=2*365)
    data = yf.download(tkrs, start=start, end=end, progress=False)['Close'].ffill().dropna()
    returns = np.log(data / data.shift(1)).dropna()
    return returns, data, assets

def geometric_brownian_motion(S0, mu, sigma, T=1, N=252, paths=50):
    dt = T/N
    price_paths = np.zeros((N, paths))
    price_paths[0] = S0
    for t in range(1, N):
        Z = np.random.standard_normal(paths)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return price_paths

def hurst_exponent(ts):
    lags = range(2, min(100, len(ts)//2))
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0, lags, tau

def lyapunov_exponent(returns):
    # Proxy ultrarrápido Divergence Rate (1D array)
    div_rate = np.log(np.abs(np.diff(returns)) + 1e-8)
    return np.mean(div_rate), div_rate

# ==========================================
# 2. CONTROLES Y SIDEBAR (UI REACTIVO)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#00f0ff;'>Control de Mando</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9ca3af; margin-bottom:20px; font-size:13px;'>Live Parameter Tuning</div>", unsafe_allow_html=True)
    
    bench_opt = st.selectbox("Benchmark de Referencia", {"Nasdaq 100": "^IXIC", "S&P 500": "SPY"}.keys(), index=0)
    bench_sym = "^IXIC" if bench_opt == "Nasdaq 100" else "SPY"
    
    # Sliders Dinámicos Generados por Array
    raw_returns, price_data, asset_list = load_market_data(bench_sym)
    
    st.markdown("<div class='sub-header' style='margin-top:20px;'>Asset Weights (%)</div>", unsafe_allow_html=True)
    raw_weights = []
    initial_weight = round(100.0 / len(asset_list), 2)
    for asset in asset_list:
        # Reemplazo de st.slider por st.number_input para permitir escritura de números estricta
        val = st.number_input(f"% {asset}", 0.0, 100.0, initial_weight, step=0.01, format="%.2f")
        raw_weights.append(val)
        
    w_sum = sum(raw_weights)
    if w_sum == 0:
        raw_weights = [initial_weight]*len(asset_list)
        w_sum = sum(raw_weights)
        
    w_arr = np.array([w/w_sum for w in raw_weights])
    
    # Tolerancia de decimales (si da 99.96 en vez de 100 por no ser divisor cerrados, no mostramos el error)
    if not math.isclose(w_sum, 100.0, abs_tol=0.1):
        st.markdown(f"<div style='color:#ef4444; font-size:12px; margin-bottom:10px;'>⚠️ Suma {w_sum:.2f}%. Auto-normalizando a 100%.</div>", unsafe_allow_html=True)
        
    st.markdown("<div class='sub-header' style='margin-top:20px;'>Riesgo Paramétrico</div>", unsafe_allow_html=True)
    r_conf = st.selectbox("Nivel de Confianza", ["90%", "95%", "99%"], index=1)

# Procesamiento Cómputos Base
port_ret = raw_returns[asset_list].dot(w_arr)
bench_ret = raw_returns[bench_sym]

r_anual = port_ret.mean() * 252
v_anual = port_ret.std() * np.sqrt(252)
sharpe = (r_anual - 0.045) / v_anual if v_anual > 0 else 0

tracking_error = (port_ret - bench_ret).std() * np.sqrt(252)
info_ratio = (r_anual - bench_ret.mean()*252) / tracking_error if tracking_error > 0 else 0
cov_m = np.cov(port_ret, bench_ret)
beta = cov_m[0,1]/cov_m[1,1]
alpha = r_anual - (0.045 + beta * (bench_ret.mean()*252 - 0.045))

conf_level = int(r_conf.strip("%"))
var_q = np.percentile(port_ret, 100 - conf_level)
cvar_val = port_ret[port_ret <= var_q].mean() * 100
port_cum = (1+port_ret).cumprod() * 100
bench_cum = (1+bench_ret).cumprod() * 100

rf_d = 0.045 / 252
downside_diff = port_ret[port_ret < rf_d] - rf_d
down_std = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252) if len(downside_diff) > 0 else 0
sortino = (r_anual - 0.045) / down_std if down_std > 0 else 0

treynor = (r_anual - 0.045) / beta if beta != 0 else 0

bench_vol = bench_ret.std() * np.sqrt(252)
m2 = 0.045 + sharpe * bench_vol

roll_max = port_cum.cummax()
drawdown = (port_cum - roll_max) / roll_max
max_dd = drawdown.min()
calmar = r_anual / abs(max_dd) if max_dd != 0 else 0

st.markdown("<div class='neon-title'>Institutional Quant Terminal <span style='font-weight:300; font-size:18px; color:#6b7280;'>| Advanced Module</span></div>", unsafe_allow_html=True)

# ==========================================
# TABS INTERACTIVOS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["Performance Overview", "Markowitz & Risk", "Chaos Theory Core", "Deep Analytics & Horizons"])

# -------------------------------------------------------------
# TAB 1: OVERVIEW BASE (El Dashboard Originial Reestructurado)
# -------------------------------------------------------------
with tab1:
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    def simple_gauge(val, title, color_hex):
        f = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': title, 'font': {'size': 12, 'color': '#a855f7'}}, gauge={'axis': {'visible': False}, 'bar': {'color': color_hex}, 'bgcolor': 'rgba(255,255,255,0.02)', 'borderwidth': 0}, number={'font': {'size': 26, 'color': '#ffffff'}}))
        f.update_layout(height=200, paper_bgcolor=COLOR_BG, margin=dict(l=0,r=0,t=20,b=0))
        return f
        
    c1.plotly_chart(simple_gauge(sharpe, "Sharpe Ratio", C_CYAN), use_container_width=True, config={'displayModeBar':False})
    c2.plotly_chart(simple_gauge(v_anual*100, "Volatilidad (%)", C_PINK), use_container_width=True, config={'displayModeBar':False})
    c3.plotly_chart(simple_gauge(r_anual*100, "Retorno Base Cero (%)", C_PURP), use_container_width=True, config={'displayModeBar':False})
    c4.plotly_chart(simple_gauge(beta, "Beta Merc.", C_CYAN), use_container_width=True, config={'displayModeBar':False})
    
    col_a, col_b = st.columns([1, 2.5])
    with col_a:
        st.markdown("<div class='sub-header'>Distribución Asignada</div>", unsafe_allow_html=True)
        # Añade gama extra de colores para soportar los 12 activos en el pastel
        colors = [C_CYAN, C_PURP, C_PINK, "#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#6366f1", "#c026d3", "#f43f5e", "#14b8a6", "#22d3ee"]
        fig_donut = go.Figure(go.Pie(labels=asset_list, values=w_arr*100, hole=0.75, marker_colors=colors, textinfo='none'))
        fig_donut = style_plotly(fig_donut)
        fig_donut.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar':False})
    with col_b:
        st.markdown(f"<div class='sub-header'>AUM Trajectory VS {bench_opt}</div>", unsafe_allow_html=True)
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=port_cum.index, y=port_cum, mode='lines', name='Mi Cartera (11 Activos)', line=dict(shape='spline', smoothing=1, color=C_PURP, width=3), fill='tozeroy', fillcolor='rgba(139,92,246,0.15)'))
        fig_area.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, mode='lines', name=bench_opt, line=dict(shape='spline', smoothing=1, color='#475569', width=2)))
        fig_area = style_plotly(fig_area)
        fig_area.update_layout(height=280, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#222', tickformat=",.0f"), showlegend=True)
        st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar':False})
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    table_html = f"""
    <div class='sub-header'>Resumen Algorítmico Activo (11)</div>
    <table class="custom-table" style="margin-top:10px;">
        <tr><th>Métrica Cuantitativa</th><th align="right">Valor de Portafolio Ajustado al Riesgo</th></tr>
        <tr><td>Retorno Esperado Anualizado</td><td align="right" style="color:#ffffff;">{r_anual*100:.2f}%</td></tr>
        <tr><td>Tracking Error (vs {bench_opt})</td><td align="right" style="color:{C_CYAN};">{tracking_error*100:.2f}%</td></tr>
        <tr><td>Information Ratio</td><td align="right" style="color:{C_PURP};">{info_ratio:.3f}</td></tr>
        <tr><td>Alpha de Jensen Expost</td><td align="right" style="color:{C_PINK};">{alpha*100:.2f}%</td></tr>
        <tr><td>Condición Value at Risk (VaR al {conf_level}%)</td><td align="right" style="color:#f97316;">{var_q*100:.2f}% Barrera</td></tr>
        <tr><td>Expected Shortfall Promedio (CVaR)</td><td align="right" style="color:#ef4444;">{cvar_val:.2f}% Riesgo de Cola</td></tr>
        <tr><td>Ratio de Sortino (Penaliza Drawdowns)</td><td align="right" style="color:#10b981;">{sortino:.3f}</td></tr>
        <tr><td>Ratio de Treynor (Riesgo Sistemático)</td><td align="right" style="color:#3b82f6;">{treynor:.3f}</td></tr>
        <tr><td>Medida Modigliani (M²) Ajustada vs Benchmark</td><td align="right" style="color:#eab308;">{m2*100:.2f}%</td></tr>
        <tr><td>Ratio de Calmar (Retorno / Riesgo Extremo)</td><td align="right" style="color:#ec4899;">{calmar:.3f} (Max DD: {max_dd*100:.2f}%)</td></tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: MARKOWITZ & MONTE CARLO
# -------------------------------------------------------------
with tab2:
    cc1, cc2 = st.columns([1, 1.5])
    
    with cc1:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Matriz de Correlación Diaria</div>", unsafe_allow_html=True)
        corr_matrix = raw_returns[asset_list].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
            colorscale=[[0, '#0f0f0f'], [0.5, C_PURP], [1.0, C_CYAN]], text=np.round(corr_matrix.values, 2), texttemplate="%{text}"
        ))
        fig_corr = style_plotly(fig_corr)
        fig_corr.update_layout(height=350, yaxis_autorange='reversed')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with cc2:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Frontera Eficiente Estocástica (Markowitz)</div>", unsafe_allow_html=True)
        
        num_ports = 1500
        p_ret = []
        p_vol = []
        cov_annual = raw_returns[asset_list].cov() * 252
        mean_ret_annual = raw_returns[asset_list].mean() * 252
        
        for _ in range(num_ports):
            weights = np.random.random(len(asset_list))
            weights /= np.sum(weights)
            p_ret.append(np.dot(weights, mean_ret_annual))
            p_vol.append(np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights))))
            
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=p_vol, y=p_ret, mode='markers', marker=dict(color=p_ret, colorscale='PuBuGn', size=5, opacity=0.6), name="Universo Random"))
        fig_ef.add_trace(go.Scatter(x=[v_anual], y=[r_anual], mode='markers', marker=dict(color=C_PINK, size=15, symbol='star', line=dict(color=C_CYAN, width=2)), name="TU PORTAFOLIO"))
        fig_ef = style_plotly(fig_ef)
        fig_ef.update_layout(height=350, xaxis_title="Riesgo (Volatilidad)", yaxis_title="Retorno Anualizado", xaxis=dict(showgrid=True, gridcolor='#222'), yaxis=dict(showgrid=True, gridcolor='#222'), showlegend=False)
        st.plotly_chart(fig_ef, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Monte Carlo Geometric Brownian Motion (Simulador de Precios)</div>", unsafe_allow_html=True)
    mc_sel = st.selectbox("Activo a Simular Futuro (1 Año, 50 Escenarios):", asset_list)
    s0 = float(price_data[mc_sel].iloc[-1])
    ret_asset = raw_returns[mc_sel]
    mu = float(ret_asset.mean() * 252)
    sigma = float(ret_asset.std() * np.sqrt(252))
    paths = geometric_brownian_motion(s0, mu, sigma)
    
    fig_mc = go.Figure()
    for i in range(50):
        fig_mc.add_trace(go.Scatter(y=paths[:,i], mode='lines', opacity=0.1, line=dict(width=1, color=C_CYAN)))
    fig_mc.add_trace(go.Scatter(y=np.mean(paths, axis=1), mode='lines', name='Promedio Proyectado', line=dict(width=3, color=C_PINK)))
    
    fig_mc = style_plotly(fig_mc)
    fig_mc.update_layout(height=400, showlegend=False, xaxis_title="Días Proyectados", yaxis_title="Precio Simulado USD", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#222', tickformat="$,.2f"))
    st.plotly_chart(fig_mc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: CHAOS THEORY (FRACTALS)
# -------------------------------------------------------------
with tab3:
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Análisis Fractal: Exponente de Hurst (R/S Proxy)</div>", unsafe_allow_html=True)
        h_sel = st.selectbox("Serie Temporal de Acción:", asset_list, key="h_sel")
        
        ts_prices = price_data[h_sel].values
        H, lags, tau = hurst_exponent(ts_prices)
        
        behavior = "Tendencia Fuerte" if H > 0.55 else "Reversión a la media" if H < 0.45 else "Caminata Aleatoria"
        col_tag = C_CYAN if H > 0.55 else C_PINK if H < 0.45 else "#9ca3af"
        
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=np.log(lags), y=np.log(tau), mode='markers', name="Datos T", marker_color=C_PURP))
        fig_h.add_trace(go.Scatter(x=np.log(lags), y=np.polyval(np.polyfit(np.log(lags), np.log(tau), 1), np.log(lags)), mode='lines', line=dict(color=C_CYAN, dash='dot'), name="Regresión"))
        fig_h = style_plotly(fig_h)
        fig_h.update_layout(height=300, xaxis_title="Log(Lag)", yaxis_title="Log(Tau/Varianza)", xaxis=dict(showgrid=True, gridcolor='#222'), yaxis=dict(showgrid=True, gridcolor='#222'))
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown(f"Exponente Computado: <span style='font-size:24px; color:{C_PURP}; font-weight:bold;'>{H:.3f}</span>", unsafe_allow_html=True)
        st.markdown(f"Clasificación Fractal: <span style='color:{col_tag}; font-weight:bold;'>{behavior}</span>", unsafe_allow_html=True)
        if H > 0.5:
             st.markdown("<span style='font-size:13px; color:#888;'>Indica memoria algorítmica: Lo que pasó ayer tiende a continuar afectando el futuro. Bueno para Trend-Following.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_y:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Divergencia Estocástica (Exponente Lyapunov Proxy)</div>", unsafe_allow_html=True)
        
        lyap_ret = raw_returns[h_sel].values
        L_exp, div_rates = lyapunov_exponent(lyap_ret)
        
        chaotic = "Altamente Caótico (No predecible)" if L_exp > 0.5 else "Estable / Predecible a Corto Plazo"
        col_l = C_PINK if L_exp > 0.5 else C_CYAN
        
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(y=div_rates[-100:], mode='lines', line=dict(shape='spline', color=C_PINK, width=2), fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'))
        fig_l = style_plotly(fig_l)
        fig_l.update_layout(height=300, xaxis_title="Últimos 100 periodos", yaxis_title="Tasa de Divergencia Local (λ)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#222'))
        st.plotly_chart(fig_l, use_container_width=True)
        
        st.markdown(f"Máximo Exponente Lyapunov (λ): <span style='font-size:24px; color:#ffffff; font-weight:bold;'>{L_exp:.3f}</span>", unsafe_allow_html=True)
        st.markdown(f"Atractor Dinámico: <span style='color:{col_l}; font-weight:bold;'>{chaotic}</span>", unsafe_allow_html=True)
        if L_exp > 0:
             st.markdown("<span style='font-size:13px; color:#888;'>Advierte dependencia altísima de condiciones iniciales. Se desaconsejan las predicciones de tiempo largo (Efecto Mariposa).</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 4: DEEP ANALYTICS & HORIZONS (PROYECCIONES Y DRAWDOWN)
# -------------------------------------------------------------
with tab4:
    st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Macro Proyección Estocástica (Capital Inicial: $100K ➔ 20 Años)</div>", unsafe_allow_html=True)
    
    t_years = 20
    d_steps = 252 * t_years
    paths_20y = geometric_brownian_motion(100000, r_anual, v_anual, T=t_years, N=d_steps, paths=20)
    
    fig_mc20 = go.Figure()
    for i in range(20):
        # Downsample visual a 1 vez al mes temporalmente (cada 21 periodos diarios) para no saturar WebGL
        fig_mc20.add_trace(go.Scatter(y=paths_20y[::21, i], mode='lines', opacity=0.15, line=dict(width=1, color=C_CYAN)))
    
    fig_mc20.add_trace(go.Scatter(y=np.mean(paths_20y[::21], axis=1), mode='lines', name='Valor Promedio Esperado AUM', line=dict(width=3, color=C_PINK)))
    fig_mc20 = style_plotly(fig_mc20)
    
    fig_mc20.update_layout(height=400, showlegend=True, xaxis_title="Tiempo Transcurrido (Meses en el Futuro)", yaxis_title="Valuación del Portafolio ($ USD)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#222', tickformat="$,.0f"))
    st.plotly_chart(fig_mc20, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_uw, col_hm = st.columns(2)
    with col_uw:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Underwater Curve (Pain Index Histórico)</div>", unsafe_allow_html=True)
        fig_uw = go.Figure()
        fig_uw.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, mode='lines', fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)', line=dict(color='#ef4444', width=2)))
        fig_uw = style_plotly(fig_uw)
        fig_uw.update_layout(height=300, xaxis_title="Eje de Tiempo Real Expost", yaxis_title="Porcentaje de Caída en Patrimonio (%)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#222'))
        st.plotly_chart(fig_uw, use_container_width=True)
        st.markdown(f"Maximum Drawdown (MDD): <span style='font-size:24px; color:#ef4444; font-weight:bold;'>{max_dd*100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_hm:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Return Seasonality (Heatmap de Retornos)</div>", unsafe_allow_html=True)
        
        df_ret = pd.DataFrame({'Ret': port_ret * 100})
        df_ret['Year'] = df_ret.index.year
        df_ret['Month'] = df_ret.index.month
        pt = df_ret.pivot_table(values='Ret', index='Year', columns='Month', aggfunc='sum')
        month_map = {1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun", 7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"}
        pt.columns = [month_map.get(m, m) for m in pt.columns]
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=pt.values, x=pt.columns, y=pt.index, text=np.round(pt.values, 1), texttemplate="%{text}%",
            colorscale=[[0, '#ef4444'], [0.5, '#0f0f0f'], [1.0, '#10b981']], zmid=0, showscale=False
        ))
        fig_hm = style_plotly(fig_hm)
        fig_hm.update_layout(height=345, yaxis_autorange='reversed', yaxis=dict(type="category"))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
