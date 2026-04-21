import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

def load_and_parse_excel(file_uploader):
    """
    Lee el Excel binario provisto por Streamlit, y limpia el formato 
    separando el Benchmark y los Activos.
    """
    try:
        excel_data = pd.read_excel(file_uploader, sheet_name=None, engine='openpyxl', header=None)
    except Exception as e:
        raise ValueError(f"Error parseando archivo Excel: {e}")

    # Regla: Ignorar 'sheet 1' si existe
    if 'sheet 1' in excel_data:
        del excel_data['sheet 1']

    if not excel_data:
        raise ValueError("El archivo Excel no tiene datos válidos después de procesarlo.")

    nombre_hoja = list(excel_data.keys())[0]
    df = excel_data[nombre_hoja]

    # Detectar dinámicamente la fila de Fechas
    date_r, date_c = None, None
    for r in range(min(20, df.shape[0])):
        for c in range(df.shape[1]):
            val = str(df.iloc[r, c]).strip().lower()
            if val in ['date', 'fecha', 'fechas']:
                date_r, date_c = r, c
                break
        if date_r is not None:
            break
            
    if date_r is None:
        raise ValueError("No se pudo identificar la columna 'Date' en el Excel.")

    nombres_columnas = df.iloc[date_r].values
    
    # Inicio numérico
    inicio_datos = date_r + 1
    for r in range(date_r + 1, min(date_r + 20, df.shape[0])):
        try:
            test_val = df.iloc[r, date_c]
            pd.to_datetime(test_val)
            if str(test_val).lower() in ['beta', 'sector', 'market cap', 'price']:
                continue
            inicio_datos = r
            break
        except:
            pass

    df_precios = df.iloc[inicio_datos:].copy()
    df_precios.columns = [str(x) if x is not None and not pd.isna(x) else f"Col_{i}" for i, x in enumerate(nombres_columnas)]
    
    col_fecha = df_precios.columns[date_c]
    df_precios[col_fecha] = pd.to_datetime(df_precios[col_fecha])
    df_precios.set_index(col_fecha, inplace=True)
    df_precios = df_precios.apply(pd.to_numeric, errors='coerce')
    df_precios.dropna(axis=1, how='all', inplace=True)
    df_precios.ffill(inplace=True)
    df_precios.bfill(inplace=True)

    # Identificación del Benchmark (Generalmente 'index', 'S&P', o la última columna)
    nombre_benchmark = df_precios.columns[-1]
    for col in df_precios.columns:
        if any(x in col.lower() for x in ['index', 's&p', 'nasdaq', 'ipc']):
            nombre_benchmark = col
            break 
            
    mercado = df_precios.pop(nombre_benchmark)
    
    return df_precios, mercado, nombre_benchmark

def calculate_returns_and_beta(df_precios, df_mercado, risk_free_rate=0.02):
    retornos_activos = np.log(df_precios / df_precios.shift(1)).dropna()
    retornos_mercado = np.log(df_mercado / df_mercado.shift(1)).dropna()
    
    retornos_activos, retornos_mercado = retornos_activos.align(retornos_mercado, join='inner', axis=0)
    
    cov_mercado = retornos_mercado.var()
    betas = {}
    
    for activo in retornos_activos.columns:
        cov_matrix = np.cov(retornos_activos[activo], retornos_mercado)
        cov_activo_mercado = cov_matrix[0, 1]
        beta = cov_activo_mercado / cov_mercado if cov_mercado != 0 else 1.0
        betas[activo] = beta
        
    # Anualización
    retornos_anualizados = retornos_activos.mean() * 252
    matriz_covarianza = retornos_activos.cov() * 252
    
    return retornos_activos, pd.Series(betas), retornos_anualizados, matriz_covarianza

def monte_carlo_frontier(retornos_anualizados, matriz_covarianza, retornos_diarios, num_portfolios=50000, risk_free_rate=0.02, min_var=-2.0, max_var=2.0):
    """
    Genera la frontera eficiente a través de miles de escenarios de rebalanceo estocástico.
    Aplica el filtro para restringir que el Value at Risk diario se mantenga entre los límites pedidos.
    """
    num_assets = len(retornos_anualizados)
    resultados = np.zeros((4, num_portfolios)) # Rendimiento, Riesgo, Sharpe, VaR_pct
    pesos_record = []
    
    z_95 = stats.norm.ppf(0.95)
    retornos_promedio_diarios = retornos_diarios.mean()
    cov_diaria = retornos_diarios.cov()

    # Operaciones Vectorizadas en bloques para velocidad
    np.random.seed(42)
    # Generar pesos (Suman estrictamente 1 por la normalización, cumple sin ventas cortas al usar Random uniform(0,1))
    weights = np.random.random((num_portfolios, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    
    # Rendimiento esperado anual
    port_returns = np.dot(weights, retornos_anualizados)
    
    # Riesgo del portafolio (Anual y Diario)
    # Volatilidad anual
    port_vols = np.sqrt(np.einsum('ij,ji->i', weights, np.dot(matriz_covarianza.values, weights.T)))
    # Volatilidad diaria
    port_vols_daily = np.sqrt(np.einsum('ij,ji->i', weights, np.dot(cov_diaria.values, weights.T)))
    
    port_returns_daily = np.dot(weights, retornos_promedio_diarios)
    
    # VaR Diario 95% Paramétrico
    var_diario = (z_score_x_vol_diaria := z_95 * port_vols_daily) - port_returns_daily
    var_diario_pct = -np.abs(var_diario * 100) # Formato en pérdida % por ej: -2.3%
    
    sharpe_ratios = (port_returns - risk_free_rate) / port_vols
    
    # Envolver resultados (Para facilitar, Var absoluto entre 0 y 2%)
    # El usuario pide entre -2% y +2%. Traducido en finanzas es: La pérdida máxima permitida no
    # debe exceder al 2%. (-VaR pct se ubica arriba de -2%).
    
    resultados[0,:] = port_returns
    resultados[1,:] = port_vols
    resultados[2,:] = sharpe_ratios
    resultados[3,:] = var_diario_pct
    
    # Filtrar aquellos que superan la estricta cuota VaR (i.e. var_diario_pct >= -2.0)
    # El usuario dice que limite el var estrictamente entre -2% y 2%. Asumimos un techo VaR.
    vaR_mask = (resultados[3,:] >= -abs(min_var)) & (resultados[3,:] <= abs(max_var))
    
    # Si no hay ninguno que cumpla un umbral tan estricto con los activos base, se suaviza
    # automáticamente o simplemente devolvemos la red general resaltando el problema
    hay_valido = np.any(vaR_mask)
    if not hay_valido:
        # Se escoge sin restringir del todo si matemáticamente choca dado el input, pero se advierte.
        vaR_mask = np.ones(num_portfolios, dtype=bool)

    pesos_filtrados = weights[vaR_mask]
    res_filtrados = resultados[:, vaR_mask]
    
    best_sharpe_idx = np.argmax(res_filtrados[2,:])
    min_vol_idx = np.argmin(res_filtrados[1,:])

    parametros_optimos = {
        'max_sharpe': {
            'pesos': pesos_filtrados[best_sharpe_idx],
            'return': res_filtrados[0, best_sharpe_idx],
            'volatility': res_filtrados[1, best_sharpe_idx],
            'sharpe': res_filtrados[2, best_sharpe_idx],
            'var': res_filtrados[3, best_sharpe_idx]
        },
        'min_vol': {
            'pesos': pesos_filtrados[min_vol_idx],
            'return': res_filtrados[0, min_vol_idx],
            'volatility': res_filtrados[1, min_vol_idx],
            'sharpe': res_filtrados[2, min_vol_idx],
            'var': res_filtrados[3, min_vol_idx]
        }
    }
    
    # Para visualización masiva devolvemos arrays acotados numéricamente (samplear 7000 para no laguear Frontend)
    subset_indices = np.random.choice(res_filtrados.shape[1], min(10000, res_filtrados.shape[1]), replace=False)
    
    return res_filtrados[:, subset_indices], parametros_optimos, hay_valido

def calcular_matriz_var(pesos, retornos_historicos, capital=100000):
    ret_pf = np.dot(pesos, retornos_historicos.mean())
    vol_pf = np.sqrt(np.dot(pesos.T, np.dot(retornos_historicos.cov(), pesos)))
    
    escalas = {'Diario': 1, 'Mensual': 21, 'Anual': 252}
    confianzas = {'95%': 1.645, '99%': 2.326}
    
    var_matrix = {}
    for cl_name, cl_z in confianzas.items():
        var_matrix[cl_name] = {}
        for temp_name, temp_t in escalas.items():
            vol_ajustada = vol_pf * np.sqrt(temp_t)
            ret_ajustado = ret_pf * temp_t
            
            var_pct = (cl_z * vol_ajustada) - ret_ajustado
            var_dinero = var_pct * capital
            var_matrix[cl_name][temp_name] = {'Pct': -abs(var_pct*100), 'Valor': -abs(var_dinero)}
            
    return var_matrix

def monte_carlo_prices(precio_actual_portafolio, volatilidad_anual, rendimiento_anual, trayectorias=50000, dias=252):
    """
    Simulación Avanzada de Trayectorias (Movimiento Browniano Geométrico)
    a 1 año del portafolio construido.
    """
    dt = 1/252
    precios = np.zeros((dias, trayectorias))
    precios[0] = precio_actual_portafolio
    
    # Vectorizamos la evolución del precio para altísima velocidad de cómputo
    shocks_aleatorios = np.random.normal(0, 1, size=(dias - 1, trayectorias))
    r_diaria = rendimiento_anual / 252
    vol_diaria = volatilidad_anual / np.sqrt(252)
    
    factor_drift = r_diaria - (0.5 * vol_diaria**2)
    delta_precios = np.exp(factor_drift + vol_diaria * shocks_aleatorios)
    
    for t in range(1, dias):
        precios[t] = precios[t-1] * delta_precios[t-1]
        
    return precios

def technical_indicators(df_serie):
    """
    Produce SMA, MACD, y RSI requeridos y mapea picos para Soporte y Resistencia 
    usando heurísticas de conglomerados 1D (Clustering de K-Means) o ventanas de mínimo.
    """
    df = df_serie.to_frame(name='Close')
    
    # Promedios móviles
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) # Prevención division by zero
    
    # Soportes y Resistencias (Básico vía máximos/mínimos locales y K-Means simple en un Array 1D)
    valid_prices = df['Close'].dropna().values.reshape(-1, 1)
    soportes, resistencias = [], []
    if len(valid_prices) > 60:
        try:
            # Encontrar niveles de liquidez (precio altamente frecuente en rebotes)
            # Extrayendo clusters base
            km = KMeans(n_clusters=4, random_state=0).fit(valid_prices)
            niveles = sorted(km.cluster_centers_.flatten())
            soportes = niveles[:2]
            resistencias = niveles[-2:]
        except:
            pass

    return df, soportes, resistencias
