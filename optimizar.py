import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

assets = ["VSAT", "PTEN", "HL", "CDE", "ICHR", "WMT", "BAX", "AVGO", "JPM", "GLD", "SGOV"]
end = datetime.today()
start = end - timedelta(days=2*365)

print("⏳ Descargando datos de mercado...")
data = yf.download(assets, start=start, end=end, progress=False)['Close'].ffill().dropna()
returns = np.log(data / data.shift(1)).dropna()
returns = returns[assets]

mean_ret = returns.mean().values * 252
cov_matrix = returns.cov().values * 252
rf = 0.045

# Benchmark
bench_data = yf.download("^IXIC", start=start, end=end, progress=False)['Close'].ffill().dropna()
bench_ret = np.log(bench_data / bench_data.shift(1)).dropna()
bench_mean = float(bench_ret.mean()) * 252
bench_vol = float(bench_ret.std()) * np.sqrt(252)

n_assets = len(assets)
n_sim = 1_000_000

print(f"🚀 Corriendo {n_sim:,} simulaciones Monte Carlo...")

# Generar todos los pesos aleatorios de una vez (vectorizado)
rng = np.random.default_rng(42)
all_weights = rng.random((n_sim, n_assets))
all_weights = all_weights / all_weights.sum(axis=1, keepdims=True)

# Calcular retornos y volatilidades vectorizados
port_returns = all_weights @ mean_ret
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, cov_matrix, all_weights))
sharpes = (port_returns - rf) / port_vols

# Calcular betas vectorizados (aproximación usando retornos del portafolio)
# Para beta necesitamos los retornos diarios del portafolio
returns_arr = returns.values
bench_arr = bench_ret.values[-len(returns_arr):]
if len(bench_arr) > len(returns_arr):
    bench_arr = bench_arr[:len(returns_arr)]
elif len(returns_arr) > len(bench_arr):
    returns_arr = returns_arr[:len(bench_arr)]

bench_var = np.var(bench_arr)

# Para los top candidatos calcularemos beta exacto
# Primero encontramos el mejor Sharpe
idx_best_sharpe = np.argmax(sharpes)

# Encontrar mejor retorno con vol razonable (< mediana de vol)
median_vol = np.median(port_vols)
mask_moderate = port_vols < median_vol
idx_best_ret_moderate = np.where(mask_moderate)[0][np.argmax(port_returns[mask_moderate])]

# Encontrar mínima volatilidad
idx_min_vol = np.argmin(port_vols)

# Calcular beta para los candidatos
def calc_beta(weights):
    port_daily = returns_arr @ weights
    cov_pb = np.cov(port_daily, bench_arr.flatten())[0, 1]
    return cov_pb / bench_var

def calc_metrics(weights, label):
    w = weights
    ret = w @ mean_ret
    vol = np.sqrt(w @ cov_matrix @ w)
    sr = (ret - rf) / vol
    beta = calc_beta(w)
    alpha = ret - (rf + beta * (bench_mean - rf))
    
    # Sortino
    port_daily = returns_arr @ w
    rf_d = rf / 252
    downside = port_daily[port_daily < rf_d] - rf_d
    down_std = np.sqrt(np.mean(downside**2)) * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (ret - rf) / down_std if down_std > 0 else 0
    
    # Max Drawdown
    cum = (1 + port_daily).cumprod()
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    calmar = ret / abs(max_dd) if max_dd != 0 else 0
    
    # VaR
    var_95 = np.percentile(port_daily, 5) * 100
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"\n  {'Activo':<8} {'Peso':>10}")
    print(f"  {'-'*20}")
    for a, wt in zip(assets, w):
        print(f"  {a:<8} {wt*100:>9.2f}%")
    print(f"\n  {'─'*40}")
    print(f"  Retorno Anualizado:    {ret*100:>8.2f}%")
    print(f"  Volatilidad Anual:     {vol*100:>8.2f}%")
    print(f"  Sharpe Ratio:          {sr:>8.3f}")
    print(f"  Sortino Ratio:         {sortino:>8.3f}")
    print(f"  Beta (vs NASDAQ):      {beta:>8.3f}")
    print(f"  Alpha de Jensen:       {alpha*100:>8.2f}%")
    print(f"  Max Drawdown:          {max_dd*100:>8.2f}%")
    print(f"  Calmar Ratio:          {calmar:>8.3f}")
    print(f"  VaR Diario (95%):      {var_95:>8.2f}%")
    
    return sr

print(f"✅ Simulaciones completadas.")

# Mostrar resultados
sr1 = calc_metrics(all_weights[idx_best_sharpe], "🏆 PORTAFOLIO ÓPTIMO (Máximo Sharpe Ratio)")
sr2 = calc_metrics(all_weights[idx_best_ret_moderate], "📈 MÁXIMO RETORNO (Vol < Mediana)")
sr3 = calc_metrics(all_weights[idx_min_vol], "🛡️  MÍNIMA VOLATILIDAD")

# Bonus: Optimización SLSQP para máximo Sharpe (determinista)
from scipy.optimize import minimize

def neg_sharpe(w):
    ret = w @ mean_ret
    vol = np.sqrt(w @ cov_matrix @ w)
    return -(ret - rf) / vol

cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0.0, 0.50) for _ in range(n_assets))
init = np.array([1/n_assets]*n_assets)
opt = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons)

calc_metrics(opt.x, "⚡ OPTIMIZACIÓN ANALÍTICA (SLSQP - Máximo Sharpe, max 50% por activo)")

print(f"\n{'='*60}")
print(f"  📊 RESUMEN: 1,000,000 simulaciones procesadas")
print(f"  Rango de Sharpe: [{sharpes.min():.3f} — {sharpes.max():.3f}]")
print(f"  Rango de Retorno: [{port_returns.min()*100:.2f}% — {port_returns.max()*100:.2f}%]")
print(f"  Rango de Volatilidad: [{port_vols.min()*100:.2f}% — {port_vols.max()*100:.2f}%]")
print(f"{'='*60}")
