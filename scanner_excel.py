import pandas as pd
import sys

ruta = "/Users/sergiozendejas/Downloads/portafolio reto.xlsx"
print(f"Abriendo: {ruta}")

try:
    xls = pd.ExcelFile(ruta)
    print("Hojas encontradas:", xls.sheet_names)
    
    for hoja in xls.sheet_names:
        print(f"\n--- EXPLORANDO HOJA: {hoja} ---")
        df = pd.read_excel(xls, sheet_name=hoja, header=None)
        
        # Buscar palabras clave en las primeras 50 filas
        found_keywords = False
        for r in range(min(50, df.shape[0])):
            fila = df.iloc[r].dropna().astype(str).str.lower().tolist()
            if any('sharpe' in x or 'ratio' in x or 'rendimiento' in x or 'ticker' in x for x in fila):
                print(f"¡INFO CLAVE ENCONTRADA en Fila {r}! -> {fila}")
                found_keywords = True
        
        if not found_keywords:
            print(f"Vista previa (primeras 5 filas):\n{df.head(5)}")
                
except Exception as e:
    print(f"Error: {e}")
