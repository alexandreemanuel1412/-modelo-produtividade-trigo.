"""
Modelagem preditiva da produtividade do trigo usando dados climáticos simulados, 
dados fenológicos, tipo de solo e manejo agrícola. O pipeline inclui pré-processamento,
ajuste de hiperparâmetros com XGBoost, comparação com Random Forest, avaliação, 
gráficos interativos e exportação de resultados.
"""

from netCDF4 import Dataset, num2date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# === 1. LER DADOS CLIMÁTICOS ===
arquivo_nc = 'clima_trigo.nc'  # Substitua pelo caminho correto
ds = Dataset(arquivo_nc)

tempo = ds.variables['time'][:]
datas = num2date(tempo, units=ds.variables['time'].units)

precip = ds.variables['precipitation'][:, 0, 0]
tmax = ds.variables['tmax'][:, 0, 0]
tmin = ds.variables['tmin'][:, 0, 0]
ur = ds.variables['humidity'][:, 0, 0]
rad = ds.variables['solar_radiation'][:, 0, 0]

df_clima = pd.DataFrame({
    'data': datas,
    'precipitacao': precip,
    'temp_max': tmax,
    'temp_min': tmin,
    'umidade_relativa': ur,
    'radiacao_solar': rad
})

# === 2. SIMULAR DADOS FENOLOGIA/SOLO/MANEJO E PRODUTIVIDADE ===
np.random.seed(42)
df_clima['fase_fenologica'] = np.random.choice(['perfilhamento', 'espigamento', 'enchimento'], size=len(df_clima))
df_clima['tipo_solo'] = np.random.choice(['argiloso', 'arenoso', 'siltoso'], size=len(df_clima))
df_clima['manejo'] = np.random.choice(['convencional', 'plantio_direto'], size=len(df_clima))

df_clima['produtividade'] = (
    20 * df_clima['radiacao_solar'] +
    5 * df_clima['precipitacao'] / 10 -
    3 * np.abs(df_clima['temp_max'] - 25) -
    2 * np.abs(df_clima['temp_min'] - 15) +
    np.random.normal(loc=0, scale=50, size=len(df_clima)) + 3000
)

# === 3. PRÉ-PROCESSAMENTO ===
dados = df_clima.copy()
categoricas = ['fase_fenologica', 'tipo_solo', 'manejo']
le_dict = {col: LabelEncoder().fit(dados[col]) for col in categoricas}
for col, le in le_dict.items():
    dados[col] = le.transform(dados[col])

X = dados.drop(columns=['data', 'produtividade'])
y = dados['produtividade']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 4. TREINAMENTO - XGBOOST COM GRIDSEARCHCV ===
param_grid = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'learning_rate': [0.1],
    'subsample': [0.7, 1.0]
}

xgb = XGBRegressor(random_state=42, n_jobs=1)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3, verbose=1, n_jobs=1)
grid_search.fit(X_train, y_train)

melhor_modelo = grid_search.best_estimator_
print("✅ Melhores parâmetros:", grid_search.best_params_)

# === 5. SALVAR MODELO E SCALER ===
joblib.dump(melhor_modelo, "modelo_xgboost_trigo.pkl")
joblib.dump(scaler, "scaler_trigo.pkl")

# === 6. AVALIAÇÃO DO MODELO ===
y_pred = melhor_modelo.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Avaliação do XGBoost:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# === 7. COMPARAÇÃO COM RANDOM FOREST ===
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\n🌲 Avaliação do Random Forest:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²: {r2_rf:.4f}")

# === 8. VALIDAÇÃO CRUZADA COM XGBOOST ===
scores = cross_val_score(melhor_modelo, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"\n🔁 RMSE médio (validação cruzada): {rmse_scores.mean():.2f}")

# === 9. GRÁFICO INTERATIVO DE RESÍDUOS ===
residuos = y_test - y_pred
fig_residuos = px.scatter(x=y_pred, y=residuos,
                          labels={'x': 'Valores Preditos', 'y': 'Resíduos'},
                          title='Gráfico Interativo de Resíduos (XGBoost)')
fig_residuos.add_hline(y=0, line_dash="dash", line_color="red")
fig_residuos.show()

# === 10. IMPORTÂNCIA DAS VARIÁVEIS ===
importancias = melhor_modelo.feature_importances_
nomes_colunas = dados.drop(columns=['data', 'produtividade']).columns

fig_importancia = px.bar(x=nomes_colunas, y=importancias,
                         labels={'x': 'Variável', 'y': 'Importância'},
                         title='Importância das Variáveis no Modelo XGBoost')
fig_importancia.update_layout(xaxis_tickangle=-45)
fig_importancia.show()

# === 11. EXPORTAÇÃO DOS RESULTADOS ===

# Inverter o escalonamento
X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)

# Reconverter categorias
for col, le in le_dict.items():
    X_test_df[col] = le.inverse_transform(X_test_df[col].astype(int))

# DataFrame com predições
df_predicoes = X_test_df.copy()
df_predicoes['produtividade_real'] = y_test.values
df_predicoes['produtividade_predita'] = y_pred
df_predicoes.to_csv("predicoes_trigo.csv", index=False)
print("\n📁 Arquivo 'predicoes_trigo.csv' salvo com sucesso!")

# Exportar para Excel com múltiplas abas
df_importancia = pd.DataFrame({
    'variavel': nomes_colunas,
    'importancia': importancias
}).sort_values(by='importancia', ascending=False)

df_residuos = pd.DataFrame({
    'produtividade_real': y_test.values,
    'produtividade_predita': y_pred,
    'residuo': residuos
})

df_parametros = pd.DataFrame({
    'parametro': list(melhor_modelo.get_params().keys()),
    'valor': list(melhor_modelo.get_params().values())
})

melhor_parametros = pd.DataFrame(grid_search.best_params_.items(), columns=["parametro", "valor"])

with pd.ExcelWriter("analise_modelo_trigo.xlsx", engine="openpyxl") as writer:
    df_predicoes.to_excel(writer, sheet_name="predicoes", index=False)
    df_importancia.to_excel(writer, sheet_name="importancia_variaveis", index=False)
    df_residuos.to_excel(writer, sheet_name="residuos", index=False)
    df_parametros.to_excel(writer, sheet_name="parametros_modelo", index=False)
    melhor_parametros.to_excel(writer, sheet_name="melhores_hiperparametros", index=False)

print("\n📊 Arquivo 'analise_modelo_trigo.xlsx' salvo com sucesso!")
