# -modelo-produtividade-trigo.
Modelo de produtividade do trigo usando dados climáticos, fenologia, solo e manejo agrícola. Utiliza XGBoost com ajuste de hiperparâmetros, avaliação comparativa com Random Forest e visualizações interativas. Projeto em Python com exportação de resultados para Excel.
# Modelo de Produtividade do Trigo usando XGBoost

Este projeto apresenta um pipeline completo para modelar a produtividade do trigo com base em dados climáticos, fenológicos, características do solo e manejo agrícola, utilizando técnicas de Machine Learning.

---

## Conteúdo

- Leitura e pré-processamento de dados climáticos a partir de arquivos NetCDF
- Simulação de dados fenológicos, solo e manejo agrícola
- Treinamento e ajuste de modelo XGBoost com GridSearchCV
- Avaliação do modelo e comparação com Random Forest
- Validação cruzada para robustez
- Visualização interativa de resíduos e importância das variáveis usando Plotly
- Exportação dos resultados (predições, importância, resíduos, parâmetros) para arquivo Excel com múltiplas abas

---

## Tecnologias e Bibliotecas Utilizadas

- Python 3.x
- numpy, pandas
- netCDF4
- scikit-learn
- xgboost
- plotly
- joblib
- openpyxl (para salvar Excel)
- matplotlib, seaborn (para possíveis visualizações adicionais)

---

## Como usar

1. **Instale as dependências**:
   ```bash
   pip install numpy pandas netCDF4 scikit-learn xgboost plotly joblib openpyxl matplotlib seaborn
