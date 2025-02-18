import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Step 1: Load the Cleaned Dataset
data_path = 'cleaned-climate-change_dataset.csv'
df = pd.read_csv(data_path)

# Step 2: Exploratory Data Analysis (EDA)
st.title("🌱 Climate Change Forecasting")
st.write("### Data Overview")
st.dataframe(df.head())

# Correlation Heatmap
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
st.pyplot(plt)

# Temperature Trend Visualization
st.write("### Temperature Trend Over the Years")
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Temperature', data=df)
st.pyplot(plt)

# Step 3: Feature Scaling
numerical_cols = ['Temperature', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)',
                  'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
                  'Extreme Weather Events', 'Forest Area (%)']
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 4: Train ARIMA Model for Temperature Prediction
st.write("### Model Training (ARIMA)")
train_data = df['Temperature'][:-12]
test_data = df['Temperature'][-12:]
model = ARIMA(train_data, order=(5, 1, 2))
model_fit = model.fit()
st.write(model_fit.summary())

# Step 5: Model Forecasting
forecast = model_fit.forecast(steps=12)
st.write("### Forecasted Temperature for Next 12 Months")
st.line_chart(pd.DataFrame({'Actual': test_data.values, 'Forecast': forecast.values}))

# Step 6: Model Evaluation
mse = mean_squared_error(test_data, forecast)
mae = mean_absolute_error(test_data, forecast)
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

# Step 7: Impact Assessment
st.write("### Climate Change Impact Assessment")
impact_df = pd.DataFrame({'Actual': test_data.values, 'Forecast': forecast.values})
impact_df['Difference'] = impact_df['Actual'] - impact_df['Forecast']
st.bar_chart(impact_df['Difference'])

st.success("🌎 Climate Change Forecasting Project Execution Completed Successfully!")
