import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import plotly.express as px

def generate_temp_avg_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'temp_avg'])

    fig = px.line(df, x='date', y='temp_avg', markers=True,
                  title="Середня температура з часом",
                  labels={'date': 'Дата', 'temp_avg': 'Температура (°C)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Температура: %{y:.2f}°C<extra></extra>')
    return fig.to_html(full_html=False)



def generate_temp_max_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'temp_max'])

    fig = px.line(df, x='date', y='temp_max', markers=True,
                  title="Максимальна температура з часом",
                  labels={'date': 'Дата', 'temp_max': 'Температура (°C)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Температура: %{y:.2f}°C<extra></extra>')
    return fig.to_html(full_html=False)

def generate_temp_min_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'temp_min'])

    fig = px.line(df, x='date', y='temp_min', markers=True,
                  title="Мінімальна температура з часом",
                  labels={'date': 'Дата', 'temp_min': 'Температура (°C)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Температура: %{y:.2f}°C<extra></extra>')
    return fig.to_html(full_html=False)

def generate_precipitation_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'precipitation'])

    fig = px.line(df, x='date', y='precipitation', markers=True,
                  title="Кількість опадів з часом",
                  labels={'date': 'Дата', 'precipitation': 'Кількість опадів (мм)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Опади: %{y:.2f} мм<extra></extra>')
    return fig.to_html(full_html=False)

def generate_pressure_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'pressure'])

    fig = px.line(df, x='date', y='pressure', markers=True,
                  title="Сила атмосферного тиску з часом",
                  labels={'date': 'Дата', 'pressure': 'Тиск (мм. рт. ст.)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Тиск: %{y:.2f} мм рт. ст.<extra></extra>')
    return fig.to_html(full_html=False)

def generate_wind_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'wind'])   # <-- fixed (was pressure in your code)

    fig = px.line(df, x='date', y='wind', markers=True,
                  title="Сила вітру з часом",
                  labels={'date': 'Дата', 'wind': 'Сила вітру (км/г)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Вітер: %{y:.2f} км/г<extra></extra>')
    return fig.to_html(full_html=False)

def generate_humidity_plot(csv_path: str) -> str:
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'humidity'])

    fig = px.line(df, x='date', y='humidity', markers=True,
                  title="Вологість повітря з часом",
                  labels={'date': 'Дата', 'humidity': 'Вологість (%)'})
    fig.update_traces(hovertemplate='Дата: %{x}<br>Вологість: %{y:.2f}%<extra></extra>')
    return fig.to_html(full_html=False)


def generate_linear_regression_plots_all(csv_path: str) -> dict:
    columns = ['temp_avg', 'temp_max', 'temp_min', 'precipitation', 'humidity', 'wind', 'pressure']
    result = {}

    # Load and preprocess CSV
    df = pd.read_csv(
        csv_path,
        delimiter=';',
        decimal=',',
        na_values=['']
    )
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')

    for col in columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in columns:
        df_clean = df.dropna(subset=['date', col]).reset_index()
        if df_clean.empty:
            result[col] = None
            continue

        df_clean["date_ordinal"] = df_clean["date"].map(pd.Timestamp.toordinal)
        X = df_clean["date_ordinal"].values.reshape(-1, 1)
        y = df_clean[col].values

        model = LinearRegression()
        model.fit(X, y)
        df_clean["pred"] = model.predict(X)

        last_date = df_clean["date"].max()

        # Create plots
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # --- 1 day extrapolation ---
        future_dates_1 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1, freq='D')
        future_ordinals_1 = future_dates_1.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_preds_1 = model.predict(future_ordinals_1)

        combined_1 = pd.concat([
            df_clean[["date", col, "pred"]],
            pd.DataFrame({"date": future_dates_1, "pred": future_preds_1})
        ], ignore_index=True)

        axs[0].plot(combined_1["date"], combined_1["pred"], color='red', label="Прогноз регресії", linestyle="--")
        axs[0].scatter(df_clean["date"], df_clean[col], label="Фактичні дані", color='blue', s=15)
        axs[0].axvline(x=last_date, color='gray', linestyle=':', label="Початок екстраполяції")
        axs[0].set_title(f"{col} → Лінійна регресія +1 день")
        axs[0].set_ylabel(col)
        axs[0].legend()
        axs[0].grid(True)

        # --- 300 days extrapolation ---
        future_dates_300 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=300, freq='D')
        future_ordinals_300 = future_dates_300.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_preds_300 = model.predict(future_ordinals_300)

        axs[1].plot(df_clean["date"], df_clean["pred"], label="Прогноз (історія)", linestyle="--", color="cyan")
        axs[1].plot(future_dates_300, future_preds_300, label="Екстраполяція +300 днів", linestyle="--", color="orange")
        axs[1].axvline(x=last_date, color='gray', linestyle=':', label="Початок екстраполяції")
        axs[1].set_title(f"{col} → Лінійна регресія +300 днів")
        axs[1].set_xlabel("Дата")
        axs[1].set_ylabel(col)
        axs[1].legend()
        axs[1].grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        encoded = base64.b64encode(buf.read()).decode('utf-8')

        result[col] = encoded

    return result


def generate_prophet_forecast_plot(csv_path: str, future_days: int, column_name: str):
    df = pd.read_csv(csv_path, delimiter=';', decimal=',', na_values=[''])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    df[column_name] = df[column_name].astype(str).str.replace(',', '.', regex=False)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=['date', column_name])

    prophet_df = df.rename(columns={'date': 'ds', column_name: 'y'})[['ds', 'y']]

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=future_days, freq='D')
    forecast = model.predict(future)
    future_only = forecast[forecast['ds'] > prophet_df['ds'].max()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'],
                             mode='lines+markers', name='Forecast',
                             hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
                             line=dict(color='green')))
    fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'],
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'],
                             fill='tonexty', fillcolor='lightgreen',
                             line=dict(width=0), showlegend=True, name='Confidence Interval'))

    fig.update_layout(title=f"Next {future_days} Days Forecast for {column_name}",
                      xaxis_title='Date', yaxis_title=f'Predicted {column_name}',
                      template='plotly_white')

    return fig.to_html(full_html=False), future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
