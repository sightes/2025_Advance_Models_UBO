
import pandas as pd
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from pmdarima import auto_arima
from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig, MetadataParam, ModelComponentsParam
)
from greykite.framework.templates.forecaster import Forecaster
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import GRU

from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import pandas as pd
import optuna
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

import logging
import prophet
import cmdstanpy

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import optuna
import numpy as np



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import optuna
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
import optuna

# Silencia cmdstanpy
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Opcional: también silencia Prophet si deseas
logging.getLogger("prophet").setLevel(logging.WARNING)


# Reducir nivel de logging de Prophet
prophet_logger = logging.getLogger('prophet')
prophet_logger.setLevel(logging.WARNING)

# Reducir logging de cmdstanpy
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.WARNING)

def fit_predict_eval_hw(training_set, test_set, model_params=None):
    """
    Entrena y predice con modelo Holt-Winters.
    """
    if model_params is None:
        model_params = {
            'trend': 'add',
            'seasonal': 'add',
            'use_boxcox': False,
            'seasonal_periods': 12
        }

    model = ExponentialSmoothing(
        endog=training_set['y'].dropna(),
        initialization_method='estimated',
        freq='MS',
        seasonal_periods=model_params['seasonal_periods'],
        trend=model_params['trend'],
        seasonal=model_params['seasonal'],
        use_boxcox=model_params['use_boxcox']
    )

    ets = model.fit()
    y_pred = pd.Series(
        ets.predict(start=test_set.index[0], end=test_set.index[-1]),
        index=test_set.index,
        name='ETS',
       # optimized=True,
    #   use_brute=True,  # útil para asegurar búsqueda global
    #method='Nelder-Mead',  
    )
    return ets, y_pred, None  # scaler is None


def fit_predict_eval_autoarima(training_set, test_set, model_params=None):
    """
    Entrena y predice con AutoARIMA usando pmdarima.
    """
    if model_params is None:
        model_params = {
            'seasonal': True,
            'm': 12,
            'stepwise': True,
            'suppress_warnings': True,
            'error_action': 'ignore'
        }

    if not model_params.get('seasonal', True):
        model_params['m'] = 1  # Evitar m > 1 si seasonal = False

    y_train = training_set['y'].dropna()
    y_test = test_set['y'].dropna()

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Training o test set está vacío después de eliminar NaNs.")

    model = auto_arima(y_train, **model_params)
    y_pred = model.predict(n_periods=len(y_test))
    y_pred_series = pd.Series(y_pred, index=test_set.index[:len(y_pred)], name='AutoARIMA')

    return model, y_pred_series, None


# modelos.py (extensión)


def fit_predict_eval_silverkite(training_set, test_set, model_params=None):
    import warnings
    warnings.filterwarnings("ignore", module="greykite")  # Limpia output
    """
    Entrena y predice con modelo Silverkite de Greykite.
    """
    if model_params is None:
        model_params = {
            'growth': 'linear',
            'fit_algorithm': 'ridge',
            'regularization_strength': 0.6
        }

    date_col, target_col = 'ds', 'y'
    train_data = training_set.copy()
    test_data = test_set.copy()
    if train_data.index.name is not None:
        train_data = train_data.reset_index()
    if test_data.index.name is not None:
        test_data = test_data.reset_index()

    if 'ds' not in train_data.columns:
        train_data.rename(columns={train_data.columns[0]: 'ds'}, inplace=True)
    if 'y' not in train_data.columns:
        train_data.rename(columns={train_data.columns[-1]: 'y'}, inplace=True)
    if 'ds' not in test_data.columns:
        test_data.rename(columns={test_data.columns[0]: 'ds'}, inplace=True)
    if 'y' not in test_data.columns:
        test_data.rename(columns={test_data.columns[-1]: 'y'}, inplace=True)

    # Renombrar para Silverkite
    train = train_data.rename(columns={'ds': 'time', 'y': 'value'})
    test = test_data.rename(columns={'ds': 'time', 'y': 'value'})

    train['time'] = pd.to_datetime(train['time'], errors='coerce')
    test['time'] = pd.to_datetime(test['time'], errors='coerce')

    metadata = MetadataParam(
        time_col='time',
        value_col='value',
        freq='MS',
        train_end_date=train['time'].iloc[-1]
    )

    regressors = {
        "regressor_cols": [col for col in train.columns if col not in ['time', 'value']]
    }

    events = dict(
        auto_holiday=False,
        holidays_to_model_separately=[],
        holiday_lookup_countries=[],
    )

    changepoints = {
        "changepoints_dict": {
            "method": "custom",
            "regularization_strength": model_params['regularization_strength'],
            "resample_freq": "7D",
            "actual_changepoint_min_distance": "100D",
            "potential_changepoint_distance": "50D",
            "no_changepoint_proportion_from_end": 0.3,
            "yearly_seasonality_order": 6,
            "dates": ["2020-05-23", "2020-07-20", "2020-12-10", "2021-04-28"],
            "combine_changepoint_min_distance": "100D",
            "keep_detected": False
        }
    }

    custom = {
        "min_admissible_value": 0,
        "max_admissible_value": 2e7,
        "fit_algorithm_dict": {
            "fit_algorithm": model_params['fit_algorithm']
        }
    }

    model_components = ModelComponentsParam(
        regressors=regressors,
        changepoints=changepoints,
        events=events,
        custom=custom,
        growth={"growth_term": model_params['growth']}
    )

    config = ForecastConfig(
        forecast_horizon=len(test),
        coverage=0.95,
        metadata_param=metadata,
        model_components_param=model_components
    )

    forecaster = Forecaster()
    gk_result = forecaster.run_forecast_config(
        df=pd.concat([train, test]),
        config=config
    )

    future_df = gk_result.timeseries.make_future_dataframe(
        periods=len(test),
        include_history=False
    )

    sk_fcst = gk_result.model.predict(future_df.merge(test))
    y_pred = sk_fcst['forecast'].rename("Silverkite")

    return gk_result, y_pred, None



# modelos.py (extensión)


def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

def fit_predict_eval_lstm(training_set, test_set, model_params=None):
    """
    Entrena y evalúa un modelo LSTM con regresores.
    """
    default_params = {
        'window_size': 12,
        'units': 50,
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    if model_params:
        default_params.update(model_params)
    p = default_params

    features = training_set.drop(columns=['ds', 'y']).columns
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_x.fit_transform(training_set[features])
    y_train = scaler_y.fit_transform(training_set[['y']])
    X_test = scaler_x.transform(test_set[features])
    y_test = scaler_y.transform(test_set[['y']])

    X_seq, y_seq = create_sequences(X_train, y_train, p['window_size'])

    model = Sequential([
        LSTM(p['units'], activation='tanh', input_shape=(p['window_size'], X_seq.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=p['learning_rate']), loss='mse')
    model.fit(X_seq, y_seq, epochs=p['epochs'], batch_size=p['batch_size'], verbose=0)

    # Construcción de secuencias para predicción sobre test
    X_full = np.vstack([X_train, X_test])
    X_pred_seq = np.array([
        X_full[i - p['window_size']:i] for i in range(len(X_train), len(X_full))
    ])

    y_pred_scaled = model.predict(X_pred_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    return model, pd.Series(y_pred, index=test_set.index, name='LSTM'), (scaler_x, scaler_y)



def fit_predict_eval_gru(training_set, test_set, model_params=None):
    """
    Entrena y evalúa un modelo GRU multivariado sobre datos de series de tiempo.
    """
    default_params = {
        'window_size': 12,
        'units': 50,
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    if model_params:
        default_params.update(model_params)
    p = default_params

    features = training_set.drop(columns=['ds', 'y']).columns
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_x.fit_transform(training_set[features])
    y_train = scaler_y.fit_transform(training_set[['y']])
    X_test = scaler_x.transform(test_set[features])
    y_test = scaler_y.transform(test_set[['y']])

    X_seq, y_seq = create_sequences(X_train, y_train, p['window_size'])

    model = Sequential([
        GRU(p['units'], activation='tanh', input_shape=(p['window_size'], X_seq.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=p['learning_rate']), loss='mse')
    model.fit(X_seq, y_seq, epochs=p['epochs'], batch_size=p['batch_size'], verbose=0)

    # Predicción secuencial
    X_full = np.vstack([X_train, X_test])
    X_pred_seq = np.array([
        X_full[i - p['window_size']:i]
        for i in range(len(training_set), len(training_set) + len(test_set))
    ])

    y_pred_scaled = model.predict(X_pred_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    return model, pd.Series(y_pred, index=test_set.index, name='GRU'), (scaler_x, scaler_y)




def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    attn = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    attn = Dropout(dropout)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(attn + inputs)

    dense = Dense(ff_dim, activation='relu')(out1)
    dense = Dense(inputs.shape[-1])(dense)
    dense = Dropout(dropout)(dense)
    return LayerNormalization(epsilon=1e-6)(dense + out1)

def fit_predict_eval_transformer(training_set, test_set, model_params=None):
    p = {
        'window_size': 12,
        'head_size': 32,
        'num_heads': 2,
        'ff_dim': 64,
        'dropout': 0.1,
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    if model_params:
        p.update(model_params)

    features = training_set.drop(columns=['ds', 'y']).columns
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_x.fit_transform(training_set[features])
    y_train = scaler_y.fit_transform(training_set[['y']])
    X_test = scaler_x.transform(test_set[features])
    y_test = scaler_y.transform(test_set[['y']])

    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X_train, y_train, p['window_size'])

    inp = Input(shape=(p['window_size'], X_seq.shape[2]))
    x = transformer_encoder(inp, p['head_size'], p['num_heads'], p['ff_dim'], p['dropout'])
    x = Dense(1)(x[:, -1, :])  # solo salida final
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="mse", optimizer=Adam(learning_rate=p['learning_rate']))
    model.fit(X_seq, y_seq, batch_size=p['batch_size'], epochs=p['epochs'], verbose=0)

    # Generar secuencias de test
    X_full = np.vstack([X_train, X_test])
    X_pred_seq = np.array([
        X_full[i - p['window_size']:i]
        for i in range(len(training_set), len(training_set) + len(test_set))
    ])

    y_pred_scaled = model.predict(X_pred_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    return model, pd.Series(y_pred, index=test_set.index, name='Transformer'), (scaler_x, scaler_y)




def fit_predict_eval_elastic_net(train_data, test_data, model_params=None):
    date_col = 'ds'
    target_col = 'y'

    # Parámetros por defecto
    alpha = model_params.get('alpha', 1.0) if model_params else 1.0
    l1_ratio = model_params.get('l1_ratio', 0.5) if model_params else 0.5

    # Preparación de features y target
    X_train = train_data.drop(columns=[date_col, target_col])
    y_train = train_data[target_col].values
    X_test = test_data.drop(columns=[date_col, target_col])
    y_test = test_data[target_col].values

    # Escalamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento modelo
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    model.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = model.predict(X_test_scaled)
    forecast_test_df = pd.DataFrame({
        date_col: test_data[date_col].values,
        'pred_test': y_pred
    })

    return model, forecast_test_df, scaler

def load_chile_holidays():
    holidays = pd.DataFrame({
        'holiday': 'chile_holiday',
        'ds': pd.to_datetime([
            '2023-01-01',  # Año Nuevo
            '2023-05-01',  # Día del Trabajador
            '2023-09-18',  # Fiestas Patrias
            '2023-09-19',  # Glorias del Ejército
            '2023-12-25',  # Navidad
        ]),
        'lower_window': 0,
        'upper_window': 1,
    })
    return holidays

def fit_predict_eval_prophet(training_set, test_set, model_params=None):
    """Entrena un modelo Prophet y realiza predicciones sin fallos de holidays."""
    default_params = dict(
        growth="linear",
        seasonality_prior_scale=0.01,
        seasonality_mode="multiplicative",
        holidays_prior_scale=5,
        changepoint_prior_scale=0.01,
        n_changepoints=25
    )
    
    if model_params:
        default_params.update(model_params)

    model = Prophet(**default_params)

    # Usar feriados manuales de Chile (evita error con 'language')
    model.holidays = load_chile_holidays()

    # Añadir regresores exógenos
    for col in training_set.columns.difference(['ds', 'y']):
        model.add_regressor(col)

    model.fit(training_set)

    future = model.make_future_dataframe(periods=test_set.shape[0], freq='MS')
    full_data = pd.concat([training_set, test_set])
    future = future.merge(full_data, on='ds', how='left')

    forecast = model.predict(future.dropna())
    predictions = forecast.tail(test_set.shape[0])['yhat'].rename('Prophet')

    return model, predictions, None




def fit_predict_eval_mlp(train_data, test_data, model_params=None):
    """Entrena un MLPRegressor y predice sobre test set."""
    date_col = 'ds'
    target_col = 'y'

    # Default hyperparameters
    params = {
        'hidden_layer_sizes': (64, 32),
        'max_iter': 500
    }
    if model_params:
        params.update(model_params)

    # Preparación de datos
    X_train = train_data.drop(columns=[date_col, target_col])
    y_train = train_data[target_col].values
    X_test = test_data.drop(columns=[date_col, target_col])
    y_test = test_data[target_col].values

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        max_iter=params['max_iter'],
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    forecast_test_df = pd.DataFrame({
        date_col: test_data[date_col].values,
        'pred_test': y_pred
    })

    return model, forecast_test_df, scaler_X




def fit_predict_eval_tree(train_data, test_data, model_params=None):
    date_col = 'ds'
    target_col = 'y'

    if model_params is None:
        model_params = {
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

    X_train = train_data.drop(columns=[date_col, target_col])
    y_train = train_data[target_col].values
    X_test = test_data.drop(columns=[date_col, target_col])
    y_test = test_data[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeRegressor(
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    forecast_test_df = pd.DataFrame({
        date_col: test_data[date_col].values,
        'pred_test': y_pred
    })

    return model, forecast_test_df, scaler



def fit_predict_eval_rf(train_data, test_data, model_params=None):
    date_col = 'ds'
    target_col = 'y'

    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

    X_train = train_data.drop(columns=[date_col, target_col])
    y_train = train_data[target_col].values
    X_test = test_data.drop(columns=[date_col, target_col])
    y_test = test_data[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    forecast_test_df = pd.DataFrame({
        date_col: test_data[date_col].values,
        'pred_test': y_pred
    })

    return model, forecast_test_df, scaler
