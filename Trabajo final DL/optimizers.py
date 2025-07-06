# optimizadores.py
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from models import fit_predict_eval_hw
from models import fit_predict_eval_autoarima
from models import fit_predict_eval_silverkite
from models import fit_predict_eval_lstm
from models import fit_predict_eval_gru
from models import fit_predict_eval_transformer
from models import fit_predict_eval_elastic_net
from models import fit_predict_eval_prophet
from models import fit_predict_eval_mlp
from models import fit_predict_eval_tree
from models import fit_predict_eval_rf
import logging
import optuna

# Silencia cmdstanpy
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Opcional: también silencia Prophet si deseas
logging.getLogger("prophet").setLevel(logging.WARNING)

# Opcional: silencia Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hw_cv(df, split, n_trials=50):
    """
    Realiza validación cruzada y optimización de parámetros para Holt-Winters.
    """
    model_dict = {}

    def objective(trial):
        # Espacio de búsqueda
        model_params = {
            'trend': trial.suggest_categorical('trend', ['add', 'mul', None]),
            'seasonal': trial.suggest_categorical('seasonal', ['add', 'mul', None]),
            'seasonal_periods': trial.suggest_int('seasonal_periods', 2, 24),
            'use_boxcox': False
        }

        mape_scores = []
        best_mape = np.inf
        best_model = None

        try:
            for train_index, test_index in split:
                training_set = df.iloc[train_index].fillna(0)
                test_set = df.iloc[test_index].fillna(0)
                model, y_pred, _ = fit_predict_eval_hw(training_set, test_set, model_params)
                if hasattr(model, 'mle_retvals') and not model.mle_retvals.get('converged', True):
                    raise RuntimeError("Holt-Winters no convergió con estos parámetros")
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)

                if mape < best_mape:
                    best_mape = mape
                    best_model = model

            trial_number = trial.number
            model_dict[trial_number] = {
                'model': best_model,
                'params': model_params,
                'mape_best': best_mape
            }

            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] HW Trial {trial.number} → {model_params} | {e}")
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Consolidar resultados
    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])
    best_params = study.best_params
    return best_params, trials_df, model_dict




def optimize_autoarima_cv(df, split, n_trials=10):
    """
    Optimización de AutoARIMA con validación cruzada.
    """
    model_dict = {}

    def objective(trial):
        model_params = {
            'seasonal': trial.suggest_categorical('seasonal', [True, False]),
            'm': trial.suggest_categorical('m', [1, 3, 6, 12]),
            'stepwise': trial.suggest_categorical('stepwise', [True, False]),
            'suppress_warnings': True,
            'error_action': 'ignore'
        }

        if not model_params['seasonal']:
            model_params['m'] = 1  # Coherencia

        mape_scores = []
        best_mape = np.inf

        try:
            for train_index, test_index in split:
                training_set = df.iloc[train_index]
                test_set = df.iloc[test_index]

                _, y_pred, _ = fit_predict_eval_autoarima(training_set, test_set, model_params)
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)

                if mape < best_mape:
                    best_mape = mape

            trial_number = trial.number
            model_dict[trial_number] = {
                'params': model_params,
                'mape_best': best_mape
            }

            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] AutoARIMA Trial {trial.number} → {model_params} | {e}")
            return np.inf

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])
    best_params = study.best_params
    return best_params, trials_df, model_dict




def optimize_silverkite_cv(df, split, n_trials=50):
    """
    Optimiza hiperparámetros de Silverkite con Optuna y validación cruzada.
    """
    model_dict = {}

    def objective(trial):
        model_params = {
            'growth': trial.suggest_categorical('growth', ['linear', 'quadratic', 'sqrt', None]),
            'fit_algorithm': trial.suggest_categorical('fit_algorithm', ['ridge', 'linear']),
            'regularization_strength': trial.suggest_float('regularization_strength', 0.001, 10.0, log=True)
        }

        mape_scores = []
        best_mape = np.inf
        best_model = None

        try:
            for train_idx, test_idx in split:
                training_set = df.iloc[train_idx]
                test_set = df.iloc[test_idx]
                model, y_pred, _ = fit_predict_eval_silverkite(training_set, test_set, model_params)
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)
                if mape < best_mape:
                    best_mape = mape
                    best_model = model

            trial_number = trial.number
            model_dict[trial_number] = {
                'model': best_model,
                'params': model_params,
                'mape_best': best_mape
            }

            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] Silverkite Trial {trial.number} → {model_params} | {e}")
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])
    best_params = study.best_params
    return best_params, trials_df, model_dict




def optimize_lstm_cv(df, split, n_trials=10):
    """
    Optimiza hiperparámetros de LSTM multivariado con validación cruzada.
    """
    model_dict = {}

    def objective(trial):
        model_params = {
            'window_size': trial.suggest_int('window_size', 6, 24),
            'units': trial.suggest_int('units', 16, 128),
            'epochs': trial.suggest_int('epochs', 20, 100),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }

        mape_scores = []
        best_mape = np.inf

        try:
            for train_index, test_index in split:
                training_set = df.iloc[train_index]
                test_set = df.iloc[test_index]
                _, y_pred, _ = fit_predict_eval_lstm(training_set, test_set, model_params)
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)
                best_mape = min(best_mape, mape)
            model_dict[trial.number] = {
                'params': model_params,
                'mape_best': best_mape
            }
            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] LSTM Trial {trial.number} → {model_params} | {e}")
            return np.inf

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])
    best_params = study.best_params
    return best_params, trials_df, model_dict



def optimize_gru_cv(df, split, n_trials=10):
    """
    Optimiza hiperparámetros de un modelo GRU multivariado usando validación cruzada.
    """
    model_dict = {}

    def objective(trial):
        model_params = {
            'window_size': trial.suggest_int('window_size', 6, 24),
            'units': trial.suggest_int('units', 16, 128),
            'epochs': trial.suggest_int('epochs', 20, 100),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }

        mape_scores = []
        best_mape = np.inf

        try:
            for train_index, test_index in split:
                training_set = df.iloc[train_index]
                test_set = df.iloc[test_index]

                _, y_pred, _ = fit_predict_eval_gru(training_set, test_set, model_params)
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)
                best_mape = min(best_mape, mape)

            model_dict[trial.number] = {
                'params': model_params,
                'mape_best': best_mape
            }

            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] GRU Trial {trial.number} → {model_params} | {e}")
            return np.inf

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(
        trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best']
    )

    return study.best_params, trials_df, model_dict



def optimize_transformer_cv(df, split, n_trials=10):
    model_dict = {}

    def objective(trial):
        model_params = {
            'window_size': trial.suggest_int('window_size', 6, 24),
            'head_size': trial.suggest_int('head_size', 16, 64),
            'num_heads': trial.suggest_int('num_heads', 1, 4),
            'ff_dim': trial.suggest_int('ff_dim', 32, 128),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'epochs': trial.suggest_int('epochs', 20, 100),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }

        mape_scores = []
        best_mape = np.inf

        try:
            for train_index, test_index in split:
                training_set = df.iloc[train_index]
                test_set = df.iloc[test_index]
                _, y_pred, _ = fit_predict_eval_transformer(training_set, test_set, model_params)
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)
                best_mape = min(best_mape, mape)

            model_dict[trial.number] = {
                'params': model_params,
                'mape_best': best_mape
            }

            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] Transformer Trial {trial.number} → {model_params} | {e}")
            return np.inf

    import optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(
        trials_data,
        columns=['trial_number', 'params', 'mape_mean', 'mape_best']
    )

    return study.best_params, trials_df, model_dict



def optimize_elastic_net_cv(df, split, n_trials=50):
    model_dict = {}

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 1e1, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

        model_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        mape_scores = []
        best_mape = np.inf
        best_model = None
        best_model_params = None

        try:
            for train_indices, test_indices in split:
                train_data = df.iloc[train_indices].copy()
                test_data = df.iloc[test_indices].copy()
                model, forecast_test_df, _ = fit_predict_eval_elastic_net(train_data, test_data, model_params)

                y_true = test_data['y'].values
                y_pred = forecast_test_df['pred_test'].values
                mape = mean_absolute_percentage_error(y_true, y_pred)
                mape_scores.append(mape)

                if mape < best_mape:
                    best_mape = mape
                    best_model = model
                    best_model_params = model_params

            model_dict[trial.number] = {
                'model': best_model,
                'params': best_model_params,
                'mape_best': best_mape
            }

            return np.mean(mape_scores)

        except Exception as e:
            print(f"[ERROR] ElasticNet Trial {trial.number} → {model_params} | {e}")
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(
        trials_data,
        columns=['trial_number', 'params', 'mape_mean', 'mape_best']
    )

    return study.best_params, trials_df, model_dict




def optimize_prophet_cv(df, split, n_trials=10):
    """Optimiza hiperparámetros de Prophet con Optuna."""
    model_dict = {}

    def objective(trial):
        params = {
            'growth': trial.suggest_categorical('growth', ['linear', 'flat']),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 30, log=True),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 5, 20, log=True),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.01, 2),
            'n_changepoints': trial.suggest_int('n_changepoints', 25, 50)
        }

        mape_scores = []
        best_mape = float('inf')
        best_model = None

        try:
            for train_idx, test_idx in split:
                train_set = df.iloc[train_idx]
                test_set = df.iloc[test_idx]

                _, y_pred, _ = fit_predict_eval_prophet(train_set, test_set, params)
                mape = mean_absolute_percentage_error(test_set['y'], y_pred)
                mape_scores.append(mape)

                if mape < best_mape:
                    best_mape = mape
                    best_model = params

            model_dict[trial.number] = {
                'params': best_model,
                'mape_best': best_mape
            }

            return sum(mape_scores) / len(mape_scores)

        except Exception as e:
            print(f"Error en el ensayo con parámetros {params}: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    trials_df = pd.DataFrame([
        (t.number, t.params, t.value, model_dict.get(t.number, {}).get('mape_best', float('inf')))
        for t in study.trials
    ], columns=['trial_number', 'params', 'mape_mean', 'mape_best'])

    return study.best_params, trials_df, model_dict



def optimize_mlp_cv(df, split, n_trials=50):
    """Optimiza hiperparámetros del MLP usando validación cruzada temporal."""
    model_dict = {}

    def objective(trial):
        model_params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64, 32), (128, 64), (256, 128)]),
            'max_iter': trial.suggest_int('max_iter', 200, 1000, step=100)
        }

        mape_scores = []
        best_mape = np.inf
        best_model = None

        for train_idx, test_idx in split:
            train_data = df.iloc[train_idx].copy()
            test_data = df.iloc[test_idx].copy()

            model, forecast_df, _ = fit_predict_eval_mlp(train_data, test_data, model_params)
            mape = mean_absolute_percentage_error(test_data['y'].values, forecast_df['pred_test'].values)
            mape_scores.append(mape)

            if mape < best_mape:
                best_mape = mape
                best_model = model

        model_dict[trial.number] = {
            'model': best_model,
            'params': model_params,
            'mape_best': best_mape
        }

        return np.mean(mape_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Compilación de resultados
    trials_data = [
        (
            trial.number,
            trial.params,
            trial.value,
            model_dict.get(trial.number, {}).get('mape_best', np.inf)
        )
        for trial in study.trials
    ]

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])
    best_params = study.best_params
    return best_params, trials_df, model_dict



def optimize_tree_cv(df, split, n_trials=50):
    model_dict = {}

    def objective(trial):
        model_params = {
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }

        mape_scores = []
        best_mape = np.inf

        for train_indices, test_indices in split:
            train_data = df.iloc[train_indices].copy()
            test_data = df.iloc[test_indices].copy()
            model, forecast_test_df, _ = fit_predict_eval_tree(train_data, test_data, model_params)

            y_true = test_data['y'].values
            y_pred = forecast_test_df['pred_test'].values
            mape = mean_absolute_percentage_error(y_true, y_pred)
            mape_scores.append(mape)

            if mape < best_mape:
                best_mape = mape

        trial_number = trial.number
        model_dict[trial_number] = {
            'params': model_params,
            'mape_best': best_mape
        }

        return np.mean(mape_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    trials_data = []
    for trial in study.trials:
        trial_number = trial.number
        params = trial.params
        mape_mean = trial.value
        mape_best = model_dict.get(trial_number, {}).get('mape_best', np.inf)
        trials_data.append((trial_number, params, mape_mean, mape_best))

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])

    return study.best_params, trials_df, model_dict



def optimize_rf_cv(df, split, n_trials=50):
    model_dict = {}

    def objective(trial):
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }

        mape_scores = []
        best_mape = np.inf

        for train_indices, test_indices in split:
            train_data = df.iloc[train_indices].copy()
            test_data = df.iloc[test_indices].copy()
            model, forecast_test_df, _ = fit_predict_eval_rf(train_data, test_data, model_params)

            y_true = test_data['y'].values
            y_pred = forecast_test_df['pred_test'].values
            mape = mean_absolute_percentage_error(y_true, y_pred)
            mape_scores.append(mape)

            if mape < best_mape:
                best_mape = mape

        trial_number = trial.number
        model_dict[trial_number] = {
            'params': model_params,
            'mape_best': best_mape
        }

        return np.mean(mape_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    trials_data = []
    for trial in study.trials:
        trial_number = trial.number
        params = trial.params
        mape_mean = trial.value
        mape_best = model_dict.get(trial_number, {}).get('mape_best', np.inf)
        trials_data.append((trial_number, params, mape_mean, mape_best))

    trials_df = pd.DataFrame(trials_data, columns=['trial_number', 'params', 'mape_mean', 'mape_best'])

    return study.best_params, trials_df, model_dict