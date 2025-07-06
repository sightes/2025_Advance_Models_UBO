import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy

def format_as_year_month(df):
    df['ds'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str))
    df.set_index('ds', inplace=True)
    return df.asfreq('MS')
def detect_outlier(series, threshold=3.5):
    z = (series - series.mean()) / series.std()
    return series[abs(z) > threshold]
def limpiar_outliers_x_agrupacion(df, group_col, target):
    df_clean = df.copy()
    all_outliers = []
    for group in df[group_col].unique():
        sub_df = df[df[group_col] == group]
        idx_outliers = detect_outlier(sub_df[target]).index
        if len(idx_outliers):
            mean_vals = sub_df.drop(idx_outliers).mean()
            df_clean.loc[idx_outliers, target] = mean_vals[target]
            all_outliers += list(idx_outliers)
    return df_clean, all_outliers
def eliminar_sufijo_lag(col):
    return col.split('_lag')[0]
def feature_selection(df_target, exog, max_lag, min_lag, target_col, ignore_col, negatives_reg_col):
    df = df_target.drop(['year', 'month'], axis=1).join(exog, how='inner')
    results = []
    variables = [col for col in df.columns if col not in [target_col] + ignore_col]

    for var in variables:
        corr = sm.tsa.stattools.ccf(df[target_col], df[var], adjusted=False)
        lags = range(min_lag, max_lag + 1) if var in negatives_reg_col else range(0, max_lag + 1)

        for lag in lags:
            idx = abs(lag)
            if idx < len(corr):
                results.append({'variable': var, 'lag': lag, 'correlación': corr[idx]})
    
    return pd.DataFrame(results)
def clean_focus_correlation(df, group, focus):
    return df.loc[df.groupby(group)[focus].idxmax()].reset_index(drop=True)
def collinearity_analysis(df_target, exog, feat_select, target_col, threshold=0.7):
    df = df_target.drop(['year', 'month'], axis=1).join(exog, how='inner')
    df_lags = pd.DataFrame()
    for _, row in feat_select.iterrows():
        df_lags[f"{row['variable']}_lag{row['lag']}"] = df[row['variable']].shift(row['lag'])
    df_lags.dropna(inplace=True)
    corr_matrix = df_lags.corr()
    grupos_colineales, revisadas, seleccionadas, resumen = [], set(), [], []
    df_lags[target_col] = df.loc[df_lags.index, target_col]
    for col in corr_matrix.columns:
        if col not in revisadas:
            grupo = corr_matrix[abs(corr_matrix[col]) > threshold].index.tolist()
            revisadas.update(grupo)
            grupos_colineales.append(grupo)
    for grupo in grupos_colineales:
        grupo_filtrado = [v for v in grupo if v not in seleccionadas]
        if not grupo_filtrado:
            continue
        correlaciones = df_lags[[target_col] + grupo_filtrado].corr()[target_col].drop(target_col)
        mejor = correlaciones.idxmax()
        seleccionadas.append(mejor)
        resumen.append([
            eliminar_sufijo_lag(mejor), mejor,
            np.round(list(correlaciones.values), 2),
            grupo_filtrado
        ])
    
    return [eliminar_sufijo_lag(col) for col in seleccionadas], resumen

def construir_dataset_familia(family_name, data_by_family, exog, feat_select, target_col='sale_amount_MM', fecha_inicio='2017-01-01', fecha_fin='2024-12-01'):
    df = data_by_family[family_name].copy()
    df['y'] = df[target_col]
    df_model = df.drop(['year', 'month'], axis=1).join(exog[[col for col in feat_select.variable]], how='inner')
    df_model.index = pd.to_datetime(df_model.index)
    df_model = df_model.loc[fecha_inicio:fecha_fin]
    for _, row in feat_select.iterrows():
        df_model[f"{row['variable']}_lag_{row['lag']}"] = df_model[row['variable']].shift(row['lag'])

    df_model = df_model[['y'] + [col for col in df_model.columns if '_lag_' in col]]
    df_model = df_model.dropna()
    df_model.columns = df_model.columns.str.replace(r'[ /-]', '_', regex=True)

    return df_model
def generar_splits(df, train_periods=72, test_periods=12, max_splits=20):
    np.random.seed(12)
    splits = [
        (list(range(i, i + train_periods)), list(range(i + train_periods, i + train_periods + test_periods)))
        for i in range(0, len(df) - train_periods - test_periods + 1)
    ]
    return splits[:max_splits]