import os

import matplotlib
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                       f_regression, mutual_info_classif,
                                       mutual_info_regression)

matplotlib.use('Agg')
import gc

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge, RidgeClassifier)
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
ACTIVE_DATAFRAMES = {}

def _session_dir(session_id: int) -> str:
    """Return (and create) per-session output directory."""
    d = os.path.join("data", f"session_{session_id}")
    os.makedirs(d, exist_ok=True)
    return d
def _get_classifiers():
    models = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "ridge_classifier": lambda: RidgeClassifier(),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=5, random_state=42
        ),
    }
    if HAS_LGBM:
        models["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=50, max_depth=6, num_leaves=31,
            verbose=-1, random_state=42, force_col_wise=True
        )
    return models


def _get_regressors():
    models = {
        "linear_regression": lambda: LinearRegression(),
        "ridge": lambda: Ridge(),
        "lasso": lambda: Lasso(),
        "decision_tree": lambda: DecisionTreeRegressor(max_depth=10, random_state=42),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=42
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=5, random_state=42
        ),
    }
    if HAS_LGBM:
        models["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=50, max_depth=6, num_leaves=31,
            verbose=-1, random_state=42, force_col_wise=True
        )
    return models


def _get_param_grids():
    grids = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        },
        "ridge_classifier": {
            "alpha": [0.1, 1.0, 10.0]
        },
        "ridge": {
            "alpha": [0.1, 1.0, 10.0]
        },
        "lasso": {
            "alpha": [0.01, 0.1, 1.0]
        },
        "linear_regression": {},
        "decision_tree": {
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5]
        },
        "random_forest": {
            "n_estimators": [30, 50],
            "max_depth": [5, 8],
            "min_samples_leaf": [2, 4]
        },
        "gradient_boosting": {
            "n_estimators": [30, 50],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        },
    }
    if HAS_LGBM:
        grids["lightgbm"] = {
            "n_estimators": [30, 50],
            "max_depth": [5, 8],
            "learning_rate": [0.05, 0.1]
        }
    return grids



def check_state(session_id: int, required_keys: list) -> dict:
    """READ: Safely fetches data."""
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found."}
    
    state = ACTIVE_DATAFRAMES[session_id]
    
    if required_keys:
        missing = [key for key in required_keys if key not in state]
        if missing:
            return {"error": f"Missing required data: {missing}."}
            
    return state

def get_df(session_id: int, df_name: str = "main"):
    try:
        return ACTIVE_DATAFRAMES[session_id][df_name]
    except KeyError:
        return None

# [x] 1. load_dataset
def load_dataset(filepath: str, session_id: int, df_name: str = "main") -> dict:
    """
    Load a CSV file into memory. CALL THIS FIRST before anything else.
    Requires: filepath to a CSV file.
    Stores: DataFrame as 'main' (or custom df_name).
    Returns: row count, column count, column names, dtypes.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return {"error": f"File not found: {filepath}"}
    except pd.errors.EmptyDataError:
        return {"error": "File is empty"}
    except Exception as e:
        return {"error": f"Failed to load: {str(e)}"}

    if session_id not in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[session_id] = {}

    ACTIVE_DATAFRAMES[session_id][df_name] = df.copy()

    return {
        "status": f"Loaded {df_name}",
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


# [x] 2. get_basic_info
def get_basic_info(session_id: int, df_name: str = "main") -> dict:
    """
    Inspect current state of a dataframe. Call anytime to see shape, types, missing values, and preview.
    Requires: load_dataset must have been called.
    Returns: shape, dtypes, missing count, missing percent, first 5 rows, summary statistics.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "head": df.head().to_dict("records"),
        "summary": df.describe().to_dict()
    }


# [x] 3. identify_target_column
def identify_target_column(session_id: int, target: str = None, df_name: str = "main") -> dict:
    """
    Find which column is the prediction target and detect if it's classification or regression.
    Auto-detects common names or accepts user-specified target.
    Requires: load_dataset must have been called.
    Stores: problem_type flag in session.
    Returns: target column name, detection method, problem type, or list of available columns if not found.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    found_target = None
    method = None

    if target and target in df.columns:
        found_target = target
        method = "user_specified"
    else:
        common_targets = [
            "target", "label", "class", "churn", "attrition",
            "price", "salary", "survived", "outcome", "y", "output"
        ]
        for name in common_targets:
            for col in df.columns:
                if col.lower().strip() == name:
                    found_target = col
                    method = "auto_detected"
                    break
            if found_target:
                break

    if found_target is None:
        return {
            "target": None,
            "method": "not_found",
            "available_columns": df.columns.tolist(),
            "message": "Could not auto-detect. Ask the user to pick from available columns."
        }

    if df[found_target].dtype == 'object' or df[found_target].dtype.name == 'category':
        problem_type = "classification"
    elif df[found_target].nunique() <= 20:
        problem_type = "classification"
    else:
        problem_type = "regression"

    ACTIVE_DATAFRAMES[session_id]["problem_type"] = problem_type

    return {
        "target": found_target,
        "method": method,
        "problem_type": problem_type,
        "target_dtype": str(df[found_target].dtype),
        "target_nunique": int(df[found_target].nunique())
    }


# [x] 4. separate_features_and_target
def separate_features_and_target(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Split dataframe into X (features) and y (target). MUST call identify_target_column first to know which column.
    Requires: load_dataset, identify_target_column.
    Stores: 'X' and 'y' separately.
    Returns: shapes of X and y.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Available: {df.columns.tolist()}"}

    y = df[target_column]
    X = df.drop(columns=target_column)

    ACTIVE_DATAFRAMES[session_id]["X"] = X
    ACTIVE_DATAFRAMES[session_id]["y"] = y

    return {
        "status": "Successfully separated into X and y.",
        "X_shape": X.shape,
        "y_shape": y.shape
    }


# [x] 5. split_data
def split_data(session_id: int, test_size: float = 0.2) -> dict:
    """
    Split X and y into training and test sets. Auto-stratifies for classification.
    Requires: separate_features_and_target must have been called.
    Stores: X_train, X_test, y_train, y_test.
    Returns: shapes of all splits, whether stratification was used.
    """
    state = check_state(session_id, ["X", "y"])
    if "error" in state:
        return state

    X = state["X"]
    y = state["y"]

    if y.dtype == 'object' or y.nunique() < 20:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=42
    )

    ACTIVE_DATAFRAMES[session_id]["X_train"] = X_train
    ACTIVE_DATAFRAMES[session_id]["X_test"] = X_test
    ACTIVE_DATAFRAMES[session_id]["y_train"] = y_train
    ACTIVE_DATAFRAMES[session_id]["y_test"] = y_test

    return {
        "status": "Split complete",
        "stratified": stratify is not None,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape
    }


# [x] 6. train_single_model
def train_single_model(session_id: int, model_type: str = "logistic_regression",
                        problem_type: str = "classification") -> dict:
    """
    Train a single model. Supports multiple algorithms and both classification and regression.
    ALL features must be numeric — if not, run encode_categorical first.
    Requires: split_data must have been called. All columns in X must be numeric.
    Accepts model_type for classification: 'logistic_regression', 'ridge_classifier', 'decision_tree',
        'random_forest', 'gradient_boosting', 'lightgbm'.
    Accepts model_type for regression: 'linear_regression', 'ridge', 'lasso', 'decision_tree',
        'random_forest', 'gradient_boosting', 'lightgbm'.
    Stores: trained_model, y_pred.
    Returns: model type, accuracy or R² depending on problem type, detailed metrics.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    if problem_type == "classification":
        registry = _get_classifiers()
    elif problem_type == "regression":
        registry = _get_regressors()
    else:
        return {"error": f"Unknown problem_type '{problem_type}'. Use 'classification' or 'regression'."}

    if model_type not in registry:
        return {"error": f"Unknown model_type '{model_type}'. Available: {list(registry.keys())}"}

    class_weight = state.get("class_weight")
    model = registry[model_type]()

    if class_weight and problem_type == "classification" and hasattr(model, 'class_weight'):
        model.set_params(class_weight=class_weight)

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

    y_pred = model.predict(X_test)

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = model
    ACTIVE_DATAFRAMES[session_id]["y_pred"] = y_pred
    ACTIVE_DATAFRAMES[session_id]["problem_type"] = problem_type

    result = {
        "status": "Training complete",
        "model": type(model).__name__,
        "model_type": model_type,
        "problem_type": problem_type
    }

    if problem_type == "classification":
        result["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        result["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
    else:
        result["r2"] = round(float(r2_score(y_test, y_pred)), 4)
        result["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        result["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

    return result


# [x] 7. generate_predictions
def generate_predictions(session_id: int, n_predictions: int = 10) -> dict:
    """
    Show a side-by-side preview of predicted vs actual values on test data.
    Requires: train_single_model must have been called.
    Returns: lists of predicted and actual values for first N rows.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    X_test = state["X_test"].head(n_predictions)
    y_test = state["y_test"].head(n_predictions)
    model = state["trained_model"]
    y_pred = model.predict(X_test)

    return {
        "status": "Generated predictions successfully",
        "predicted_y_values": y_pred.tolist(),
        "actual_y_values": y_test.tolist()
    }


# [x] 8. handle_missing_features
def handle_missing_features(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Fill all missing feature values. Median for numbers, mode for categories.
    Run this BEFORE encode_categorical or train_single_model.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with missing values filled.
    Returns: fill methods used per column, remaining null count.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    filled = {}

    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if col == target_column:
            continue
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            df[col] = df[col].fillna(fill_val)
            filled[col] = f"median ({fill_val})"

    cat_cols = df.select_dtypes(exclude='number').columns
    for col in cat_cols:
        if col == target_column:
            continue
        if df[col].isnull().sum() > 0:
            mode_series = df[col].mode()
            if len(mode_series) > 0:
                fill_val = mode_series.iloc[0]
                filled[col] = f"mode ({fill_val})"
            else:
                fill_val = "Unknown"
                filled[col] = "placeholder ('Unknown')"
            df[col] = df[col].fillna(fill_val)

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Missing values handled",
        "filled": filled,
        "remaining_nulls": int(df.isnull().sum().sum())
    }


# [x] 9. encode_categorical
def encode_categorical(session_id: int, target_column: str, df_name: str = "main", max_unique_values: int = 15) -> dict:
    """
    One-hot encode categorical columns. Label encode if too many unique values.
    Run this AFTER handle_missing_features, BEFORE split_data.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with encoded columns.
    Returns: columns one-hot encoded, columns label encoded, shape before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    prev_shape = df.shape
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if target_column in cat_cols:
        cat_cols.remove(target_column)
        le_target = LabelEncoder()
        df[target_column] = le_target.fit_transform(df[target_column].astype(str))

    onehot_cols = []
    label_cols = []

    for col in cat_cols:
        if df[col].nunique() <= max_unique_values:
            onehot_cols.append(col)
        else:
            label_cols.append(col)

    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype='int8')

    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str)) # type: ignore

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Categorical data encoded",
        "onehot_encoded": onehot_cols,
        "label_encoded": label_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }


# [x] 10. drop_missing_target_rows
def drop_missing_target_rows(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Drop rows where the target column is null. Run this BEFORE separate_features_and_target.
    Requires: load_dataset and identify_target_column.
    Stores: Overwrites dataframe with null-target rows removed.
    Returns: previous row count, current row count.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {df.columns.tolist()}"}

    previous_row_count = len(df)

    if df[target_column].isnull().sum() == 0:
        return {"status": "No missing targets found", "row_count": previous_row_count}

    df = df.dropna(subset=[target_column])
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Dropped rows successfully",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 11. drop_high_cardinality_columns
def drop_high_cardinality_columns(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.8) -> dict:
    """
    Drops categorical columns that are almost entirely unique (like IDs, Names, or Hashes).
    Run this BEFORE encode_categorical.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with high-cardinality columns removed.
    Returns: threshold used, columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if target_column in cat_cols:
        cat_cols.remove(target_column)

    dropped_cols = []
    total_rows = len(df)

    for col in cat_cols:
        unique_ratio = df[col].nunique() / total_rows
        if unique_ratio >= threshold:
            dropped_cols.append(col)

    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "High cardinality check complete",
        "threshold_used": threshold,
        "columns_dropped": dropped_cols,
        "remaining_columns": df.columns.tolist()
    }

# [x] 12. drop_columns
def drop_columns(session_id: int, columns: list, df_name: str = "main") -> dict:
    """
    Drop one or more specific columns by name.
    Run this anytime BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with specified columns removed.
    Returns: columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    df = df.drop(columns=columns)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Columns dropped",
        "columns_dropped": columns,
        "remaining_columns": df.columns.tolist()
    }

# [x] 12. drop_duplicates
def drop_duplicates(session_id: int, df_name: str = "main") -> dict:
    """
    Remove duplicate rows from the dataframe.
    Run this AFTER load_dataset, BEFORE handle_missing_features.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with duplicates removed.
    Returns: previous row count, current row count, rows removed.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    previous_row_count = len(df)
    df = df.drop_duplicates()
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Duplicates removed",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 13. detect_outliers
def detect_outliers(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Detect outliers in numeric feature columns using the IQR method.
    Run this AFTER handle_missing_features, BEFORE remove_outliers.
    Requires: load_dataset must have been called.
    Returns: outlier counts per column, bounds, total outlier rows.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    outlier_info = {}
    outlier_rows = set()

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count > 0:
            outlier_info[col] = {
                "count": count,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4)
            }
            outlier_rows.update(df[mask].index.tolist())

    return {
        "status": "Outlier detection complete",
        "outlier_info": outlier_info,
        "total_outlier_rows": len(outlier_rows),
        "total_rows": len(df)
    }


# [x] 14. remove_outliers
def remove_outliers(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Remove rows containing outliers in any numeric feature column using the IQR method.
    Run this AFTER detect_outliers, BEFORE encode_categorical.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with outlier rows removed.
    Returns: previous row count, current row count, rows removed.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    previous_row_count = len(df)
    keep_mask = pd.Series(True, index=df.index)

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        keep_mask = keep_mask & (df[col] >= lower) & (df[col] <= upper)

    df = df[keep_mask]
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Outliers removed",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 15. scale_features
def scale_features(session_id: int) -> dict:
    """
    Standardize features to zero mean and unit variance. Fits on X_train, transforms both X_train and X_test.
    Run this AFTER split_data, BEFORE train_single_model.
    Requires: split_data must have been called.
    Stores: Overwrites X_train and X_test with scaled versions. Stores scaler.
    Returns: columns scaled, means and stds from training data.
    """
    state = check_state(session_id, ["X_train", "X_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    ACTIVE_DATAFRAMES[session_id]["X_train"] = X_train_scaled
    ACTIVE_DATAFRAMES[session_id]["X_test"] = X_test_scaled
    ACTIVE_DATAFRAMES[session_id]["scaler"] = scaler

    return {
        "status": "Features scaled",
        "columns_scaled": X_train.columns.tolist(),
        "means": dict(zip(X_train.columns, scaler.mean_.round(4).tolist())),
        "stds": dict(zip(X_train.columns, scaler.scale_.round(4).tolist()))
    }


# [x] 16. compute_correlations
def compute_correlations(session_id: int, df_name: str = "main") -> dict:
    """
    Compute pairwise Pearson correlation matrix for all numeric columns.
    Run this AFTER encode_categorical.
    Requires: load_dataset must have been called.
    Stores: correlation_matrix.
    Returns: matrix shape, top 10 correlated pairs.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    num_df = df.select_dtypes(include='number')

    if num_df.empty:
        return {"error": "No numeric columns found."}

    corr = num_df.corr().round(4)
    ACTIVE_DATAFRAMES[session_id]["correlation_matrix"] = corr

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "col_1": corr.columns[i],
                "col_2": corr.columns[j],
                "correlation": float(corr.iloc[i, j])
            })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "status": "Correlations computed",
        "shape": corr.shape,
        "top_pairs": pairs[:10]
    }


# [x] 17. drop_low_variance
def drop_low_variance(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.01) -> dict:
    """
    Drop numeric feature columns with variance below threshold.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with low-variance columns removed.
    Returns: columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    dropped = []
    for col in num_cols:
        if df[col].var() < threshold:
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Low variance check complete",
        "threshold": threshold,
        "columns_dropped": dropped,
        "remaining_columns": df.columns.tolist()
    }


# [x] 18. drop_correlated
def drop_correlated(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.95) -> dict:
    """
    Drop one column from each pair of highly correlated features.
    Run this AFTER compute_correlations, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with redundant columns removed.
    Returns: columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        df = df.drop(columns=to_drop)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Correlated feature check complete",
        "threshold": threshold,
        "columns_dropped": to_drop,
        "remaining_columns": df.columns.tolist()
    }


# [x] 19. rank_features
def rank_features(session_id: int) -> dict:
    """
    Rank features by importance using trained model. Works with both linear models (coefficients)
    and tree-based models (feature importances).
    Requires: train_single_model or compare_models must have been called.
    Returns: feature rankings sorted by absolute importance, method used.
    """
    state = check_state(session_id, ["trained_model", "X_train"])
    if "error" in state:
        return state

    model = state["trained_model"]
    feature_names = state["X_train"].columns.tolist()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        method = "feature_importances"
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 0 else np.abs(model.coef_)
        method = "coefficients"
    else:
        return {"error": f"Model type '{type(model).__name__}' does not expose feature importances or coefficients."}

    rankings = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "status": "Features ranked",
        "method": method,
        "model_type": type(model).__name__,
        "rankings": [{"feature": name, "importance": round(imp, 4)} for name, imp in rankings]
    }


# [x] 20. plot_distribution
def plot_distribution(session_id: int, column: str, df_name: str = "main", output_path: str = None) -> dict:
    """
    Plot the distribution of a single column. Histogram for numeric, bar chart for categorical.
    Requires: load_dataset must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), f"plot_distribution_{column}.png")

    plt.figure(figsize=(10, 6))

    if df[column].dtype in ['object', 'category']:
        counts = df[column].value_counts()
        sns.barplot(x=counts.index, y=counts.values)
        plt.xticks(rotation=45, ha='right')
    else:
        sns.histplot(df[column].dropna(), kde=True)

    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 21. plot_correlations
def plot_correlations(session_id: int, output_path: str = None) -> dict:
    """
    Plot a heatmap of the correlation matrix.
    Requires: compute_correlations must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["correlation_matrix"])
    if "error" in state:
        return state

    corr = state["correlation_matrix"]

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_correlations.png")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 22. plot_feature_importance
def plot_feature_importance(session_id: int, top_n: int = 20, output_path: str = None) -> dict:
    """
    Plot horizontal bar chart of feature importances. Works with both linear and tree-based models.
    Requires: train_single_model or compare_models must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["trained_model", "X_train"])
    if "error" in state:
        return state

    model = state["trained_model"]
    feature_names = state["X_train"].columns.tolist()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        label = "Importance"
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 0 else np.abs(model.coef_)
        label = "Importance (|coefficient|)"
    else:
        return {"error": f"Model type '{type(model).__name__}' does not support feature importance."}

    indices = np.argsort(importances)[::-1][:top_n]

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_feature_importance.png")

    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(indices)),
        importances[indices][::-1]
    )
    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices][::-1]
    )
    plt.xlabel(label)
    plt.title(f"Feature Importance ({type(model).__name__})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 23. plot_confusion_matrix
def plot_confusion_matrix(session_id: int, output_path: str = None) -> dict:
    """
    Plot the confusion matrix heatmap from test predictions.
    Requires: train_single_model must have been called.
    Returns: path to saved plot, confusion matrix values.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_confusion_matrix.png")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "confusion_matrix": cm.tolist()
    }


# [x] 24. plot_roc
def plot_roc(session_id: int, output_path: str = None) -> dict:
    """
    Plot ROC curve with AUC score. Binary classification only.
    Requires: train_single_model must have been called. Target must be binary.
    Returns: path to saved plot, AUC score.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    model = state["trained_model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    if y_test.nunique() != 2:
        return {"error": f"ROC curve requires binary target. Found {y_test.nunique()} classes."}

    if not hasattr(model, 'predict_proba'):
        return {"error": "Model does not support probability predictions."}

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = float(auc(fpr, tpr))

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_roc.png")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "auc": round(roc_auc, 4)
    }


# [x] 25. save_predictions
def save_predictions(session_id: int, output_path: str = None) -> dict:
    """
    Save test set predictions alongside actual values to a CSV file.
    Requires: train_single_model must have been called.
    Returns: path to saved file, row count.
    """
    state = check_state(session_id, ["X_test", "y_test", "y_pred"])
    if "error" in state:
        return state

    X_test = state["X_test"]
    y_test = state["y_test"]
    y_pred = state["y_pred"]

    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = y_pred

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "predictions.csv")
    results.to_csv(output_path, index=False)

    return {
        "status": "Predictions saved",
        "path": output_path,
        "row_count": len(results)
    }


# [x] 26. save_model
def save_model(session_id: int, output_path: str = None) -> dict:
    """
    Save the trained model to disk using joblib.
    Requires: train_single_model must have been called.
    Returns: path to saved model file, model type.
    """
    state = check_state(session_id, ["trained_model"])
    if "error" in state:
        return state

    model = state["trained_model"]
    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "model.joblib")
    joblib.dump(model, output_path)

    return {
        "status": "Model saved",
        "path": output_path,
        "model_type": type(model).__name__
    }


# [x] 27. load_model
def load_model(session_id: int, model_path: str = "model.joblib") -> dict:
    """
    Load a previously saved model from disk.
    Requires: A model file must exist at the given path.
    Stores: trained_model.
    Returns: model type, load status.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[session_id] = {}

    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    model = joblib.load(model_path)
    ACTIVE_DATAFRAMES[session_id]["trained_model"] = model

    return {
        "status": "Model loaded",
        "model_type": type(model).__name__,
        "path": model_path
    }


# [x] 28. generate_report
def generate_report(session_id: int, output_path: str = None) -> dict:
    """
    Generate a text summary of the full pipeline run including data shape, model type, and metrics.
    Requires: train_single_model must have been called.
    Returns: path to saved report, report content.
    """
    state = check_state(session_id, ["trained_model", "y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]
    model = state["trained_model"]

    lines = ["=" * 60, "ML PIPELINE REPORT", "=" * 60, ""]

    if "main" in state:
        df = state["main"]
        lines.append(f"Dataset shape: {df.shape}")
        lines.append(f"Columns: {df.columns.tolist()}")
        lines.append("")

    if "X_train" in state:
        lines.append(f"X_train shape: {state['X_train'].shape}")
        lines.append(f"X_test shape:  {state['X_test'].shape}")
        lines.append("")

    lines.append(f"Model: {type(model).__name__}")
    lines.append(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append(classification_report(y_test, y_pred))

    report = "\n".join(lines)

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "report.txt")
    with open(output_path, "w") as f:
        f.write(report)

    return {
        "status": "Report generated",
        "path": output_path,
        "content": report
    }


# [x] 29. get_pipeline_state
def get_pipeline_state(session_id: int) -> dict:
    """
    Check what steps have been completed in the current session. Call anytime.
    Requires: Any session must exist.
    Returns: all stored object names, their types, and shapes where applicable.
    """
    state = check_state(session_id, [])
    if "error" in state:
        return state

    objects = {}
    for key, value in state.items():
        if isinstance(value, pd.DataFrame):
            objects[key] = {"type": "DataFrame", "shape": value.shape}
        elif isinstance(value, pd.Series):
            objects[key] = {"type": "Series", "shape": value.shape}
        elif isinstance(value, np.ndarray):
            objects[key] = {"type": "ndarray", "shape": value.shape}
        else:
            objects[key] = {"type": type(value).__name__}

    return {
        "status": "State retrieved",
        "session_id": session_id,
        "objects_stored": len(objects),
        "objects": objects
    }      
    
# [x] 31. concat_csvs
def concat_csvs(session_id: int, filepaths: list, df_name: str = "main") -> dict:
    """
    Load and concatenate multiple CSV files into a single dataframe.
    Run this instead of load_dataset when data is split across files.
    Requires: list of filepaths to CSV files.
    Stores: Combined DataFrame.
    Returns: row count per file, total shape, column names.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[session_id] = {}

    dfs = []
    file_counts = {}

    for fp in filepaths:
        try:
            df = pd.read_csv(fp)
            file_counts[fp] = len(df)
            dfs.append(df)
        except FileNotFoundError:
            return {"error": f"File not found: {fp}"}
        except Exception as e:
            return {"error": f"Failed to load {fp}: {str(e)}"}

    if not dfs:
        return {"error": "No files loaded."}

    combined = pd.concat(dfs, ignore_index=True)
    ACTIVE_DATAFRAMES[session_id][df_name] = combined

    return {
        "status": "Files concatenated",
        "files_loaded": len(dfs),
        "rows_per_file": file_counts,
        "total_shape": combined.shape,
        "columns": combined.columns.tolist()
    }


# [x] 32. cast_types
def cast_types(session_id: int, type_map: dict, df_name: str = "main") -> dict:
    """
    Cast columns to specified dtypes. Use when pandas infers wrong types.
    Run this AFTER load_dataset, BEFORE any cleaning.
    Accepts: dict like {"age": "int", "price": "float", "zip_code": "str"}.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with retyped columns.
    Returns: previous and new dtypes for each cast column.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    not_found = [col for col in type_map if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    changes = {}
    errors = {}

    for col, dtype in type_map.items():
        old_dtype = str(df[col].dtype)
        try:
            df[col] = df[col].astype(dtype)
            changes[col] = {"from": old_dtype, "to": str(df[col].dtype)}
        except (ValueError, TypeError) as e:
            errors[col] = {"from": old_dtype, "target": dtype, "error": str(e)}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Type casting complete",
        "changes": changes,
        "remaining_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    if errors:
        result["errors"] = errors

    return result


# [x] 33. rename_columns
def rename_columns(session_id: int, rename_map: dict = None, clean_all: bool = False, df_name: str = "main") -> dict:
    """
    Rename columns by map or auto-clean all names (lowercase, strip, replace spaces with underscores).
    Run this AFTER load_dataset, BEFORE any processing.
    Accepts: rename_map like {"Old Name": "new_name"} OR clean_all=True.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with renamed columns.
    Returns: mapping of old names to new names.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    old_names = df.columns.tolist()

    if rename_map:
        not_found = [col for col in rename_map if col not in df.columns]
        if not_found:
            return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}
        df = df.rename(columns=rename_map)

    elif clean_all:
        clean_map = {}
        for col in df.columns:
            new_name = col.strip().lower().replace(" ", "_").replace("-", "_")
            new_name = ''.join(c if c.isalnum() or c == '_' else '' for c in new_name)
            clean_map[col] = new_name
        df = df.rename(columns=clean_map)
        rename_map = clean_map

    else:
        return {"error": "Provide 'rename_map' (dict) or set clean_all=True."}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Columns renamed",
        "renamed": rename_map,
        "columns": df.columns.tolist()
    }


# [x] 34. filter_rows
def filter_rows(session_id: int, column: str, condition: str, value=None, df_name: str = "main") -> dict:
    """
    Filter rows based on a condition applied to a column.
    Run this anytime BEFORE separate_features_and_target.
    Accepts conditions: 'eq', 'neq', 'gt', 'gte', 'lt', 'lte', 'contains', 'not_contains', 'isin', 'notin'.
    For 'isin'/'notin', pass value as a list.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with filtered rows.
    Returns: previous row count, current row count, rows removed.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if value is None and condition not in ['notnull', 'isnull']:
        return {"error": "Provide a value for the condition."}

    previous_row_count = len(df)

    conditions = {
        "eq": lambda: df[column] == value,
        "neq": lambda: df[column] != value,
        "gt": lambda: df[column] > value,
        "gte": lambda: df[column] >= value,
        "lt": lambda: df[column] < value,
        "lte": lambda: df[column] <= value,
        "contains": lambda: df[column].astype(str).str.contains(str(value), na=False),
        "not_contains": lambda: ~df[column].astype(str).str.contains(str(value), na=False),
        "isin": lambda: df[column].isin(value),
        "notin": lambda: ~df[column].isin(value),
        "isnull": lambda: df[column].isnull(),
        "notnull": lambda: df[column].notna()
    }

    if condition not in conditions:
        return {"error": f"Unknown condition '{condition}'. Available: {list(conditions.keys())}"}

    try:
        mask = conditions[condition]()
        df = df[mask]
    except Exception as e:
        return {"error": f"Filter failed: {str(e)}"}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Rows filtered",
        "condition": f"{column} {condition} {value}",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 35. clip_values
def clip_values(session_id: int, column: str, lower: float = None, upper: float = None, df_name: str = "main") -> dict:
    """
    Clip numeric column values to specified bounds. Values outside bounds are set to the bound.
    Run this AFTER handle_missing_features as an alternative to remove_outliers.
    Requires: load_dataset must have been called. Column must be numeric.
    Stores: Overwrites dataframe with clipped values.
    Returns: column clipped, bounds used, values affected.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. dtype: {df[column].dtype}"}

    if lower is None and upper is None:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

    clipped_lower = int((df[column] < lower).sum()) if lower is not None else 0
    clipped_upper = int((df[column] > upper).sum()) if upper is not None else 0

    df[column] = df[column].clip(lower=lower, upper=upper)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Values clipped",
        "column": column,
        "lower_bound": lower,
        "upper_bound": upper,
        "values_clipped_lower": clipped_lower,
        "values_clipped_upper": clipped_upper,
        "total_clipped": clipped_lower + clipped_upper
    }


# [x] 36. replace_values
def replace_values(session_id: int, column: str, replace_map: dict, df_name: str = "main") -> dict:
    """
    Replace specific values in a column.
    Run this anytime BEFORE encode_categorical.
    Accepts: replace_map like {"old_value": "new_value", "typo": "correct"}.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with replaced values.
    Returns: column name, replacements made, value counts after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    counts_before = {}
    for old_val in replace_map:
        counts_before[str(old_val)] = int((df[column] == old_val).sum())

    df[column] = df[column].replace(replace_map)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Values replaced",
        "column": column,
        "replacements": replace_map,
        "occurrences_found": counts_before,
        "unique_values_after": int(df[column].nunique())
    }


# [x] 37. strip_whitespace
def strip_whitespace(session_id: int, df_name: str = "main") -> dict:
    """
    Strip leading/trailing whitespace from all string columns and column names.
    Run this AFTER load_dataset, BEFORE any processing.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with cleaned strings.
    Returns: columns cleaned, column names cleaned.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    old_names = df.columns.tolist()
    df.columns = df.columns.str.strip()
    names_changed = [
        {"from": old, "to": new}
        for old, new in zip(old_names, df.columns)
        if old != new
    ]

    str_cols = df.select_dtypes(include='object').columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Whitespace stripped",
        "column_names_cleaned": names_changed,
        "string_columns_stripped": str_cols
    }


# [x] 38. bin_continuous
def bin_continuous(session_id: int, column: str, n_bins: int = 5, strategy: str = "quantile", labels: list = None, df_name: str = "main") -> dict:
    """
    Bin a continuous column into discrete intervals.
    Run this AFTER handle_missing_features, BEFORE encode_categorical.
    Accepts strategy: 'quantile' (equal frequency) or 'uniform' (equal width).
    Requires: load_dataset must have been called. Column must be numeric.
    Stores: Overwrites column with binned values. Original column saved as '{column}_original'.
    Returns: bin edges, value counts per bin.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. dtype: {df[column].dtype}"}

    df[f"{column}_original"] = df[column].copy()

    try:
        if strategy == "quantile":
            df[column], bin_edges = pd.qcut(df[column], q=n_bins, labels=labels, retbins=True, duplicates='drop')
        elif strategy == "uniform":
            df[column], bin_edges = pd.cut(df[column], bins=n_bins, labels=labels, retbins=True)
        else:
            return {"error": f"Unknown strategy '{strategy}'. Use 'quantile' or 'uniform'."}
    except Exception as e:
        return {"error": f"Binning failed: {str(e)}"}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Column binned",
        "column": column,
        "strategy": strategy,
        "n_bins": n_bins,
        "bin_edges": [round(float(e), 4) for e in bin_edges],
        "value_counts": df[column].value_counts().sort_index().to_dict()
    }


# [x] 39. log_transform
def log_transform(session_id: int, columns: list = None, target_column: str = None, df_name: str = "main") -> dict:
    """
    Apply log1p transform to skewed numeric columns. Auto-detects skewed columns if none specified.
    Run this AFTER handle_missing_features, BEFORE scale_features.
    Requires: load_dataset must have been called. Columns must be numeric and non-negative.
    Stores: Overwrites dataframe with transformed values.
    Returns: columns transformed, skewness before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if columns is None:
        columns = []
        for col in num_cols:
            skew = abs(df[col].skew())
            if skew > 1.0 and (df[col] >= 0).all():
                columns.append(col)
        if not columns:
            return {"status": "No skewed columns found", "skipped": True}

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    transforms = {}
    skipped = []

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            skipped.append({"column": col, "reason": "not numeric"})
            continue
        if (df[col] < 0).any():
            skipped.append({"column": col, "reason": "contains negative values"})
            continue

        skew_before = round(float(df[col].skew()), 4)
        df[col] = np.log1p(df[col])
        skew_after = round(float(df[col].skew()), 4)

        transforms[col] = {
            "skew_before": skew_before,
            "skew_after": skew_after
        }

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Log transform complete",
        "columns_transformed": list(transforms.keys()),
        "transform_details": transforms
    }
    if skipped:
        result["skipped"] = skipped

    return result


# [x] 40. extract_datetime_parts
def extract_datetime_parts(session_id: int, column: str, parts: list = None, drop_original: bool = True, df_name: str = "main") -> dict:
    """
    Parse a datetime column and extract year, month, day, dayofweek, hour as separate features.
    Run this AFTER load_dataset, BEFORE encode_categorical.
    Accepts parts: any subset of ['year', 'month', 'day', 'dayofweek', 'hour', 'minute', 'quarter', 'is_weekend'].
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with new datetime feature columns.
    Returns: new columns created, rows that failed to parse.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    try:
        dt_series = pd.to_datetime(df[column], infer_datetime_format=True, errors='coerce')
    except Exception as e:
        return {"error": f"Failed to parse datetime: {str(e)}"}

    failed_count = int(dt_series.isnull().sum() - df[column].isnull().sum())

    if parts is None:
        parts = ['year', 'month', 'day', 'dayofweek']

    extractors = {
        'year': lambda dt: dt.dt.year,
        'month': lambda dt: dt.dt.month,
        'day': lambda dt: dt.dt.day,
        'dayofweek': lambda dt: dt.dt.dayofweek,
        'hour': lambda dt: dt.dt.hour,
        'minute': lambda dt: dt.dt.minute,
        'quarter': lambda dt: dt.dt.quarter,
        'is_weekend': lambda dt: dt.dt.dayofweek.isin([5, 6]).astype(int)
    }

    invalid_parts = [p for p in parts if p not in extractors]
    if invalid_parts:
        return {"error": f"Invalid parts: {invalid_parts}. Available: {list(extractors.keys())}"}

    new_cols = []
    for part in parts:
        col_name = f"{column}_{part}"
        df[col_name] = extractors[part](dt_series)
        new_cols.append(col_name)

    if drop_original:
        df = df.drop(columns=[column])

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Datetime features extracted",
        "original_column": column,
        "new_columns": new_cols,
        "original_dropped": drop_original,
        "parse_failures": failed_count
    }


# [x] 41. create_interactions
def create_interactions(session_id: int, column_pairs: list = None, target_column: str = None, df_name: str = "main") -> dict:
    """
    Create interaction features by multiplying pairs of numeric columns.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Accepts: list of tuples like [("col_a", "col_b"), ("col_c", "col_d")].
    If None, creates interactions for top 5 correlated pairs with target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with new interaction columns.
    Returns: new columns created, shapes before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    prev_shape = df.shape

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if column_pairs is None:
        if target_column and target_column in df.columns:
            corrs = df[num_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
            top_cols = corrs.head(5).index.tolist()
        else:
            top_cols = num_cols[:5]

        column_pairs = []
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                column_pairs.append((top_cols[i], top_cols[j]))

    new_cols = []
    skipped = []

    for pair in column_pairs:
        if len(pair) != 2:
            skipped.append({"pair": pair, "reason": "must be length 2"})
            continue
        col_a, col_b = pair
        if col_a not in df.columns or col_b not in df.columns:
            skipped.append({"pair": pair, "reason": "column not found"})
            continue
        if not pd.api.types.is_numeric_dtype(df[col_a]) or not pd.api.types.is_numeric_dtype(df[col_b]):
            skipped.append({"pair": pair, "reason": "not numeric"})
            continue

        col_name = f"{col_a}_x_{col_b}"
        df[col_name] = df[col_a] * df[col_b]
        new_cols.append(col_name)

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Interactions created",
        "new_columns": new_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }
    if skipped:
        result["skipped"] = skipped

    return result


# [x] 42. create_polynomials
def create_polynomials(session_id: int, columns: list = None, degree: int = 2, target_column: str = None, df_name: str = "main") -> dict:
    """
    Create polynomial features (squared, cubed, etc.) for specified numeric columns.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called. Columns must be numeric.
    Stores: Overwrites dataframe with new polynomial columns.
    Returns: new columns created, shapes before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    prev_shape = df.shape

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if columns is None:
        columns = num_cols[:10]

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    new_cols = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        for d in range(2, degree + 1):
            col_name = f"{col}_pow{d}"
            df[col_name] = df[col] ** d
            new_cols.append(col_name)

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Polynomial features created",
        "degree": degree,
        "new_columns": new_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }


# [x] 43. select_features
def select_features(session_id: int, columns: list, df_name: str = "main") -> dict:
    """
    Keep only specified columns, drop everything else. The inverse of drop_columns.
    Run this BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with only selected columns.
    Returns: columns kept, columns removed, shape.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    removed = [col for col in df.columns if col not in columns]
    df = df[columns]
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Features selected",
        "columns_kept": columns,
        "columns_removed": removed,
        "shape": df.shape
    }


# [x] 44. create_folds
def create_folds(session_id: int, n_folds: int = 5) -> dict:
    """
    Create stratified k-fold indices for cross-validation.
    Run this AFTER separate_features_and_target.
    Requires: separate_features_and_target must have been called.
    Stores: fold_indices as list of (train_idx, val_idx) tuples.
    Returns: number of folds, rows per fold.
    """
    state = check_state(session_id, ["X", "y"])
    if "error" in state:
        return state

    X = state["X"]
    y = state["y"]

    if y.dtype == 'object' or y.nunique() < 20:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
        stratified = True
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(kf.split(X))
        stratified = False

    ACTIVE_DATAFRAMES[session_id]["fold_indices"] = folds

    fold_info = []
    for i, (train_idx, val_idx) in enumerate(folds):
        fold_info.append({
            "fold": i + 1,
            "train_rows": len(train_idx),
            "val_rows": len(val_idx)
        })

    return {
        "status": "Folds created",
        "n_folds": n_folds,
        "stratified": stratified,
        "fold_details": fold_info
    }


# [x] 45. cross_validate
def cross_validate_model(session_id: int, model_type: str = "logistic_regression",
                          problem_type: str = "classification", n_folds: int = 5) -> dict:
    """
    Run stratified k-fold cross-validation on the full X and y.
    Run this AFTER separate_features_and_target. Alternative to split_data + train_single_model.
    Requires: separate_features_and_target must have been called. All features must be numeric.
    Returns: per-fold score, mean score, std.
    """
    state = check_state(session_id, ["X", "y"])
    if "error" in state:
        return state

    X = state["X"]
    y = state["y"]

    non_numeric = X.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    if problem_type == "classification":
        registry = _get_classifiers()
        scoring = "accuracy"
        metric_name = "accuracy"
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    elif problem_type == "regression":
        registry = _get_regressors()
        scoring = "r2"
        metric_name = "r2"
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        return {"error": f"Unknown problem_type '{problem_type}'. Use 'classification' or 'regression'."}

    if model_type not in registry:
        return {"error": f"Unknown model_type '{model_type}'. Available: {list(registry.keys())}"}

    model = registry[model_type]()

    class_weight = state.get("class_weight")
    if class_weight and problem_type == "classification" and hasattr(model, 'class_weight'):
        model.set_params(class_weight=class_weight)

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    except Exception as e:
        return {"error": f"Cross-validation failed: {str(e)}"}

    del model
    gc.collect()

    return {
        "status": "Cross-validation complete",
        "model_type": model_type,
        "problem_type": problem_type,
        "metric": metric_name,
        "n_folds": n_folds,
        "fold_scores": [round(float(s), 4) for s in scores],
        f"mean_{metric_name}": round(float(scores.mean()), 4),
        f"std_{metric_name}": round(float(scores.std()), 4)
    }


# [x] 46. tune_hyperparameters
def tune_hyperparameters(session_id: int, n_folds: int = 5) -> dict:
    """
    Grid search over LogisticRegression hyperparameters using cross-validation.
    Run this AFTER split_data, scale_features.
    Requires: split_data must have been called. All features must be numeric.
    Stores: Overwrites trained_model with best estimator, stores y_pred.
    Returns: best parameters, best score, all results.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    if y_train.dtype == 'object' or y_train.nunique() < 20:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True
    )

    try:
        grid.fit(X_train, y_train)
    except Exception as e:
        return {"error": f"Grid search failed: {str(e)}"}

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = best_model
    ACTIVE_DATAFRAMES[session_id]["y_pred"] = y_pred

    results = []
    for i in range(len(grid.cv_results_['params'])):
        results.append({
            "params": grid.cv_results_['params'][i],
            "mean_score": round(float(grid.cv_results_['mean_test_score'][i]), 4),
            "std_score": round(float(grid.cv_results_['std_test_score'][i]), 4)
        })
    results.sort(key=lambda x: x["mean_score"], reverse=True)

    return {
        "status": "Hyperparameter tuning complete",
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "all_results": results
    }
def tune_hyperparameters(session_id: int, model_type: str = "logistic_regression",
                          problem_type: str = "classification", n_folds: int = 3) -> dict:
    """
    Grid search over hyperparameters for the specified model using cross-validation.
    Run this AFTER split_data, scale_features.
    Requires: split_data must have been called. All features must be numeric.
    Stores: Overwrites trained_model with best estimator, stores y_pred.
    Returns: best parameters, best score, all results.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    if problem_type == "classification":
        registry = _get_classifiers()
        scoring = "accuracy"
        metric_name = "accuracy"
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    elif problem_type == "regression":
        registry = _get_regressors()
        scoring = "r2"
        metric_name = "r2"
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        return {"error": f"Unknown problem_type '{problem_type}'. Use 'classification' or 'regression'."}

    if model_type not in registry:
        return {"error": f"Unknown model_type '{model_type}'. Available: {list(registry.keys())}"}

    param_grids = _get_param_grids()

    if model_type not in param_grids or not param_grids[model_type]:
        return {"error": f"No hyperparameter grid defined for '{model_type}'. Use train_single_model instead."}

    model = registry[model_type]()

    class_weight = state.get("class_weight")
    if class_weight and problem_type == "classification" and hasattr(model, 'class_weight'):
        model.set_params(class_weight=class_weight)

    grid = GridSearchCV(
        model,
        param_grids[model_type],
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    try:
        grid.fit(X_train, y_train)
    except Exception as e:
        return {"error": f"Grid search failed: {str(e)}"}

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    del grid
    gc.collect()

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = best_model
    ACTIVE_DATAFRAMES[session_id]["y_pred"] = y_pred
    ACTIVE_DATAFRAMES[session_id]["problem_type"] = problem_type

    result = {
        "status": "Hyperparameter tuning complete",
        "model_type": model_type,
        "problem_type": problem_type,
        "best_params": best_model.get_params(),
        "best_cv_score": round(float(grid.best_score_), 4) if hasattr(grid, 'best_score_') else None
    }

    if problem_type == "classification":
        result["test_accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
    else:
        result["test_r2"] = round(float(r2_score(y_test, y_pred)), 4)
        result["test_mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        result["test_rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

    return result

# [x] 47. compare_models
def compare_models(session_id: int, problem_type: str = "classification") -> dict:
    """
    Train and compare multiple models on the same train/test split.
    Trains one at a time and frees memory between each.
    Run this AFTER split_data, scale_features.
    Requires: split_data must have been called. All features must be numeric.
    Stores: trained_model and y_pred from the best performing model.
    Returns: metrics per model, best model name.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    if problem_type == "classification":
        registry = _get_classifiers()
        metric_name = "accuracy"
    elif problem_type == "regression":
        registry = _get_regressors()
        metric_name = "r2"
    else:
        return {"error": f"Unknown problem_type '{problem_type}'. Use 'classification' or 'regression'."}

    class_weight = state.get("class_weight")

    results = []
    best_score = -float('inf')
    best_name = None
    best_model = None
    best_pred = None

    for name, builder in registry.items():
        try:
            model = builder()

            if class_weight and problem_type == "classification" and hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weight)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "classification":
                score = float(accuracy_score(y_test, y_pred))
                entry = {
                    "model": name,
                    "accuracy": round(score, 4),
                    "status": "success"
                }
            else:
                score = float(r2_score(y_test, y_pred))
                mae = float(mean_absolute_error(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                entry = {
                    "model": name,
                    "r2": round(score, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "status": "success"
                }

            results.append(entry)

            if score > best_score:
                if best_model is not None:
                    del best_model
                best_score = score
                best_name = name
                best_model = model
                best_pred = y_pred.copy()
                del y_pred
            else:
                del model
                del y_pred

            gc.collect()

        except Exception as e:
            results.append({
                "model": name,
                metric_name: None,
                "status": f"failed: {str(e)}"
            })
            gc.collect()

    results.sort(
        key=lambda x: x.get(metric_name) if x.get(metric_name) is not None else -float('inf'),
        reverse=True
    )

    if best_model is not None:
        ACTIVE_DATAFRAMES[session_id]["trained_model"] = best_model
        ACTIVE_DATAFRAMES[session_id]["y_pred"] = best_pred
        ACTIVE_DATAFRAMES[session_id]["problem_type"] = problem_type

    return {
        "status": "Model comparison complete",
        "problem_type": problem_type,
        "metric": metric_name,
        "results": results,
        "best_model": best_name,
        f"best_{metric_name}": round(best_score, 4)
    }


# [x] 49. plot_learning_curve (MODIFIED)

def plot_learning_curve(session_id: int, n_points: int = 8, output_path: str = None) -> dict:
    """
    Plot training and validation score as a function of training set size.
    Uses the same model type as the trained model in the session.
    Run this AFTER train_single_model or compare_models.
    Requires: trained_model, X_train, X_test, y_train, y_test must exist. All features must be numeric.
    Returns: path to saved plot, train/val scores at each size.
    """
    state = check_state(session_id, ["trained_model", "X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    trained_model = state["trained_model"]
    problem_type = state.get("problem_type", "classification")

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}."}

    model_class = type(trained_model)
    model_params = trained_model.get_params()

    if problem_type == "classification":
        score_fn = accuracy_score
        metric_name = "accuracy"
    else:
        score_fn = r2_score
        metric_name = "r2"

    sizes = np.linspace(0.1, 1.0, n_points)
    train_scores = []
    val_scores = []
    actual_sizes = []

    for frac in sizes:
        n = max(int(len(X_train) * frac), 2)
        X_sub = X_train.iloc[:n]
        y_sub = y_train.iloc[:n]

        if problem_type == "classification" and y_sub.nunique() < 2:
            continue

        try:
            model = model_class(**model_params)
            model.fit(X_sub, y_sub)
            train_acc = float(score_fn(y_sub, model.predict(X_sub)))
            val_acc = float(score_fn(y_test, model.predict(X_test)))
            train_scores.append(train_acc)
            val_scores.append(val_acc)
            actual_sizes.append(n)
        except Exception:
            continue
        finally:
            if 'model' in dir():
                del model
            gc.collect()

    if not train_scores:
        return {"error": "Could not generate learning curve."}

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_learning_curve.png")

    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, train_scores, 'o-', label='Train')
    plt.plot(actual_sizes, val_scores, 'o-', label='Validation')
    plt.xlabel("Training Set Size")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Learning Curve ({type(trained_model).__name__})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "metric": metric_name,
        "train_scores": [round(s, 4) for s in train_scores],
        "val_scores": [round(s, 4) for s in val_scores],
        "sizes": actual_sizes
    }


# [x] 48. compute_metrics
def compute_metrics(session_id: int) -> dict:
    """
    Compute detailed classification metrics from test predictions.
    Run this AFTER train_single_model or compare_models.
    Requires: train_single_model must have been called.
    Returns: accuracy, per-class precision/recall/f1, confusion matrix, support.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    result = {
        "status": "Metrics computed",
        "accuracy": round(acc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "total_predictions": len(y_test)
    }

    if hasattr(state.get("trained_model"), 'predict_proba') and y_test.nunique() == 2:
        model = state["trained_model"]
        X_test = state["X_test"]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        result["auc"] = round(float(auc(fpr, tpr)), 4)

    return result


# [x] 49. plot_learning_curve
def plot_learning_curve(session_id: int, n_points: int = 10, output_path: str = None) -> dict:
    """
    Plot training and validation score as a function of training set size.
    Uses the same model type as the trained model in the session.
    Run this AFTER train_single_model or compare_models.
    Requires: trained_model, X_train, X_test, y_train, y_test must exist. All features must be numeric.
    Returns: path to saved plot, train/val scores at each size.
    """
    state = check_state(session_id, ["trained_model", "X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    trained_model = state["trained_model"]
    problem_type = state.get("problem_type", "classification")

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}."}

    model_class = type(trained_model)
    model_params = trained_model.get_params()

    if problem_type == "classification":
        score_fn = accuracy_score
        metric_name = "accuracy"
    else:
        score_fn = r2_score
        metric_name = "r2"

    sizes = np.linspace(0.1, 1.0, n_points)
    train_scores = []
    val_scores = []
    actual_sizes = []

    for frac in sizes:
        n = max(int(len(X_train) * frac), 2)
        X_sub = X_train.iloc[:n]
        y_sub = y_train.iloc[:n]

        if problem_type == "classification" and y_sub.nunique() < 2:
            continue

        try:
            model = model_class(**model_params)
            model.fit(X_sub, y_sub)
            train_acc = float(score_fn(y_sub, model.predict(X_sub)))
            val_acc = float(score_fn(y_test, model.predict(X_test)))
            train_scores.append(train_acc)
            val_scores.append(val_acc)
            actual_sizes.append(n)
        except Exception:
            continue

    if not train_scores:
        return {"error": "Could not generate learning curve."}

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_learning_curve.png")

    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, train_scores, 'o-', label='Train')
    plt.plot(actual_sizes, val_scores, 'o-', label='Validation')
    plt.xlabel("Training Set Size")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Learning Curve ({type(trained_model).__name__})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "metric": metric_name,
        "train_scores": [round(s, 4) for s in train_scores],
        "val_scores": [round(s, 4) for s in val_scores],
        "sizes": actual_sizes
    }

# [x] 50. plot_predictions
def plot_predictions(session_id: int, n_samples: int = 50, output_path: str = None) -> dict:
    """
    Plot predicted vs actual values as a bar chart or scatter for visual comparison.
    Run this AFTER train_single_model.
    Requires: train_single_model must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]

    n = min(n_samples, len(y_test))
    y_actual = y_test.values[:n]
    y_predicted = y_pred[:n]
    indices = range(n)

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_predictions.png")

    fig, ax = plt.subplots(figsize=(14, 6))

    width = 0.35
    x = np.arange(n)
    ax.bar(x - width / 2, y_actual, width, label='Actual', alpha=0.8)
    ax.bar(x + width / 2, y_predicted, width, label='Predicted', alpha=0.8)

    mismatches = [i for i in range(n) if y_actual[i] != y_predicted[i]]
    for i in mismatches:
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red')

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Class")
    ax.set_title(f"Predicted vs Actual (first {n} samples, {len(mismatches)} mismatches highlighted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "samples_shown": n,
        "mismatches": len(mismatches)
    }


# [x] 51. plot_calibration
def plot_calibration(session_id: int, n_bins: int = 10, output_path: str = None) -> dict:
    """
    Plot calibration curve showing predicted probability vs actual frequency. Binary classification only.
    Run this AFTER train_single_model.
    Requires: train_single_model must have been called. Target must be binary. Model must support predict_proba.
    Returns: path to saved plot, brier-like summary.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    model = state["trained_model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    if y_test.nunique() != 2:
        return {"error": f"Calibration plot requires binary target. Found {y_test.nunique()} classes."}

    if not hasattr(model, 'predict_proba'):
        return {"error": "Model does not support probability predictions."}

    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        fraction_pos, mean_predicted = calibration_curve(y_test, y_prob, n_bins=n_bins)
    except Exception as e:
        return {"error": f"Calibration curve failed: {str(e)}"}

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_calibration.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(mean_predicted, fraction_pos, 'o-', label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(y_prob, bins=n_bins * 2, edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    mean_gap = float(np.mean(np.abs(fraction_pos - mean_predicted)))

    return {
        "status": "Plot saved",
        "path": output_path,
        "n_bins": n_bins,
        "mean_calibration_gap": round(mean_gap, 4),
        "bin_predicted": [round(float(x), 4) for x in mean_predicted],
        "bin_actual": [round(float(x), 4) for x in fraction_pos]
    }

# [x] 52. merge_datasets

def merge_datasets(session_id: int, left_name: str, right_name: str, on: str = None,
                   left_on: str = None, right_on: str = None, how: str = "inner",
                   result_name: str = "main") -> dict:
    """
    Join two dataframes on key columns. Use this instead of concat_csvs when data is relational.
    Run this AFTER loading both dataframes with load_dataset using different df_names.
    Accepts how: 'inner', 'left', 'right', 'outer'.
    Requires: Both dataframes must be loaded.
    Stores: Merged DataFrame under result_name.
    Returns: merge type, key columns, shape before and after.
    """
    state = check_state(session_id, [left_name, right_name])
    if "error" in state:
        return state

    left = state[left_name]
    right = state[right_name]

    valid_how = ["inner", "left", "right", "outer"]
    if how not in valid_how:
        return {"error": f"Invalid merge type '{how}'. Use one of: {valid_how}"}

    if on is None and left_on is None and right_on is None:
        common = list(set(left.columns) & set(right.columns))
        if not common:
            return {"error": "No common columns found. Specify 'on', or 'left_on' and 'right_on'."}
        on = common[0]

    try:
        if on:
            merged = left.merge(right, on=on, how=how)
            key_info = {"on": on}
        else:
            merged = left.merge(right, left_on=left_on, right_on=right_on, how=how)
            key_info = {"left_on": left_on, "right_on": right_on}
    except Exception as e:
        return {"error": f"Merge failed: {str(e)}"}

    ACTIVE_DATAFRAMES[session_id][result_name] = merged

    return {
        "status": "Merge complete",
        "how": how,
        "keys": key_info,
        "left_shape": left.shape,
        "right_shape": right.shape,
        "merged_shape": merged.shape,
        "columns": merged.columns.tolist()
    }


# [x] 53. save_csv

def save_csv(session_id: int, df_name: str = "main", output_path: str = None,
             include_index: bool = False) -> dict:
    """
    Export any stored dataframe to a CSV file.
    Run this anytime after load_dataset. Use to save cleaned or transformed data.
    Requires: The specified dataframe must exist in the session.
    Returns: path to saved file, row count, column count.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if not isinstance(df, pd.DataFrame):
        return {"error": f"'{df_name}' is not a DataFrame. Type: {type(df).__name__}"}

    try:
        if output_path is None:
            output_path = os.path.join(_session_dir(session_id), f"{df_name}.csv")
        df.to_csv(output_path, index=include_index)
    except Exception as e:
        return {"error": f"Failed to save: {str(e)}"}

    return {
        "status": "CSV saved",
        "path": output_path,
        "row_count": len(df),
        "column_count": len(df.columns)
    }


# [x] 54. aggregate_features

def aggregate_features(session_id: int, group_column: str, agg_columns: list,
                       agg_functions: list = None, df_name: str = "main") -> dict:
    """
    Create aggregated features via groupby. E.g., mean purchase per customer, count per category.
    Run this AFTER load_dataset, BEFORE separate_features_and_target.
    Accepts agg_functions: any subset of ['mean', 'sum', 'count', 'min', 'max', 'std', 'median'].
    If None, defaults to ['mean', 'sum', 'count'].
    Requires: load_dataset must have been called. group_column and agg_columns must exist.
    Stores: Overwrites dataframe with new aggregated columns merged in.
    Returns: new columns created, shape before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if group_column not in df.columns:
        return {"error": f"Group column '{group_column}' not found. Available: {df.columns.tolist()}"}

    not_found = [col for col in agg_columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    if agg_functions is None:
        agg_functions = ["mean", "sum", "count"]

    valid_agg = ["mean", "sum", "count", "min", "max", "std", "median"]
    invalid = [f for f in agg_functions if f not in valid_agg]
    if invalid:
        return {"error": f"Invalid aggregation functions: {invalid}. Available: {valid_agg}"}

    prev_shape = df.shape
    new_cols = []

    for col in agg_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        for func in agg_functions:
            col_name = f"{group_column}_{col}_{func}"
            agg_values = df.groupby(group_column)[col].transform(func)
            df[col_name] = agg_values
            new_cols.append(col_name)

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Aggregated features created",
        "group_column": group_column,
        "new_columns": new_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }


# [x] 55. compute_regression_metrics

def compute_regression_metrics(session_id: int) -> dict:
    """
    Compute regression metrics from test predictions. Use this instead of compute_metrics for regression.
    Run this AFTER train_single_model on a regression problem.
    Requires: train_single_model must have been called. y_test and y_pred must exist.
    Returns: MAE, MSE, RMSE, R², adjusted R², prediction summary.
    """
    state = check_state(session_id, ["y_test", "y_pred", "X_test"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]
    X_test = state["X_test"]

    n = len(y_test)
    p = X_test.shape[1]

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    if n > p + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = None

    residuals = np.array(y_test) - np.array(y_pred)

    return {
        "status": "Regression metrics computed",
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4) if adj_r2 is not None else None,
        "total_predictions": n,
        "residual_mean": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4)
    }


# [x] 56. plot_residuals

def plot_residuals(session_id: int, output_path: str = None) -> dict:
    """
    Plot residual analysis for regression: predicted vs actual scatter and residual distribution.
    Run this AFTER train_single_model on a regression problem.
    Requires: train_single_model must have been called. y_test and y_pred must exist.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = np.array(state["y_test"])
    y_pred = np.array(state["y_pred"])
    residuals = y_test - y_pred

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_residuals.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_pred, y_test, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted")

    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 57. plot_precision_recall_curve

def plot_precision_recall_curve(session_id: int, output_path: str = None) -> dict:
    """
    Plot precision-recall curve with average precision score. Binary classification only.
    Better than ROC for imbalanced datasets. Run this AFTER train_single_model.
    Requires: train_single_model must have been called. Target must be binary. Model must support predict_proba.
    Returns: path to saved plot, average precision score.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    model = state["trained_model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    if y_test.nunique() != 2:
        return {"error": f"PR curve requires binary target. Found {y_test.nunique()} classes."}

    if not hasattr(model, 'predict_proba'):
        return {"error": "Model does not support probability predictions."}

    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = float(average_precision_score(y_test, y_prob))

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_precision_recall.png")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "average_precision": round(ap, 4)
    }


# [x] 58. plot_boxplots

def plot_boxplots(session_id: int, columns: list = None, target_column: str = None,
                  max_columns: int = 12, df_name: str = "main", output_path: str = None) -> dict:
    """
    Plot boxplots for numeric columns to visualize spread and outliers.
    Run this AFTER load_dataset, anytime during exploration.
    If columns is None, auto-selects up to max_columns numeric columns.
    Requires: load_dataset must have been called.
    Returns: path to saved plot, columns plotted.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if columns is not None:
        not_found = [col for col in columns if col not in df.columns]
        if not_found:
            return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}
        num_cols = [col for col in columns if col in num_cols]

    if not num_cols:
        return {"error": "No numeric columns to plot."}

    num_cols = num_cols[:max_columns]

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_boxplots.png")

    n_cols = min(3, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i])
        axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "columns_plotted": num_cols
    }


# [x] 59. handle_class_imbalance

def handle_class_imbalance(session_id: int, strategy: str = "class_weight") -> dict:
    """
    Handle imbalanced classes in the training set. Does NOT use SMOTE (too memory-heavy).
    Run this AFTER split_data, BEFORE train_single_model.
    Accepts strategy: 'class_weight' (flag for model), 'undersample' (reduce majority),
        'oversample' (duplicate minority rows).
    Requires: split_data must have been called.
    Stores: Modified X_train and y_train for undersample/oversample. Stores class_weight flag.
    Returns: strategy used, class distribution before and after.
    """
    state = check_state(session_id, ["X_train", "y_train"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    y_train = state["y_train"]

    valid_strategies = ["class_weight", "undersample", "oversample"]
    if strategy not in valid_strategies:
        return {"error": f"Invalid strategy '{strategy}'. Use one of: {valid_strategies}"}

    dist_before = y_train.value_counts().to_dict()

    if strategy == "class_weight":
        ACTIVE_DATAFRAMES[session_id]["class_weight"] = "balanced"
        return {
            "status": "Class weight flag set",
            "strategy": "class_weight",
            "note": "Models will use class_weight='balanced' during training.",
            "class_distribution": dist_before
        }

    elif strategy == "undersample":
        min_count = y_train.value_counts().min()
        frames = []
        for cls in y_train.unique():
            cls_data = X_train[y_train == cls]
            cls_sampled = cls_data.sample(n=min_count, random_state=42)
            frames.append(cls_sampled)

        X_resampled = pd.concat(frames)
        y_resampled = y_train.loc[X_resampled.index]

        shuffle_idx = X_resampled.sample(frac=1, random_state=42).index
        X_resampled = X_resampled.loc[shuffle_idx]
        y_resampled = y_resampled.loc[shuffle_idx]

        ACTIVE_DATAFRAMES[session_id]["X_train"] = X_resampled
        ACTIVE_DATAFRAMES[session_id]["y_train"] = y_resampled

    elif strategy == "oversample":
        max_count = y_train.value_counts().max()
        frames_X = []
        frames_y = []
        for cls in y_train.unique():
            cls_X = X_train[y_train == cls]
            cls_y = y_train[y_train == cls]
            if len(cls_X) < max_count:
                extra_idx = cls_X.sample(n=max_count - len(cls_X), replace=True, random_state=42).index
                cls_X = pd.concat([cls_X, cls_X.loc[extra_idx]])
                cls_y = pd.concat([cls_y, cls_y.loc[extra_idx]])
            frames_X.append(cls_X)
            frames_y.append(cls_y)

        X_resampled = pd.concat(frames_X).reset_index(drop=True)
        y_resampled = pd.concat(frames_y).reset_index(drop=True)

        shuffle_idx = X_resampled.sample(frac=1, random_state=42).index
        X_resampled = X_resampled.loc[shuffle_idx].reset_index(drop=True)
        y_resampled = y_resampled.loc[shuffle_idx].reset_index(drop=True)

        ACTIVE_DATAFRAMES[session_id]["X_train"] = X_resampled
        ACTIVE_DATAFRAMES[session_id]["y_train"] = y_resampled

    dist_after = ACTIVE_DATAFRAMES[session_id]["y_train"].value_counts().to_dict()

    return {
        "status": "Class imbalance handled",
        "strategy": strategy,
        "distribution_before": dist_before,
        "distribution_after": dist_after,
        "previous_train_size": len(y_train),
        "current_train_size": len(ACTIVE_DATAFRAMES[session_id]["y_train"])
    }


# [x] 60. drop_high_missing_columns

def drop_high_missing_columns(session_id: int, target_column: str = None,
                               threshold: float = 0.5, df_name: str = "main") -> dict:
    """
    Drop columns where the percentage of missing values exceeds threshold.
    Run this AFTER load_dataset, BEFORE handle_missing_features.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with high-missing columns removed.
    Returns: threshold used, columns dropped with their missing percentages, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    missing_pct = df.isnull().mean()
    dropped = {}
    to_drop = []

    for col in df.columns:
        if col == target_column:
            continue
        if missing_pct[col] >= threshold:
            dropped[col] = round(float(missing_pct[col] * 100), 2)
            to_drop.append(col)

    if to_drop:
        df = df.drop(columns=to_drop)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "High-missing column check complete",
        "threshold": f"{threshold * 100}%",
        "columns_dropped": dropped,
        "remaining_columns": df.columns.tolist()
    }


# [x] 61. check_data_leakage

def check_data_leakage(session_id: int, target_column: str, threshold: float = 0.95,
                        df_name: str = "main") -> dict:
    """
    Flag features with suspiciously high correlation to the target. May indicate data leakage.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called. Target column must be numeric.
    Returns: flagged columns with their correlation to target, recommendation.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return {"error": f"Target column '{target_column}' is not numeric. Encode it first."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    if not num_cols:
        return {"error": "No numeric feature columns to check."}

    flagged = {}
    safe = {}

    for col in num_cols:
        try:
            corr = abs(float(df[col].corr(df[target_column])))
        except Exception:
            continue

        if corr >= threshold:
            flagged[col] = round(corr, 4)
        else:
            safe[col] = round(corr, 4)

    top_safe = dict(sorted(safe.items(), key=lambda x: x[1], reverse=True)[:10])

    result = {
        "status": "Leakage check complete",
        "threshold": threshold,
        "flagged_columns": flagged,
        "top_10_safe_correlations": top_safe,
        "total_features_checked": len(num_cols)
    }

    if flagged:
        result["recommendation"] = (
            f"Found {len(flagged)} suspicious column(s). "
            "These may contain information derived from the target. "
            "Consider dropping them with drop_columns."
        )
    else:
        result["recommendation"] = "No leakage detected."

    return result


# [x] 62. select_k_best_features

def select_k_best_features(session_id: int, target_column: str, k: int = 10,
                            method: str = "f_classif", df_name: str = "main") -> dict:
    """
    Select top K features using statistical tests. Different from rank_features which uses model coefficients.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Accepts method: 'f_classif' (classification), 'f_regression' (regression),
        'mutual_info_classif', 'mutual_info_regression', 'chi2' (non-negative only).
    Requires: load_dataset must have been called. All features must be numeric.
    Stores: Overwrites dataframe with only target + top K features.
    Returns: selected features with scores, dropped features.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {df.columns.tolist()}"}

    feature_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in feature_cols:
        feature_cols.remove(target_column)

    non_numeric = [col for col in df.columns if col not in feature_cols and col != target_column]
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    if not feature_cols:
        return {"error": "No feature columns found."}


    score_functions = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
        "chi2": chi2
    }

    if method not in score_functions:
        return {"error": f"Unknown method '{method}'. Available: {list(score_functions.keys())}"}

    X = df[feature_cols]
    y = df[target_column]

    if X.isnull().any().any():
        X = X.fillna(X.median())

    k = min(k, len(feature_cols))

    try:
        selector = SelectKBest(score_func=score_functions[method], k=k)
        selector.fit(X, y)
    except Exception as e:
        return {"error": f"Feature selection failed: {str(e)}"}

    scores = selector.scores_
    selected_mask = selector.get_support()

    selected = []
    dropped = []
    for i, col in enumerate(feature_cols):
        entry = {"feature": col, "score": round(float(scores[i]), 4)}
        if selected_mask[i]:
            selected.append(entry)
        else:
            dropped.append(entry)

    selected.sort(key=lambda x: x["score"], reverse=True)
    dropped.sort(key=lambda x: x["score"], reverse=True)

    keep_cols = [target_column] + [s["feature"] for s in selected]
    df = df[keep_cols]
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Feature selection complete",
        "method": method,
        "k": k,
        "selected_features": selected,
        "dropped_features": dropped,
        "shape": df.shape
    }


# [x] 63. create_ratio_features

def create_ratio_features(session_id: int, column_pairs: list, target_column: str = None,
                           df_name: str = "main") -> dict:
    """
    Create ratio features by dividing pairs of numeric columns. E.g., salary/experience.
    Different from create_interactions which multiplies. Run this AFTER handle_missing_features.
    Accepts: list of tuples like [("salary", "experience"), ("revenue", "employees")].
    Requires: load_dataset must have been called. Columns must be numeric.
    Stores: Overwrites dataframe with new ratio columns.
    Returns: new columns created, shapes before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    prev_shape = df.shape

    new_cols = []
    skipped = []

    for pair in column_pairs:
        if len(pair) != 2:
            skipped.append({"pair": pair, "reason": "must be length 2"})
            continue

        numerator, denominator = pair

        if numerator not in df.columns or denominator not in df.columns:
            skipped.append({"pair": pair, "reason": "column not found"})
            continue

        if not pd.api.types.is_numeric_dtype(df[numerator]) or not pd.api.types.is_numeric_dtype(df[denominator]):
            skipped.append({"pair": pair, "reason": "not numeric"})
            continue

        col_name = f"{numerator}_div_{denominator}"
        df[col_name] = df[numerator] / df[denominator].replace(0, np.nan)

        inf_count = int(np.isinf(df[col_name]).sum())
        null_count = int(df[col_name].isnull().sum())
        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)

        median_val = df[col_name].median()
        fill_val = 0 if pd.isna(median_val) else median_val
        df[col_name] = df[col_name].fillna(fill_val)

        new_cols.append({
            "column": col_name,
            "infs_replaced": inf_count,
            "nulls_filled": null_count
        })

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Ratio features created",
        "new_columns": new_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }
    if skipped:
        result["skipped"] = skipped

    return result


# [x] 64. retrain_on_full_data
def retrain_on_full_data(session_id: int) -> dict:
    """
    Retrain the best model on the full dataset (X + y, not just X_train).
    Run this AFTER compare_models or train_single_model as the final step before deployment.
    Requires: trained_model and full X and y must exist.
    Stores: Overwrites trained_model with model trained on all data.
    Returns: model type, training rows, previous vs full training size.
    """
    state = check_state(session_id, ["trained_model", "X", "y"])
    if "error" in state:
        return state

    model = state["trained_model"]
    X = state["X"]
    y = state["y"]

    non_numeric = X.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    model_class = type(model)
    params = model.get_params()

    fresh_model = model_class(**params)

    try:
        fresh_model.fit(X, y)
    except Exception as e:
        return {"error": f"Retraining failed: {str(e)}"}

    prev_train_size = len(state.get("X_train", []))

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = fresh_model

    return {
        "status": "Model retrained on full data",
        "model_type": type(fresh_model).__name__,
        "previous_train_size": prev_train_size,
        "full_train_size": len(X),
        "features": X.columns.tolist()
    }


# [x] 65. visualize_missing

def visualize_missing(session_id: int, df_name: str = "main", output_path: str = None) -> dict: # type: ignore
    """
    Plot missingness patterns as a bar chart and heatmap. Different from get_basic_info which returns numbers only.
    Run this AFTER load_dataset, during exploration.
    Requires: load_dataset must have been called.
    Returns: path to saved plot, columns with missing values.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        return {"status": "No missing values found", "path": None}

    missing_pct = (missing / len(df) * 100).round(2)

    if output_path is None:
        output_path = os.path.join(_session_dir(session_id), "plot_missing_values.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(missing) * 0.4)))

    axes[0].barh(range(len(missing)), missing_pct.values)
    axes[0].set_yticks(range(len(missing)))
    axes[0].set_yticklabels(missing.index)
    axes[0].set_xlabel("Missing %")
    axes[0].set_title("Missing Values by Column")
    axes[0].invert_yaxis()

    for i, (count, pct) in enumerate(zip(missing.values, missing_pct.values)):
        axes[0].text(pct + 0.5, i, f"{count} ({pct}%)", va='center', fontsize=8)

    sample_size = min(100, len(df))
    sample = df[missing.index].head(sample_size)
    axes[1].imshow(sample.isnull().values, aspect='auto', cmap='YlOrRd', interpolation='none')
    axes[1].set_xticks(range(len(missing.index)))
    axes[1].set_xticklabels(missing.index, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel(f"Rows (first {sample_size})")
    axes[1].set_title("Missingness Pattern")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "columns_with_missing": missing_pct.to_dict()
    }

def subsample_data(session_id: int, n_rows: int = 30000, df_name: str = "main",
                   related_dfs: list = None, key_column: str = None,
                   stratify_column: str = None) -> dict:
    """
    Reduce dataframe size to prevent out-of-memory errors on large datasets.
    If you have multiple files to merge, pass the other df_names in related_dfs with the 
    shared key_column so they are filtered to match the exact same IDs.
    Run this IMMEDIATELY after load_dataset if rows > 50,000.
    Requires: All dataframes must be loaded via load_dataset first.
    Stores: Overwrites dataframes with smaller versions.
    Returns: row counts before and after for all affected dataframes.
    """
    
    
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    
    if len(df) <= n_rows:
        return {"status": "skipped", "reason": f"Row count {len(df)} is already <= {n_rows}"}

    if related_dfs:
        missing_dfs = [r for r in related_dfs if r not in state]
        if missing_dfs:
            return {"error": f"Related dataframes not found in session: {missing_dfs}"}
        if not key_column:
            return {"error": "Must provide key_column if providing related_dfs."}
        if key_column not in df.columns:
            return {"error": f"key_column '{key_column}' not found in {df_name}"}
        for rdf in related_dfs:
            if key_column not in state[rdf].columns:
                return {"error": f"key_column '{key_column}' not found in related df '{rdf}'"}

    stats = {df_name: {"before": len(df)}}

    # 1. Subsample Main DF
    if stratify_column and stratify_column in df.columns:
        try:
            df_sampled = df.groupby(stratify_column, group_keys=False).apply(
                lambda x: x.sample(int(np.rint(n_rows * len(x) / len(df))), random_state=42)
            ).sample(frac=1, random_state=42).reset_index(drop=True)
        except Exception:
            df_sampled = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    else:
        df_sampled = df.sample(n=n_rows, random_state=42).reset_index(drop=True)

    ACTIVE_DATAFRAMES[session_id][df_name] = df_sampled
    stats[df_name]["after"] = len(df_sampled)

    # 2. Filter Related DFs to match the exact same keys
    if related_dfs and key_column:
        valid_keys = set(df_sampled[key_column])
        for rdf in related_dfs:
            rdf_df = state[rdf]
            stats[rdf] = {"before": len(rdf_df)}
            rdf_df_filtered = rdf_df[rdf_df[key_column].isin(valid_keys)].reset_index(drop=True)
            ACTIVE_DATAFRAMES[session_id][rdf] = rdf_df_filtered
            stats[rdf]["after"] = len(rdf_df_filtered)

    gc.collect()

    return {
        "status": "Subsampling complete",
        "target_rows": n_rows,
        "details": stats
    }    