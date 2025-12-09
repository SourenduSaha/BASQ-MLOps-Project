"""
Airflow DAG for Pokemon Card Price Prediction Model Training Pipeline
Trains models for priceChange24hr, 7d_priceChange, price, and 7d_stddevPopPrice
with automatic model selection (XGBoost, Random Forest, Gradient Boosting, Ridge)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pendulum import datetime as pendulum_datetime
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import logging
import os
from dotenv import load_dotenv
# Use NON-LEGACY Evidently API (compatible with CloudWorkspace)
from evidently import Report, Regression, DataDefinition, Dataset
from evidently.presets import RegressionPreset, DataDriftPreset
from evidently.ui.workspace import CloudWorkspace

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Evidently configuration
EVIDENTLY_PROJECT_ID = os.getenv('EVIDENTLY_PROJECT_ID')
EVIDENTLY_API_TOKEN = os.getenv('EVIDENTLY_API_TOKEN')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'card_game_model_training_pipeline',
    default_args=default_args,
    description='Train card game price prediction models with auto-selection',
    schedule='@weekly',  # Run weekly (Airflow 3 uses 'schedule' instead of 'schedule_interval')
    start_date=pendulum_datetime(2024, 12, 1, tz="UTC"),
    catchup=False,
    tags=['machine-learning', 'card-game', 'price-prediction'],
)


def load_and_validate_data(**context):
    """
    Task 1: Load the dataset and perform initial validation
    """
    logger.info("Loading dataset...")
    
    # Load data
    df = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_training.csv')
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Validate required columns exist
    required_cols = ['priceChange24hr', '7d_priceChange', 'price', '7d_stddevPopPrice']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Store basic statistics
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'target_variables': required_cols,
        'missing_values': {
            'priceChange24hr': int(df['priceChange24hr'].isnull().sum()),
            '7d_priceChange': int(df['7d_priceChange'].isnull().sum()),
            'price': int(df['price'].isnull().sum()),
            '7d_stddevPopPrice': int(df['7d_stddevPopPrice'].isnull().sum())
        }
    }
    
    logger.info(f"Data validation complete: {stats}")
    
    # Push stats to XCom for downstream tasks
    context['task_instance'].xcom_push(key='data_stats', value=stats)
    
    return "Data loaded and validated successfully"


def prepare_features(df, target_column):
    """
    Prepare features for modeling with data cleaning and feature engineering
    """
    logger.info(f"Preparing features for: {target_column}")
    
    # Create a copy
    df_model = df.copy()
    
    # Drop rows where target is missing
    df_model = df_model[df_model[target_column].notna()].copy()
    logger.info(f"  Samples after removing missing targets: {len(df_model)}")
    
    # Select numeric features
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target variables and ID columns
    exclude_cols = ['priceChange24hr', '7d_priceChange', 'price', '7d_stddevPopPrice',
                    'tcgplayerId', 'tcgplayerSkuId', 'lastUpdated']
    
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Remove problematic columns
    cols_to_remove = []
    for col in numeric_features:
        if df_model[col].isna().all():
            cols_to_remove.append(col)
            logger.info(f"  Removing {col}: All NaN values")
        elif (df_model[col].fillna(0) == 0).all():
            cols_to_remove.append(col)
            logger.info(f"  Removing {col}: All zero values")
        elif df_model[col].nunique() == 1:
            cols_to_remove.append(col)
            logger.info(f"  Removing {col}: No variance")
    
    numeric_features = [col for col in numeric_features if col not in cols_to_remove]
    
    # Remove highly sparse columns (>95% missing)
    for col in numeric_features.copy():
        missing_ratio = df_model[col].isna().sum() / len(df_model)
        if missing_ratio > 0.95:
            numeric_features.remove(col)
            logger.info(f"  Removing {col}: {missing_ratio*100:.1f}% missing")
    
    # Handle missing values in numeric features
    for col in numeric_features:
        if df_model[col].isna().any():
            df_model[col].fillna(df_model[col].median(), inplace=True)
    
    # Encode categorical features
    categorical_features = ['game', 'rarity', 'condition', 'printing', 'language']
    label_encoders = {}
    
    for col in categorical_features:
        if col in df_model.columns:
            df_model[col].fillna('Unknown', inplace=True)
            le = LabelEncoder()
            df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
            numeric_features.append(col + '_encoded')
    
    # Remove outliers using IQR method
    Q1 = df_model[target_column].quantile(0.01)
    Q3 = df_model[target_column].quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    original_size = len(df_model)
    df_model = df_model[(df_model[target_column] >= lower_bound) & 
                        (df_model[target_column] <= upper_bound)]
    removed_outliers = original_size - len(df_model)
    
    if removed_outliers > 0:
        logger.info(f"  Removed {removed_outliers} outliers ({removed_outliers/original_size*100:.2f}%)")
    
    # Prepare feature matrix and target
    X = df_model[numeric_features]
    y = df_model[target_column]
    
    logger.info(f"  Final features: {len(numeric_features)}, samples: {len(X)}")
    logger.info(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, numeric_features, label_encoders


def train_best_model(X, y, target_name):
    """
    Train all model types and select the best based on test R²
    """
    logger.info(f"Training models for: {target_name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define all models
    model_configs = {
        'xgboost': {
            'model': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            ),
            'use_scaled': False
        },
        'random_forest': {
            'model': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'use_scaled': False
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'use_scaled': False
        },
        'ridge': {
            'model': Ridge(alpha=1.0, random_state=42),
            'use_scaled': True
        }
    }
    
    # Train all models and collect results
    results = {}
    
    for name, config in model_configs.items():
        logger.info(f"  Training {name}...")
        model = config['model']
        
        # Train
        if config['use_scaled']:
            model.fit(X_train_scaled, y_train)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results[name] = {
            'model': model,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        logger.info(f"    {name}: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_result = results[best_model_name]
    
    logger.info(f"BEST MODEL: {best_model_name.upper()}")
    logger.info(f"R2={best_result['test_r2']:.4f}")
    
    return best_result['model'], scaler, best_model_name, {
        'test_r2': best_result['test_r2'],
        'test_rmse': best_result['test_rmse'],
        'test_mae': best_result['test_mae'],
        'model_type': best_model_name
    }


def create_evidently_report(target_name, y_true, y_pred, dataset_type='training_local'):
    """
    Create and upload Evidently AI report for model evaluation
    """
    logger.info(f"Creating Evidently report for {target_name} ({dataset_type})")
    
    try:
        # Create DataFrame with predictions
        report_df = pd.DataFrame({
            'target': y_true,
            'prediction': y_pred
        })
        
        # Define data definition (NON-LEGACY API)
        data_definition = DataDefinition(
            regression=[Regression(name='default', target='target', prediction='prediction')]
        )
        
        # Create datasets (NON-LEGACY API)
        reference_dataset = Dataset.from_pandas(report_df, data_definition=data_definition)
        current_dataset = Dataset.from_pandas(report_df, data_definition=data_definition)
        
        # Create report (NON-LEGACY API)
        regression_report = Report(metrics=[RegressionPreset()])
        
        # Set tags BEFORE running (on Report object, not Snapshot)
        regression_report.tags = [target_name, dataset_type, 'regression']
        
        # Run report and CAPTURE return value (NON-LEGACY API returns Snapshot)
        eval_report = regression_report.run(
            reference_data=reference_dataset,
            current_data=current_dataset
        )
        
        # Save report locally
        report_dir = '/home/qthompson/airflow/dags/reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = f'{report_dir}/{target_name}_{dataset_type}_report.html'
        eval_report.save_html(report_path)
        logger.info(f"  Saved HTML report to: {report_path}")
        
        # Upload to Evidently Cloud (0.7.17+ API)
        if EVIDENTLY_PROJECT_ID and EVIDENTLY_API_TOKEN:
            try:
                ws = CloudWorkspace(
                    token=EVIDENTLY_API_TOKEN,
                    url="https://app.evidently.cloud"
                )
                
                # Upload the snapshot using add_run (0.7.17+ API)
                result = ws.add_run(
                    EVIDENTLY_PROJECT_ID,
                    eval_report  # Pass eval_report snapshot directly
                )
                
                # Get URL from result
                upload_url = result.url if hasattr(result, 'url') else f"https://app.evidently.cloud/projects/{EVIDENTLY_PROJECT_ID}/reports/{result.id}"
                logger.info(f"Uploaded report to Evidently Cloud: {upload_url}")
                logger.info(f"Tags: {target_name}, {dataset_type}, regression")
                logger.info(f"Report ID: {result.id if hasattr(result, 'id') else 'N/A'}")
                    
            except Exception as e:
                logger.warning(f"Could not upload to Evidently Cloud: {str(e)}")
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error creating Evidently report: {str(e)}")
        return None


def train_priceChange24hr_model(**context):
    """
    Task 2: Train model for priceChange24hr
    """
    logger.info("=" * 60)
    logger.info("Training model for: priceChange24hr")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_training.csv')
    
    # Prepare features
    X, y, features, encoders = prepare_features(df, 'priceChange24hr')
    
    # Train best model
    model, scaler, model_type, metrics = train_best_model(X, y, 'priceChange24hr')
    
    # Generate predictions for Evidently report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    if model_type == 'ridge':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Create Evidently report
    report_path = create_evidently_report('priceChange24hr', y_test.values, y_pred, 'training_local')
    
    # Save model artifacts
    joblib.dump(model, f'/home/qthompson/airflow/dags/models/model_priceChange24hr_{model_type}.pkl')
    joblib.dump(scaler, f'/home/qthompson/airflow/dags/models/scaler_priceChange24hr_{model_type}.pkl')
    joblib.dump(features, f'/home/qthompson/airflow/dags/models/features_priceChange24hr_{model_type}.pkl')
    
    logger.info(f"Model saved: priceChange24hr ({model_type})")
    
    # Push metrics to XCom
    metrics['report_path'] = report_path
    context['task_instance'].xcom_push(key='priceChange24hr_metrics', value=metrics)
    
    return f"priceChange24hr model trained: {model_type}"


def train_7d_priceChange_model(**context):
    """
    Task 3: Train model for 7d_priceChange
    """
    logger.info("=" * 60)
    logger.info("Training model for: 7d_priceChange")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_training.csv')
    
    # Prepare features
    X, y, features, encoders = prepare_features(df, '7d_priceChange')
    
    # Train best model
    model, scaler, model_type, metrics = train_best_model(X, y, '7d_priceChange')
    
    # Generate predictions for Evidently report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    if model_type == 'ridge':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Create Evidently report
    report_path = create_evidently_report('7d_priceChange', y_test.values, y_pred, 'training_local')
    
    # Save model artifacts
    joblib.dump(model, f'/home/qthompson/airflow/dags/models/model_7d_priceChange_{model_type}.pkl')
    joblib.dump(scaler, f'/home/qthompson/airflow/dags/models/scaler_7d_priceChange_{model_type}.pkl')
    joblib.dump(features, f'/home/qthompson/airflow/dags/models/features_7d_priceChange_{model_type}.pkl')
    
    logger.info(f"Model saved: 7d_priceChange ({model_type})")
    
    # Push metrics to XCom
    metrics['report_path'] = report_path
    context['task_instance'].xcom_push(key='7d_priceChange_metrics', value=metrics)
    
    return f"7d_priceChange model trained: {model_type}"


def train_price_model(**context):
    """
    Task 4: Train model for price
    """
    logger.info("=" * 60)
    logger.info("Training model for: price")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_training.csv')
    
    # Prepare features
    X, y, features, encoders = prepare_features(df, 'price')
    
    # Train best model
    model, scaler, model_type, metrics = train_best_model(X, y, 'price')
    
    # Generate predictions for Evidently report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    if model_type == 'ridge':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Create Evidently report
    report_path = create_evidently_report('price', y_test.values, y_pred, 'training_local')
    
    # Save model artifacts
    joblib.dump(model, f'/home/qthompson/airflow/dags/models/model_price_{model_type}.pkl')
    joblib.dump(scaler, f'/home/qthompson/airflow/dags/models/scaler_price_{model_type}.pkl')
    joblib.dump(features, f'/home/qthompson/airflow/dags/models/features_price_{model_type}.pkl')
    
    logger.info(f"Model saved: price ({model_type})")
    
    # Push metrics to XCom
    metrics['report_path'] = report_path
    context['task_instance'].xcom_push(key='price_metrics', value=metrics)
    
    return f"price model trained: {model_type}"


def train_7d_stddevPopPrice_model(**context):
    """
    Task 5: Train model for 7d_stddevPopPrice
    """
    logger.info("=" * 60)
    logger.info("Training model for: 7d_stddevPopPrice")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_training.csv')
    
    # Prepare features
    X, y, features, encoders = prepare_features(df, '7d_stddevPopPrice')
    
    # Train best model
    model, scaler, model_type, metrics = train_best_model(X, y, '7d_stddevPopPrice')
    
    # Generate predictions for Evidently report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    if model_type == 'ridge':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Create Evidently report
    report_path = create_evidently_report('7d_stddevPopPrice', y_test.values, y_pred, 'training_local')
    
    # Save model artifacts
    joblib.dump(model, f'/home/qthompson/airflow/dags/models/model_7d_stddevPopPrice_{model_type}.pkl')
    joblib.dump(scaler, f'/home/qthompson/airflow/dags/models/scaler_7d_stddevPopPrice_{model_type}.pkl')
    joblib.dump(features, f'/home/qthompson/airflow/dags/models/features_7d_stddevPopPrice_{model_type}.pkl')
    
    logger.info(f"Model saved: 7d_stddevPopPrice ({model_type})")
    
    # Push metrics to XCom
    metrics['report_path'] = report_path
    context['task_instance'].xcom_push(key='7d_stddevPopPrice_metrics', value=metrics)
    
    return f"7d_stddevPopPrice model trained: {model_type}"


def test_models_on_weekpass(**context):
    """
    Task 6: Test trained models on weekpass dataset
    """
    logger.info("=" * 60)
    logger.info("TESTING MODELS ON WEEKPASS DATASET")
    logger.info("=" * 60)
    
    # Load weekpass test data
    logger.info("Loading weekpass test dataset...")
    df_test = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_weekpass.csv')
    logger.info(f"Weekpass dataset loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    
    target_variables = ['priceChange24hr', '7d_priceChange', 'price', '7d_stddevPopPrice']
    test_results = {}
    
    for target in target_variables:
        logger.info(f"\nTesting model for: {target}")
        logger.info("-" * 60)
        
        try:
            # Pull training metrics to get model type
            ti = context['task_instance']
            train_metrics = ti.xcom_pull(task_ids=f'train_{target}_model', key=f'{target}_metrics')
            model_type = train_metrics['model_type']
            
            # Load model artifacts
            model = joblib.load(f'/home/qthompson/airflow/dags/models/model_{target}_{model_type}.pkl')
            scaler = joblib.load(f'/home/qthompson/airflow/dags/models/scaler_{target}_{model_type}.pkl')
            features = joblib.load(f'/home/qthompson/airflow/dags/models/features_{target}_{model_type}.pkl')
            
            logger.info(f"  Loaded {model_type} model and artifacts")
            
            # Prepare test features (same preprocessing as training)
            X_test, y_test, _, _ = prepare_features(df_test, target)
            
            # Ensure same feature set
            missing_features = [f for f in features if f not in X_test.columns]
            if missing_features:
                logger.warning(f"  Missing features in test data: {missing_features}")
                for feat in missing_features:
                    X_test[feat] = 0
            
            # Select only the features used during training
            X_test = X_test[features]
            
            # Scale and predict
            X_test_scaled = scaler.transform(X_test)
            
            # Determine if model uses scaled features
            if model_type == 'ridge':
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            
            # Create Evidently report for weekpass data
            weekpass_report_path = create_evidently_report(target, y_test.values, y_pred, 'weekpass')
            
            test_results[target] = {
                'model_type': model_type,
                'test_samples': len(y_test),
                'test_r2': float(test_r2),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'target_mean': float(y_test.mean()),
                'target_std': float(y_test.std()),
                'report_path': weekpass_report_path
            }
            
            logger.info(f"  Test Results on Weekpass Data:")
            logger.info(f"    Samples:    {len(y_test)}")
            logger.info(f"    R2:         {test_r2:.4f}")
            logger.info(f"    RMSE:       {test_rmse:.4f}")
            logger.info(f"    MAE:        {test_mae:.4f}")
            logger.info(f"    Target Mean: {y_test.mean():.2f}")
            logger.info(f"    Target Std:  {y_test.std():.2f}")
            
        except Exception as e:
            logger.error(f"  Error testing {target} model: {str(e)}")
            test_results[target] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # Push test results to XCom
    context['task_instance'].xcom_push(key='weekpass_test_results', value=test_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("WEEKPASS TESTING COMPLETE")
    logger.info("=" * 60)
    
    return "Weekpass testing completed"


def generate_summary_report(**context):
    """
    Task 7: Generate summary report of all models including weekpass test results
    """
    logger.info("=" * 60)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 60)
    
    ti = context['task_instance']
    
    # Pull training metrics from XCom
    metrics_24hr = ti.xcom_pull(task_ids='train_priceChange24hr_model', key='priceChange24hr_metrics')
    metrics_7d = ti.xcom_pull(task_ids='train_7d_priceChange_model', key='7d_priceChange_metrics')
    metrics_price = ti.xcom_pull(task_ids='train_price_model', key='price_metrics')
    metrics_stddev = ti.xcom_pull(task_ids='train_7d_stddevPopPrice_model', key='7d_stddevPopPrice_metrics')
    
    # Pull weekpass test results
    weekpass_results = ti.xcom_pull(task_ids='test_models_on_weekpass', key='weekpass_test_results')
    
    # Create summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'training_metrics': {
            'priceChange24hr': metrics_24hr,
            '7d_priceChange': metrics_7d,
            'price': metrics_price,
            '7d_stddevPopPrice': metrics_stddev
        },
        'weekpass_test_metrics': weekpass_results
    }
    
    # Log training summary
    logger.info("\nMODEL TRAINING SUMMARY:")
    logger.info("-" * 60)
    for target, metrics in summary['training_metrics'].items():
        logger.info(f"{target}:")
        logger.info(f"  Model Type: {metrics['model_type'].upper()}")
        logger.info(f"  Test R²:    {metrics['test_r2']:.4f}")
        logger.info(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
        logger.info(f"  Test MAE:   {metrics['test_mae']:.4f}")
        logger.info("-" * 60)
    
    # Log weekpass test summary
    logger.info("\nWEEKPASS TEST SUMMARY:")
    logger.info("-" * 60)
    for target, results in weekpass_results.items():
        if 'error' not in results:
            logger.info(f"{target}:")
            logger.info(f"  Model Type:  {results['model_type'].upper()}")
            logger.info(f"  Samples:     {results['test_samples']}")
            logger.info(f"  Test R²:     {results['test_r2']:.4f}")
            logger.info(f"  Test RMSE:   {results['test_rmse']:.4f}")
            logger.info(f"  Test MAE:    {results['test_mae']:.4f}")
            logger.info("-" * 60)
        else:
            logger.info(f"{target}: FAILED - {results['error']}")
            logger.info("-" * 60)
    
    # Save summary to file
    with open('/home/qthompson/airflow/dags/models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Summary report saved to training_summary.json")
    
    return "Summary report generated successfully"


def detect_data_drift(**context):
    """
    Task 8: Detect data drift between training and weekpass datasets using Evidently
    """
    logger.info("=" * 60)
    logger.info("DETECTING DATA DRIFT")
    logger.info("=" * 60)
    
    # Load both datasets
    logger.info("Loading training and weekpass datasets...")
    df_training = pd.read_csv('/home/qthompson/airflow/dags/data/all_cards_master_data_training.csv')
    df_weekpass = pd.read_csv('/home/qthompson/all_cards_master_data_weekpass.csv')
    
    logger.info(f"Training data: {df_training.shape[0]} rows")
    logger.info(f"Weekpass data: {df_weekpass.shape[0]} rows")
    
    # Select numeric columns for drift detection
    numeric_cols = df_training.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID columns and target variables for cleaner drift analysis
    exclude_cols = ['tcgplayerId', 'tcgplayerSkuId', 'lastUpdated']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Keep only common columns between both datasets
    common_cols = list(set(numeric_cols) & set(df_weekpass.columns))
    
    # Filter out columns that are empty or have all NaN values in either dataset
    valid_cols = []
    for col in common_cols:
        # Check if column has any non-null values in both datasets
        training_valid = df_training[col].notna().any() and not (df_training[col].fillna(0) == 0).all()
        weekpass_valid = df_weekpass[col].notna().any() and not (df_weekpass[col].fillna(0) == 0).all()
        
        if training_valid and weekpass_valid:
            valid_cols.append(col)
        else:
            logger.info(f"Skipping column '{col}': empty or all zeros")
    
    common_cols = valid_cols
    logger.info(f"Analyzing {len(common_cols)} valid common numeric features for drift")
    
    # Ensure we have at least some columns to analyze
    if len(common_cols) == 0:
        logger.error("No valid common columns found for drift detection")
        return "Data drift detection skipped: no valid columns"
    
    # Sample if datasets are too large
    if len(df_training) > 10000:
        df_training_sample = df_training[common_cols].sample(10000, random_state=42)
    else:
        df_training_sample = df_training[common_cols].copy()
        
    if len(df_weekpass) > 10000:
        df_weekpass_sample = df_weekpass[common_cols].sample(10000, random_state=42)
    else:
        df_weekpass_sample = df_weekpass[common_cols].copy()
    
    # Fill any remaining NaN values with median to avoid Evidently errors
    for col in common_cols:
        if df_training_sample[col].isna().any():
            df_training_sample[col].fillna(df_training_sample[col].median(), inplace=True)
        if df_weekpass_sample[col].isna().any():
            df_weekpass_sample[col].fillna(df_weekpass_sample[col].median(), inplace=True)
    
    # Create Evidently Datasets
    reference_dataset = Dataset.from_pandas(df_training_sample)
    current_dataset = Dataset.from_pandas(df_weekpass_sample)
    
    # Generate data drift report
    logger.info("Generating data drift report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # Set tags BEFORE running (on Report object, not Snapshot)
    drift_report.tags = ['data_drift', 'training_vs_weekpass', 'monitoring']
    
    # Run report and capture Snapshot
    drift_snapshot = drift_report.run(
        reference_data=reference_dataset,
        current_data=current_dataset
    )
    
    # Save HTML report locally using Snapshot
    report_path = '/home/qthompson/airflow/dags/reports/data_drift_report.html'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    drift_snapshot.save_html(report_path)
    logger.info(f"Drift report saved to: {report_path}")
    
    # Create drift summary (Snapshot doesn't support as_dict())
    drift_summary = {
        'timestamp': datetime.now().isoformat(),
        'reference_samples': len(df_training_sample),
        'current_samples': len(df_weekpass_sample),
        'features_analyzed': len(common_cols),
        'report_path': report_path,
        'note': 'Detailed drift metrics available in HTML report'
    }
    
    logger.info("\nDRIFT DETECTION SUMMARY:")
    logger.info(f"  Reference samples: {len(df_training_sample)}")
    logger.info(f"  Current samples: {len(df_weekpass_sample)}")
    logger.info(f"  Features analyzed: {len(common_cols)}")
    logger.info(f"  Report saved: {report_path}")
    
    # Upload to Evidently Cloud if configured
    if EVIDENTLY_PROJECT_ID and EVIDENTLY_API_TOKEN:
        try:
            logger.info("Uploading drift report to Evidently Cloud...")
            ws = CloudWorkspace(
                token=EVIDENTLY_API_TOKEN,
                url="https://app.evidently.cloud"
            )
            
            # Upload the snapshot
            result = ws.add_run(EVIDENTLY_PROJECT_ID, drift_snapshot)
            upload_url = result.url if hasattr(result, 'url') else f"https://app.evidently.cloud/projects/{EVIDENTLY_PROJECT_ID}/reports/{result.id}"
            logger.info(f"✅ Drift report uploaded to Evidently Cloud: {upload_url}")
            logger.info(f"   Tags: data_drift, training_vs_weekpass, monitoring")
            drift_summary['uploaded_to_cloud'] = True
            drift_summary['upload_url'] = upload_url
        except Exception as e:
            logger.warning(f"Failed to upload to Evidently Cloud: {e}")
            drift_summary['uploaded_to_cloud'] = False
    else:
        logger.info("Evidently Cloud credentials not configured, skipping upload")
        drift_summary['uploaded_to_cloud'] = False
    
    # Save drift summary to JSON
    summary_path = '/home/qthompson/airflow/dags/reports/drift_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(drift_summary, f, indent=2)
    
    logger.info(f"Drift summary saved to: {summary_path}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='drift_summary', value=drift_summary)
    
    logger.info("=" * 60)
    logger.info("DATA DRIFT DETECTION COMPLETE")
    logger.info("=" * 60)
    
    return "Data drift detection completed"


# Define tasks
task_load_data = PythonOperator(
    task_id='load_and_validate_data',
    python_callable=load_and_validate_data,
    dag=dag,
)

task_train_24hr = PythonOperator(
    task_id='train_priceChange24hr_model',
    python_callable=train_priceChange24hr_model,
    dag=dag,
)

task_train_7d = PythonOperator(
    task_id='train_7d_priceChange_model',
    python_callable=train_7d_priceChange_model,
    dag=dag,
)

task_train_price = PythonOperator(
    task_id='train_price_model',
    python_callable=train_price_model,
    dag=dag,
)

task_train_stddev = PythonOperator(
    task_id='train_7d_stddevPopPrice_model',
    python_callable=train_7d_stddevPopPrice_model,
    dag=dag,
)

task_test_weekpass = PythonOperator(
    task_id='test_models_on_weekpass',
    python_callable=test_models_on_weekpass,
    dag=dag,
)

task_summary = PythonOperator(
    task_id='generate_summary_report',
    python_callable=generate_summary_report,
    dag=dag,
)

task_drift_detection = PythonOperator(
    task_id='detect_data_drift',
    python_callable=detect_data_drift,
    dag=dag,
)

# Define task dependencies
# Load data first, then train all models in parallel, then test on weekpass data, 
# then generate summary and detect drift in parallel, as they are independent
task_load_data >> [task_train_24hr, task_train_7d, task_train_price, task_train_stddev] >> task_test_weekpass >> [task_summary, task_drift_detection]
