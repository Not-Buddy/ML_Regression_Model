from scipy.stats import ks_2samp
import numpy as np

def detect_data_drift(self, X_train, X_new, threshold=0.05):
    """Detect data drift using Kolmogorov-Smirnov test"""
    drift_detected = {}
    
    for column in X_train.columns:
        if column in X_new.columns:
            # Perform KS test
            statistic, p_value = ks_2samp(X_train[column], X_new[column])
            drift_detected[column] = {
                'p_value': p_value,
                'drift': p_value < threshold,
                'severity': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
            }
    
    return drift_detected

def monitor_prediction_quality(self, y_true, y_pred, baseline_metrics=None):
    """Monitor model performance over time"""
    current_metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
    
    if baseline_metrics:
        performance_change = {
            metric: ((current_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]) * 100
            for metric in current_metrics.keys()
        }
        return current_metrics, performance_change
    
    return current_metrics
