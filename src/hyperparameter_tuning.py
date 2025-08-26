from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import optuna  # pip install optuna

def tune_hyperparameters_advanced(self, X, y, method='optuna'):
    """Advanced hyperparameter tuning with multiple methods"""
    if method == 'optuna':
        return self._optuna_tuning(X, y)
    elif method == 'randomized':
        return self._randomized_search(X, y)
    else:
        return self.tune_hyperparameters(X, y)  # fallback to grid search

def _optuna_tuning(self, X, y):
    """Optuna-based hyperparameter optimization"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        model = RandomForestRegressor(random_state=42, **params)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    return RandomForestRegressor(random_state=42, **study.best_params)
