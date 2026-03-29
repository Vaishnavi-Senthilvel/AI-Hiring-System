"""
Phase 4-5: Machine Learning Models & Evaluation
- Train multiple models: Logistic Regression, Decision Tree, Random Forest, SVM
- Hyperparameter tuning with GridSearchCV
- Model evaluation: accuracy, precision, recall, F1-score, ROC-AUC
- Compare all models and select the best one
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Train and evaluate machine learning models for candidate selection"""
    
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.evaluation_results = {}
    
    def prepare_data(self, X, y):
        """
        Split and scale data
        
        Args:
            X: Feature matrix
            y: Target variable (0: Reject, 1: Shortlist)
        """
        # Handle both dataframe and numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Data split: {len(self.X_train)} train, {len(self.X_test)} test")
    
    def train_logistic_regression(self, hyperparameter_tuning=True):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        if hyperparameter_tuning:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            }
            
            lr = LogisticRegression(random_state=self.random_state)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = LogisticRegression(random_state=self.random_state, max_iter=200)
            model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        return model
    
    def train_decision_tree(self, hyperparameter_tuning=True):
        """Train Decision Tree model"""
        logger.info("Training Decision Tree...")
        
        if hyperparameter_tuning:
            param_grid = {
                'max_depth': [3, 5, 7, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            dt = DecisionTreeClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = DecisionTreeClassifier(random_state=self.random_state)
            model.fit(self.X_train, self.y_train)
        
        self.models['Decision Tree'] = model
        return model
    
    def train_random_forest(self, hyperparameter_tuning=True):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        return model
    
    def train_svm(self, hyperparameter_tuning=True):
        """Train Support Vector Machine model"""
        logger.info("Training SVM...")
        
        if hyperparameter_tuning:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
            
            svm = SVC(random_state=self.random_state, probability=True)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best params: {grid_search.best_params_}")
        else:
            model = SVC(random_state=self.random_state, probability=True)
            model.fit(self.X_train, self.y_train)
        
        self.models['SVM'] = model
        return model
    
    def train_all_models(self, hyperparameter_tuning=True):
        """Train all models"""
        self.train_logistic_regression(hyperparameter_tuning)
        self.train_decision_tree(hyperparameter_tuning)
        self.train_random_forest(hyperparameter_tuning)
        self.train_svm(hyperparameter_tuning)
        
        logger.info(f"Trained {len(self.models)} models")
    
    def evaluate_model(self, model, model_name):
        """
        Evaluate a single model with multiple metrics
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        return metrics
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        results = []
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, model_name)
            results.append(metrics)
            self.evaluation_results[model_name] = metrics
        
        results_df = pd.DataFrame(results)
        logger.info("\nModel Evaluation Results:")
        logger.info(results_df.to_string())
        
        # Select best model based on F1-Score
        self.best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"\nBest Model: {self.best_model_name} (F1-Score: {self.evaluation_results[self.best_model_name]['F1-Score']:.4f})")
        
        return results_df
    
    def get_detailed_evaluation(self, model_name):
        """Get detailed evaluation for a model"""
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        print(f"\n=== Detailed Evaluation: {model_name} ===")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
    
    def plot_model_comparison(self):
        """Plot comparison of all models"""
        results_df = pd.DataFrame(list(self.evaluation_results.values()))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for metric, pos in zip(metrics, positions):
            ax = axes[pos]
            results_df.set_index('Model')[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, model_name=None):
        """Plot confusion matrix for a model"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Shortlisted'],
                   yticklabels=['Rejected', 'Shortlisted'])
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance from tree-based models"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"{model_name} doesn't have feature_importances_")
            return None
        
        importances = model.feature_importances_
        return importances
    
    def save_model(self, model_name, filepath):
        """Save model to disk"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def cross_validate(self, cv_folds=5):
        """Perform cross-validation for all models"""
        cv_results = {}
        
        for model_name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, 
                                   cv=cv_folds, scoring='f1')
            cv_results[model_name] = {
                'Mean F1-Score': scores.mean(),
                'Std Dev': scores.std(),
                'Fold Scores': scores
            }
            logger.info(f"{model_name} - CV F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
