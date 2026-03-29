"""
Phase 8: Model Persistence & Utilities
- Save and load trained models
- Model version management
- Pipeline serialization
"""

import joblib
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handle model saving and loading"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(self, model, model_name, version=None):
        """
        Save model to disk
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Version identifier (auto-generated if None)
        
        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{model_name}_{version}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
        
        # Save metadata
        self._save_metadata(model_name, version, filepath)
        
        return filepath
    
    def load_model(self, model_name, version=None):
        """
        Load model from disk
        
        Args:
            model_name: Name of model
            version: Version to load (latest if None)
        
        Returns:
            Loaded model object
        """
        if version is None:
            # Load latest version
            filepath = self._find_latest_model(model_name)
        else:
            filename = f"{model_name}_{version}.pkl"
            filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded: {filepath}")
        
        return model
    
    def save_vectorizer(self, vectorizer, name, version=None):
        """Save TF-IDF vectorizer"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"vectorizer_{name}_{version}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(vectorizer, filepath)
        logger.info(f"Vectorizer saved: {filepath}")
        
        return filepath
    
    def load_vectorizer(self, name, version=None):
        """Load TF-IDF vectorizer"""
        if version is None:
            filepath = self._find_latest_model(f"vectorizer_{name}")
        else:
            filename = f"vectorizer_{name}_{version}.pkl"
            filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer not found: {filepath}")
        
        vectorizer = joblib.load(filepath)
        logger.info(f"Vectorizer loaded: {filepath}")
        
        return vectorizer
    
    def save_scaler(self, scaler, version=None):
        """Save feature scaler"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"scaler_{version}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(scaler, filepath)
        logger.info(f"Scaler saved: {filepath}")
        
        return filepath
    
    def load_scaler(self, version=None):
        """Load feature scaler"""
        if version is None:
            filepath = self._find_latest_model("scaler")
        else:
            filename = f"scaler_{version}.pkl"
            filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler not found: {filepath}")
        
        scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded: {filepath}")
        
        return scaler
    
    def _find_latest_model(self, model_prefix):
        """Find latest version of a model"""
        matching_files = [
            f for f in os.listdir(self.models_dir)
            if f.startswith(model_prefix) and f.endswith('.pkl')
        ]
        
        if not matching_files:
            raise FileNotFoundError(f"No models found with prefix: {model_prefix}")
        
        # Sort by date (assuming YYYYMMDD format)
        latest = sorted(matching_files)[-1]
        return os.path.join(self.models_dir, latest)
    
    def _save_metadata(self, model_name, version, filepath):
        """Save model metadata"""
        metadata = {
            'model_name': model_name,
            'version': version,
            'path': filepath,
            'timestamp': datetime.now().isoformat(),
            'type': 'sklearn_model'
        }
        
        metadata_file = os.path.join(self.models_dir, f"{model_name}_{version}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def list_models(self):
        """List all available models"""
        models = {}
        
        files = os.listdir(self.models_dir)
        
        for file in sorted(files):
            if file.endswith('.pkl') and not file.startswith('vectorizer_') and not file.startswith('scaler_'):
                model_name = file.split('_')[0]
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(file)
        
        return models
    
    def get_model_info(self, model_name):
        """Get information about a model"""
        metadata_files = [
            f for f in os.listdir(self.models_dir)
            if f.startswith(f"{model_name}_") and f.endswith('_metadata.json')
        ]
        
        info = []
        for metadata_file in sorted(metadata_files, reverse=True):
            filepath = os.path.join(self.models_dir, metadata_file)
            with open(filepath, 'r') as f:
                info.append(json.load(f))
        
        return info


class PipelineState:
    """Manage state of entire ML pipeline"""
    
    def __init__(self, state_file='pipeline_state.json'):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self):
        """Load pipeline state from file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'last_trained': None,
            'best_model': None,
            'metrics': {},
            'data_version': None
        }
    
    def update_state(self, **kwargs):
        """Update pipeline state"""
        self.state.update(kwargs)
        self._save_state()
    
    def _save_state(self):
        """Save pipeline state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
        logger.info(f"Pipeline state saved to {self.state_file}")
    
    def get_state(self):
        """Get current pipeline state"""
        return self.state.copy()


class PredictionCache:
    """Cache predictions for faster retrieval"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_predictions(self, predictions_df, cache_id):
        """Save predictions to cache"""
        filepath = os.path.join(self.cache_dir, f"{cache_id}.pkl")
        joblib.dump(predictions_df, filepath)
        logger.info(f"Predictions cached: {filepath}")
        return filepath
    
    def load_predictions(self, cache_id):
        """Load predictions from cache"""
        filepath = os.path.join(self.cache_dir, f"{cache_id}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cache not found: {cache_id}")
        
        predictions = joblib.load(filepath)
        logger.info(f"Predictions loaded from cache: {cache_id}")
        return predictions
    
    def clear_cache(self):
        """Clear all cached predictions"""
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, file))
        logger.info("Cache cleared")
