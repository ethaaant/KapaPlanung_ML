"""
Model versioning and management system.
Handles model saving, loading, versioning, and metadata tracking.
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import joblib


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    version: str
    model_type: str
    created_at: datetime
    created_by: str
    
    # Training info
    training_data_hash: str = ""
    training_samples: int = 0
    training_date_range: tuple = ("", "")
    features_used: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_active: bool = False
    description: str = ""
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelManager:
    """
    Manages model lifecycle: save, load, version, and track models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
        self.active_dir = self.models_dir / "active"
        self.active_dir.mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.models_dir / "models_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        """Load the model registry."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    registry = {}
                    for model_id, versions in data.items():
                        registry[model_id] = [
                            ModelMetadata.from_dict(v) for v in versions
                        ]
                    return registry
            except (json.JSONDecodeError, KeyError):
                return {}
        return {}
    
    def _save_registry(self):
        """Save the model registry."""
        data = {}
        for model_id, versions in self.registry.items():
            data[model_id] = [v.to_dict() for v in versions]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _generate_version(self, model_id: str) -> str:
        """Generate a new version number for a model."""
        if model_id not in self.registry:
            return "1.0.0"
        
        versions = self.registry[model_id]
        if not versions:
            return "1.0.0"
        
        # Get latest version
        latest = versions[-1].version
        parts = [int(p) for p in latest.split(".")]
        parts[2] += 1  # Increment patch version
        
        # Roll over if needed
        if parts[2] >= 100:
            parts[2] = 0
            parts[1] += 1
        if parts[1] >= 100:
            parts[1] = 0
            parts[0] += 1
        
        return ".".join(map(str, parts))
    
    def _compute_data_hash(self, data) -> str:
        """Compute a hash of the training data."""
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            # Use shape and sample of data for hash
            hash_str = f"{data.shape}_{data.columns.tolist()}_{data.iloc[0].tolist() if len(data) > 0 else ''}"
        else:
            hash_str = str(data)
        
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    def save_model(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        created_by: str,
        training_data=None,
        metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None,
        features_used: List[str] = None,
        target_columns: List[str] = None,
        description: str = "",
        set_active: bool = True
    ) -> ModelMetadata:
        """
        Save a model with versioning.
        
        Args:
            model: The model object to save
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., "HistGradientBoostingRegressor")
            created_by: Username of the creator
            training_data: Optional training data for hash
            metrics: Optional performance metrics
            config: Optional configuration dict
            features_used: Optional list of feature names
            target_columns: Optional list of target columns
            description: Optional description
            set_active: Whether to set this version as active
            
        Returns:
            ModelMetadata object
        """
        version = self._generate_version(model_id)
        now = datetime.now()
        
        # Compute data hash if training data provided
        data_hash = ""
        training_samples = 0
        date_range = ("", "")
        
        if training_data is not None:
            import pandas as pd
            data_hash = self._compute_data_hash(training_data)
            if isinstance(training_data, pd.DataFrame):
                training_samples = len(training_data)
                if "timestamp" in training_data.columns:
                    date_range = (
                        training_data["timestamp"].min().isoformat(),
                        training_data["timestamp"].max().isoformat()
                    )
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            model_type=model_type,
            created_at=now,
            created_by=created_by,
            training_data_hash=data_hash,
            training_samples=training_samples,
            training_date_range=date_range,
            features_used=features_used or [],
            target_columns=target_columns or [],
            metrics=metrics or {},
            config=config or {},
            is_active=set_active,
            description=description
        )
        
        # Save model file
        model_filename = f"{model_id}_v{version.replace('.', '_')}.joblib"
        model_path = self.versions_dir / model_filename
        joblib.dump(model, model_path)
        
        # Update registry
        if model_id not in self.registry:
            self.registry[model_id] = []
        
        # If setting as active, deactivate all other versions
        if set_active:
            for v in self.registry[model_id]:
                v.is_active = False
            
            # Also save to active directory
            active_path = self.active_dir / f"{model_id}.joblib"
            joblib.dump(model, active_path)
            
            # Save metadata alongside
            meta_path = self.active_dir / f"{model_id}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        
        self.registry[model_id].append(metadata)
        self._save_registry()
        
        return metadata
    
    def load_model(
        self,
        model_id: str,
        version: str = None
    ) -> tuple[Any, ModelMetadata]:
        """
        Load a model.
        
        Args:
            model_id: The model identifier
            version: Optional version to load (defaults to active version)
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_id not in self.registry:
            raise FileNotFoundError(f"Model '{model_id}' not found in registry")
        
        versions = self.registry[model_id]
        
        if version:
            # Find specific version
            metadata = next((v for v in versions if v.version == version), None)
            if not metadata:
                raise FileNotFoundError(f"Version {version} not found for model '{model_id}'")
            
            model_filename = f"{model_id}_v{version.replace('.', '_')}.joblib"
            model_path = self.versions_dir / model_filename
        else:
            # Load active version
            active_path = self.active_dir / f"{model_id}.joblib"
            meta_path = self.active_dir / f"{model_id}_metadata.json"
            
            if active_path.exists():
                model = joblib.load(active_path)
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = ModelMetadata.from_dict(json.load(f))
                else:
                    # Find active in registry
                    metadata = next((v for v in versions if v.is_active), versions[-1])
                return model, metadata
            
            # Fallback to latest version
            metadata = versions[-1]
            model_filename = f"{model_id}_v{metadata.version.replace('.', '_')}.joblib"
            model_path = self.versions_dir / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        return model, metadata
    
    def get_model_versions(self, model_id: str) -> List[ModelMetadata]:
        """Get all versions of a model."""
        return self.registry.get(model_id, [])
    
    def get_active_version(self, model_id: str) -> Optional[ModelMetadata]:
        """Get the active version of a model."""
        versions = self.registry.get(model_id, [])
        return next((v for v in versions if v.is_active), None)
    
    def set_active_version(self, model_id: str, version: str):
        """Set a specific version as active."""
        if model_id not in self.registry:
            raise FileNotFoundError(f"Model '{model_id}' not found")
        
        versions = self.registry[model_id]
        target = next((v for v in versions if v.version == version), None)
        
        if not target:
            raise FileNotFoundError(f"Version {version} not found")
        
        # Deactivate all, activate target
        for v in versions:
            v.is_active = False
        target.is_active = True
        
        # Copy to active directory
        model_filename = f"{model_id}_v{version.replace('.', '_')}.joblib"
        model_path = self.versions_dir / model_filename
        
        if model_path.exists():
            active_path = self.active_dir / f"{model_id}.joblib"
            joblib.dump(joblib.load(model_path), active_path)
            
            meta_path = self.active_dir / f"{model_id}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(target.to_dict(), f, indent=2)
        
        self._save_registry()
    
    def delete_version(self, model_id: str, version: str):
        """Delete a specific model version."""
        if model_id not in self.registry:
            return
        
        versions = self.registry[model_id]
        target = next((v for v in versions if v.version == version), None)
        
        if not target:
            return
        
        if target.is_active:
            raise ValueError("Cannot delete the active version")
        
        # Remove file
        model_filename = f"{model_id}_v{version.replace('.', '_')}.joblib"
        model_path = self.versions_dir / model_filename
        if model_path.exists():
            model_path.unlink()
        
        # Remove from registry
        self.registry[model_id] = [v for v in versions if v.version != version]
        self._save_registry()
    
    def list_models(self) -> Dict[str, ModelMetadata]:
        """List all models with their active versions."""
        result = {}
        for model_id, versions in self.registry.items():
            active = next((v for v in versions if v.is_active), versions[-1] if versions else None)
            if active:
                result[model_id] = active
        return result
    
    def list_all_models(self) -> Dict[str, List[ModelMetadata]]:
        """List all models with all their versions."""
        return {model_id: versions for model_id, versions in self.registry.items() if versions}
    
    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists."""
        return model_id in self.registry and len(self.registry[model_id]) > 0
    
    def get_model_summary(self, model_id: str) -> dict:
        """Get a summary of a model's history."""
        if model_id not in self.registry:
            return {}
        
        versions = self.registry[model_id]
        active = next((v for v in versions if v.is_active), None)
        
        return {
            "model_id": model_id,
            "total_versions": len(versions),
            "active_version": active.version if active else None,
            "created_at": versions[0].created_at if versions else None,
            "last_updated": versions[-1].created_at if versions else None,
            "model_type": active.model_type if active else None,
            "latest_metrics": active.metrics if active else {}
        }


# Global model manager instance
_model_manager = None


def get_model_manager(models_dir: str = "models") -> ModelManager:
    """Get or create the global model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(models_dir)
    return _model_manager

