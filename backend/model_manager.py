"""
Model manager for loading and caching models at startup.
"""

import os
import sys
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cbm.models.loaders import load_model_and_tokenizer


def setup_module_aliases():
    """Set up module aliases to redirect run_cebab imports to cbm.models."""
    import cbm.models.template_models as template_models
    import cbm.models.cbm_models as cbm_models
    
    # Create aliases in sys.modules
    sys.modules['run_cebab'] = cbm_models
    sys.modules['run_cebab.cbm_template_models'] = template_models
    sys.modules['run_cebab.cbm_models'] = cbm_models


def cleanup_module_aliases():
    """Clean up module aliases after loading."""
    aliases_to_remove = ['run_cebab', 'run_cebab.cbm_template_models', 'run_cebab.cbm_models']
    for alias in aliases_to_remove:
        if alias in sys.modules:
            del sys.modules[alias]


class ModelManager:
    """Singleton class for managing model loading and caching."""
    
    _instance = None
    _models = {}
    _tokenizers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.initialized = True
    
    def load_trained_model(self, model_name: str, mode: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load trained model and head from saved files."""
        # Determine file paths based on mode
        if mode == 'standard':
            model_file = f"{model_name}_model_standard.pth"
            head_file = f"{model_name}_classifier_standard.pth"
        elif mode == 'joint':
            model_file = f"{model_name}_joint.pth"
            head_file = f"{model_name}_ModelXtoCtoY_layer_joint.pth"
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # Load from saved_models/<dataset>/<model_name> directory
        dataset_name = os.getenv("CBM_DATASET", "essay")
        base_dir = project_root / "saved_models" / dataset_name / model_name

        # Fallback: auto-discover dataset directory that contains this model if default doesn't exist
        if not base_dir.exists():
            saved_models_root = project_root / "saved_models"
            if saved_models_root.exists():
                for possible_dataset_dir in saved_models_root.iterdir():
                    if possible_dataset_dir.is_dir():
                        candidate_dir = possible_dataset_dir / model_name
                        if candidate_dir.exists():
                            base_dir = candidate_dir
                            break

        model_path = base_dir / model_file
        head_path = base_dir / head_file
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not head_path.exists():
            raise FileNotFoundError(f"Head file not found: {head_path}")
        
        try:
            # Set up module aliases for run_cebab imports
            setup_module_aliases()
            
            # Load models with proper device mapping
            model = torch.load(model_path, weights_only=False, map_location=self.device)
            head = torch.load(head_path, weights_only=False, map_location=self.device)
            
            # Clean up module aliases
            cleanup_module_aliases()
            
            # Move to device
            model = model.to(self.device)
            head = head.to(self.device)
            
            # Fix configuration compatibility issues
            if hasattr(model, 'config'):
                # Add missing attributes to config if they don't exist
                if not hasattr(model.config, '_output_attentions'):
                    model.config._output_attentions = False
                if not hasattr(model.config, '_output_hidden_states'):
                    model.config._output_hidden_states = False
                if not hasattr(model.config, 'output_attentions'):
                    model.config.output_attentions = False
                if not hasattr(model.config, 'output_hidden_states'):
                    model.config.output_hidden_states = False
            
            model.eval()
            head.eval()
            
            # Attach dataset configuration to head for inference
            if mode == 'joint':
                # Determine dataset type based on head structure
                # For now, assume Essay dataset (most common case)
                # TODO: Make this more robust by detecting from model structure
                head.dataset_config = {
                    'final_label': ['score'],
                    'final_label_vals': [0, 1, 2, 3, 4, 5],
                    'concepts': ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR'],
                    'concept_vals': [0, 1, 2, 3, 4]
                }
            
            return model, head
        except Exception as e:
            print(f"Warning: Could not load saved model due to compatibility issues: {e}")
            print("Falling back to pretrained model (untrained head)...")
            return self.load_pretrained_model(model_name)
    
    def load_pretrained_model(self, model_name: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load pretrained model with untrained head for demonstration."""
        model, tokenizer, hidden_size = load_model_and_tokenizer(model_name)
        
        # Determine number of classes based on model name or context
        # For essay dataset, use 6 classes (0-5 scoring); for restaurant dataset, use 5 classes
        num_classes = 6  # Default to essay dataset (6-class: 0-5 scoring)
        
        # Create a simple untrained classification head
        head = torch.nn.Linear(hidden_size, num_classes)
        
        # Initialize weights for better random predictions
        torch.nn.init.xavier_uniform_(head.weight)
        torch.nn.init.zeros_(head.bias)
        
        model.to(self.device)
        head.to(self.device)
        model.eval()
        head.eval()
        
        return model, head
    
    def get_model(self, model_name: str, mode: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Get cached model or load if not cached."""
        key = f"{model_name}_{mode}"
        
        if key not in self._models:
            try:
                model, head = self.load_trained_model(model_name, mode)
                # Check if the head has the correct output size
                output_size = None
                if hasattr(head, 'out_features'):
                    output_size = head.out_features
                elif hasattr(head, '__len__') and len(head) > 0:
                    # For Sequential containers, check the last layer
                    last_layer = head[-1]
                    if hasattr(last_layer, 'out_features'):
                        output_size = last_layer.out_features
                
                if output_size is not None and output_size not in [2, 5, 6]:
                    print(f"Warning: Loaded model head has {output_size} outputs, expected 2, 5, or 6. Using pretrained model instead.")
                    model, head = self.load_pretrained_model(model_name)
                self._models[key] = (model, head)
            except FileNotFoundError:
                # Fallback to pretrained model
                model, head = self.load_pretrained_model(model_name)
                self._models[key] = (model, head)
        
        return self._models[key]
    
    def get_tokenizer(self, model_name: str):
        """Get cached tokenizer or load if not cached."""
        if model_name not in self._tokenizers:
            _, tokenizer, _ = load_model_and_tokenizer(model_name)
            self._tokenizers[model_name] = tokenizer
        
        return self._tokenizers[model_name]
    
    def load_default_models(self):
        """Load default models at startup."""
        default_models = [
            ("bert-base-uncased", "standard"),
            ("bert-base-uncased", "joint"),
        ]
        
        for model_name, mode in default_models:
            try:
                print(f"Loading {model_name} in {mode} mode...")
                # Load models only; defer tokenizer to first request to avoid slow downloads at startup
                self.get_model(model_name, mode)
                print(f"✓ {model_name} ({mode}) loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {model_name} ({mode}): {e}")
    
    def get_loaded_models(self) -> Dict[str, list]:
        """Get list of loaded models and modes."""
        loaded = {}
        for key in self._models.keys():
            model_name, mode = key.rsplit('_', 1)
            if model_name not in loaded:
                loaded[model_name] = []
            loaded[model_name].append(mode)
        return loaded
    
    def is_model_loaded(self, model_name: str, mode: str) -> bool:
        """Check if a specific model and mode is loaded."""
        key = f"{model_name}_{mode}"
        return key in self._models


# Global model manager instance
model_manager = ModelManager()
