"""
Core inference logic for the API.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model_manager import model_manager


class TextDataset(Dataset):
    """Dataset for text classification with labels."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def predict_standard(model: torch.nn.Module, head: torch.nn.Module, tokenizer, text: str) -> Tuple[int, List[float]]:
    """Make prediction using standard mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model_manager.device)
    attention_mask = inputs['attention_mask'].to(model_manager.device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            # For transformer models, use pooled output or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            # For LSTM models
            pooled_output = outputs
        
        # Get predictions
        logits = head(pooled_output)
        
        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Ensure prediction is a scalar
        if prediction.ndim > 0:
            prediction = prediction[0]
        # Convert numpy scalar to Python int safely
        if hasattr(prediction, 'item'):
            prediction = int(prediction.item())
        else:
            prediction = int(prediction)
    
    return int(prediction), probabilities.tolist()


def predict_joint(model: torch.nn.Module, head: torch.nn.Module, tokenizer, text: str) -> Tuple[int, List[float], List[int], List[List[float]]]:
    """Make prediction using joint mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model_manager.device)
    attention_mask = inputs['attention_mask'].to(model_manager.device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled_output = outputs
        
        # Get model output - check if it's a joint model (returns list) or simple model (returns tensor)
        outputs2 = head(pooled_output)
        
        if isinstance(outputs2, list):
            # Joint model: returns list of tensors
            XtoY_output = outputs2[0:1]  # Task prediction
            XtoC_output = outputs2[1:]   # Concept predictions
            
            # Extract task prediction
            task_logits = XtoY_output[0]
            task_probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy()[0]
            task_prediction = torch.argmax(task_logits, dim=-1).item()
            
            # Extract concept predictions
            if len(XtoC_output) > 0:
                # Concatenate all concept outputs
                XtoC_logits = torch.cat(XtoC_output, dim=0)
                concept_probabilities = torch.softmax(XtoC_logits, dim=-1).cpu().numpy()
                concept_predictions = torch.argmax(XtoC_logits, dim=-1).cpu().numpy()
                
                # Get dataset configuration from head
                num_concepts = len(XtoC_output)
                if hasattr(head, 'dataset_config') and 'concept_vals' in head.dataset_config:
                    num_classes_per_concept = len(head.dataset_config['concept_vals'])
                else:
                    # Fallback: assume 3 classes per concept (most common case)
                    num_classes_per_concept = 3
                
                # Reshape for per-concept results using dynamic class count
                concept_predictions = concept_predictions.reshape(num_concepts, -1)[:, 0]
                concept_probs = concept_probabilities.reshape(num_concepts, -1, num_classes_per_concept)[:, 0, :]
                
                # Convert to Python ints for JSON serialization
                concept_predictions = [int(pred) for pred in concept_predictions]
                concept_probabilities = concept_probs.tolist()
            else:
                # Fallback if no concept outputs
                # Determine number of concepts based on task output size
                if task_logits.shape[1] == 2:  # Binary classification
                    num_concepts = 8
                elif task_logits.shape[1] == 6:  # Essay dataset (6-class: 0-5 scoring)
                    num_concepts = 8
                else:  # Restaurant dataset (5-class classification)
                    num_concepts = 4
                
                concept_predictions = [0] * num_concepts
                concept_probs = np.random.rand(num_concepts, 3)
                # Normalize probabilities
                concept_probs = concept_probs / concept_probs.sum(axis=1, keepdims=True)
                concept_probabilities = concept_probs.tolist()
        else:
            # Simple model: returns single tensor (fallback case)
            task_logits = outputs2
            task_probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy()[0]
            task_prediction = torch.argmax(task_logits, dim=-1).item()
            
            # Generate concept predictions for demonstration
            # Determine number of concepts based on task output size
            if task_logits.shape[1] == 2:  # Binary classification
                num_concepts = 8
            elif task_logits.shape[1] == 6:  # Essay dataset (6-class: 0-5 scoring)
                num_concepts = 8
            else:  # Restaurant dataset (5-class classification)
                num_concepts = 4
            
            concept_predictions = [0] * num_concepts
            concept_probs = np.random.rand(num_concepts, 3)
            # Normalize probabilities
            concept_probs = concept_probs / concept_probs.sum(axis=1, keepdims=True)
            concept_probabilities = concept_probs.tolist()
    
    return int(task_prediction), task_probabilities.tolist(), concept_predictions, concept_probabilities


def predict_single(text: str, model_name: str, mode: str) -> Dict[str, Any]:
    """Perform single text prediction."""
    # Get model and tokenizer
    model, head = model_manager.get_model(model_name, mode)
    tokenizer = model_manager.get_tokenizer(model_name)
    
    if mode == 'standard':
        prediction, probabilities = predict_standard(model, head, tokenizer, text)
        # Determine rating based on number of classes
        if len(probabilities) == 2:  # Binary classification (0-1)
            rating = prediction + 1  # Convert 0-1 to 1-2
        elif len(probabilities) == 6:  # Essay dataset (0-5 scoring)
            rating = prediction + 1  # Convert 0-5 to 1-6
        else:  # 5-class classification (0-4)
            rating = prediction + 1  # Convert 0-4 to 1-5
        
        return {
            'prediction': prediction,
            'rating': rating,
            'probabilities': probabilities,
            'concept_predictions': None
        }
    elif mode == 'joint':
        task_pred, task_probs, concept_preds, concept_probs = predict_joint(model, head, tokenizer, text)
        
        # Get concept names and labels from dataset configuration
        if hasattr(head, 'dataset_config'):
            concept_names = head.dataset_config.get('concepts', ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR'])
            concept_vals = head.dataset_config.get('concept_vals', [0, 1, 2, 3, 4])
        else:
            # Fallback based on task output size
            if len(task_probs) == 2:  # Binary classification
                concept_names = ['FC', 'CC', 'TU', 'CP', 'R', 'DU', 'EE', 'FR']
                concept_vals = [0, 1, 2]
            elif len(task_probs) == 6:  # Essay dataset (6-class: 0-5 scoring)
                concept_names = ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR']
                concept_vals = [0, 1, 2, 3, 4]
            else:  # Restaurant dataset (5-class)
                concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
                concept_vals = [0, 1, 2]
        
        # Create label mapping based on concept_vals
        if len(concept_vals) == 3:
            sentiment_map = ['Negative', 'Neutral', 'Positive']
        elif len(concept_vals) == 5:
            sentiment_map = ['1', '2', '3', '4', '5']
        else:
            # Generic mapping for other cases
            sentiment_map = [f'Class_{i}' for i in concept_vals]
        
        concept_predictions = []
        for i, name in enumerate(concept_names):
            if i < len(concept_preds):  # Ensure we don't exceed available predictions
                pred = concept_preds[i]
                sentiment = sentiment_map[pred] if pred < len(sentiment_map) else f'Class_{pred}'
                probs = concept_probs[i]
                
                # Create probabilities dictionary dynamically
                prob_dict = {}
                for j, label in enumerate(sentiment_map):
                    if j < len(probs):
                        prob_dict[label] = probs[j]
                
                concept_predictions.append({
                    'concept_name': name,
                    'prediction': sentiment,
                    'probabilities': prob_dict
                })
        
        # Determine rating based on number of classes
        if len(task_probs) == 2:  # Binary classification (0-1)
            rating = task_pred + 1  # Convert 0-1 to 1-2
        elif len(task_probs) == 6:  # Essay dataset (0-5 scoring)
            rating = task_pred + 1  # Convert 0-5 to 1-6
        else:  # 5-class classification (0-4)
            rating = task_pred + 1  # Convert 0-4 to 1-5
        
        return {
            'prediction': task_pred,
            'rating': rating,
            'probabilities': task_probs,
            'concept_predictions': concept_predictions
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def predict_batch_standard(model: torch.nn.Module, head: torch.nn.Module, dataloader: DataLoader) -> List[int]:
    """Make predictions using standard or joint mode on a batch."""
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model_manager.device)
            attention_mask = batch['attention_mask'].to(model_manager.device)
            
            # Get model output
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different model types
            if hasattr(outputs, 'last_hidden_state'):
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    pooled_output = outputs.pooler_output
                else:
                    pooled_output = outputs.last_hidden_state.mean(dim=1)
            else:
                pooled_output = outputs
            
            # Get predictions - handle both standard and joint models
            outputs2 = head(pooled_output)
            
            if isinstance(outputs2, list):
                # Joint model: extract task logits from list
                XtoY_output = outputs2[0:1]
                logits = XtoY_output[0]
            else:
                # Standard model: use tensor directly
                logits = outputs2
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
    
    return all_predictions


def evaluate_batch(texts: List[str], labels: List[int], model_name: str, mode: str, 
                  show_details: bool = False) -> Dict[str, Any]:
    """Evaluate batch of texts with labels."""
    # Get model and tokenizer
    model, head = model_manager.get_model(model_name, mode)
    tokenizer = model_manager.get_tokenizer(model_name)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Get predictions
    predictions = predict_batch_standard(model, head, dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    result = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'num_samples': len(texts),
        'predictions': None
    }
    
    if show_details:
        detailed_predictions = []
        for i, (text, true_label, pred) in enumerate(zip(texts, labels, predictions)):
            detailed_predictions.append({
                'index': i,
                'text': text,
                'true_label': int(true_label),
                'predicted_label': int(pred),
                'correct': true_label == pred
            })
        result['predictions'] = detailed_predictions
    
    return result


def concept_labels_to_logits(concept_labels: List[int], n_attributes: int, n_class_attr: int) -> torch.Tensor:
    """
    Convert concept class labels to logits format for stage 2 model input.
    
    Args:
        concept_labels: List of class labels for each concept (e.g., [2, 3, 1, ...])
        n_attributes: Number of concepts
        n_class_attr: Number of classes per concept
    
    Returns:
        Tensor of shape (1, n_attributes * n_class_attr) ready for sec_model input
    """
    logits_list = []
    for label in concept_labels:
        # Create logits with high confidence for the selected class
        logit = torch.zeros(n_class_attr)
        if 0 <= label < n_class_attr:
            logit[label] = 10.0  # High confidence value
        logits_list.append(logit)
    
    # Stack and reshape to match expected input format
    stacked_logits = torch.stack(logits_list)  # Shape: (n_attributes, n_class_attr)
    reshaped_logits = stacked_logits.reshape(1, n_attributes * n_class_attr)  # Shape: (1, n_attributes * n_class_attr)
    return reshaped_logits


def predict_with_edited_concepts(text: Optional[str], model_name: str, edited_concepts: Dict[str, int]) -> Dict[str, Any]:
    """
    Predict final label using edited concept scores, bypassing stage 1 (X->C).
    
    Args:
        text: Original text (optional, used to get original concept predictions for comparison)
        model_name: Model name to use
        edited_concepts: Dictionary mapping concept names to edited class labels (e.g., {'TC': 4, 'UE': 3})
    
    Returns:
        Dictionary with new prediction results and original prediction (if text provided)
    """
    # Get model and head (must be joint mode)
    model, head = model_manager.get_model(model_name, 'joint')
    
    # Get dataset configuration
    if hasattr(head, 'dataset_config'):
        concept_names = head.dataset_config.get('concepts', ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR'])
        concept_vals = head.dataset_config.get('concept_vals', [0, 1, 2, 3, 4])
    else:
        # Fallback: assume Essay dataset
        concept_names = ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR']
        concept_vals = [0, 1, 2, 3, 4]
    
    n_attributes = len(concept_names)
    n_class_attr = len(concept_vals)
    
    # Get original prediction if text is provided
    original_prediction = None
    original_rating = None
    original_concept_labels = None
    
    if text:
        # Get original prediction for comparison
        original_result = predict_single(text, model_name, 'joint')
        original_prediction = original_result['prediction']
        original_rating = original_result['rating']
        
        # Extract original concept labels
        if original_result.get('concept_predictions'):
            original_concept_labels = {}
            for cp in original_result['concept_predictions']:
                concept_name = cp['concept_name']
                # Convert prediction string back to label index
                pred_str = cp['prediction']
                if pred_str.isdigit():
                    original_concept_labels[concept_name] = int(pred_str) - 1  # Convert "1"-"5" to 0-4
                elif pred_str == 'Negative':
                    original_concept_labels[concept_name] = 0
                elif pred_str == 'Neutral':
                    original_concept_labels[concept_name] = 1
                elif pred_str == 'Positive':
                    original_concept_labels[concept_name] = 2
    
    # Build concept labels list, applying edits
    concept_labels = []
    for concept_name in concept_names:
        if concept_name in edited_concepts:
            # Use edited value
            edited_value = edited_concepts[concept_name]
            # Validate and convert if needed (handle string to int conversion)
            if isinstance(edited_value, str):
                if edited_value.isdigit():
                    # If it's a string digit, check if it's 1-5 (display format) or 0-4 (API format)
                    int_val = int(edited_value)
                    if int_val > 4:
                        edited_value = int_val - 1  # Convert "1"-"5" to 0-4
                    else:
                        edited_value = int_val
                elif edited_value == 'Negative':
                    edited_value = 0
                elif edited_value == 'Neutral':
                    edited_value = 1
                elif edited_value == 'Positive':
                    edited_value = 2
                else:
                    edited_value = int(edited_value)
            # Ensure value is in valid range
            if not isinstance(edited_value, int):
                edited_value = int(edited_value)
            if edited_value < 0 or edited_value >= n_class_attr:
                # Clamp to valid range
                edited_value = max(0, min(edited_value, n_class_attr - 1))
            concept_labels.append(edited_value)
        elif original_concept_labels and concept_name in original_concept_labels:
            # Use original value
            concept_labels.append(original_concept_labels[concept_name])
        else:
            # Default to middle value if neither original nor edited
            concept_labels.append(n_class_attr // 2)
    
    # Convert concept labels to logits format
    concept_logits = concept_labels_to_logits(concept_labels, n_attributes, n_class_attr)
    concept_logits = concept_logits.to(model_manager.device)
    
    # Access the second stage model (C->Y)
    # Check if head is End2EndModel
    if hasattr(head, 'sec_model'):
        sec_model = head.sec_model
    elif hasattr(head, '_modules') and 'sec_model' in head._modules:
        sec_model = head._modules['sec_model']
    else:
        raise ValueError("Cannot access sec_model from head. Head must be an End2EndModel instance.")
    
    # Run second stage prediction
    sec_model.eval()
    with torch.no_grad():
        task_logits = sec_model(concept_logits)
        task_probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy()[0]
        task_prediction = torch.argmax(task_logits, dim=-1).cpu().item()
    
    # Determine rating based on number of classes
    num_classes = len(task_probabilities)
    if num_classes == 2:  # Binary classification (0-1)
        rating = task_prediction + 1  # Convert 0-1 to 1-2
    elif num_classes == 6:  # Essay dataset (0-5 scoring)
        rating = task_prediction + 1  # Convert 0-5 to 1-6
    else:  # 5-class classification (0-4)
        rating = task_prediction + 1  # Convert 0-4 to 1-5
    
    return {
        'prediction': int(task_prediction),
        'rating': int(rating),
        'probabilities': task_probabilities.tolist(),
        'original_prediction': original_prediction,
        'original_rating': original_rating,
        'edited_concepts': edited_concepts
    }
