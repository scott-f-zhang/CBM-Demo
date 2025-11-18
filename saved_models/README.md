# Saved Models Documentation

This directory contains pre-trained models for the CBM (Concept Bottleneck Models) NLP project. The models are organized by backbone architecture and training methodology.

## Project Overview

This project implements Concept Bottleneck Models (CBM) for natural language processing tasks, comparing different training approaches:

- **PLMs**: Standard Pre-trained Language Models (baseline)
- **CBE-PLMs**: Concept Bottleneck Enhanced Pre-trained Language Models (joint training)

## Model Organization

Models are organized in subdirectories by backbone architecture:
- `bert-base-uncased/` - BERT-based models
- `roberta-base/` - RoBERTa-based models

## Model Files

### BERT Models (`bert-base-uncased/`)

| File Name | Training Mode | Description | Usage |
|-----------|---------------|-------------|-------|
| `bert-base-uncased_model_standard.pth` | Standard | BERT backbone trained with standard end-to-end approach | Main encoder for PLMs baseline |
| `bert-base-uncased_classifier_standard.pth` | Standard | Classification head for standard training | Task prediction head for PLMs |
| `bert-base-uncased_joint.pth` | Joint | BERT backbone trained with joint concept-task supervision | Main encoder for CBE-PLMs |
| `bert-base-uncased_ModelXtoCtoY_layer_joint.pth` | Joint | Concept bottleneck layer for joint training | X→C→Y pathway for CBE-PLMs |

### RoBERTa Models (`roberta-base/`)

| File Name | Training Mode | Description | Usage |
|-----------|---------------|-------------|-------|
| `roberta-base_model_standard.pth` | Standard | RoBERTa backbone trained with standard end-to-end approach | Main encoder for PLMs baseline |
| `roberta-base_classifier_standard.pth` | Standard | Classification head for standard training | Task prediction head for PLMs |
| `roberta-base_joint.pth` | Joint | RoBERTa backbone trained with joint concept-task supervision | Main encoder for CBE-PLMs |
| `roberta-base_ModelXtoCtoY_layer_joint.pth` | Joint | Concept bottleneck layer for joint training | X→C→Y pathway for CBE-PLMs |

## File Naming Convention

The naming convention follows the pattern:
```
{model_name}_{component}_{training_mode}.pth
```

Where:
- `model_name`: Backbone architecture (bert-base-uncased, roberta-base)
- `component`: Model component (model, classifier, ModelXtoCtoY_layer)
- `training_mode`: Training approach (standard, joint)

