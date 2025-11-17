### Essay Results (Task Acc/F1, percent)

Models | Baseline (Acc/F1) | EssayCBM (Acc/F1)
--- | --- | ---
BERT | 80.70/60.01 | 81.14/62.38
RoBERTa | 80.70/61.98 | 79.39/58.88
GPT2 | 78.07/57.81 | 78.07/57.81
LSTM | 79.39/44.25 | 79.39/44.25

### Methods (brief)
- **Dataset**: `dataset/essay/cleaned/{train,dev,test}.csv` with concept columns `[TC, UE, OC, GM, VA, SV, CTD, FR]`.
- **Splits**: `train` for training, `dev` as validation, `test` for final evaluation.
- **Task**: Essay score classification (labels 0–5). Concepts are present but not used for Baseline; jointly modeled in EssayCBM.
- **Metrics**: Accuracy and Macro-F1 on the test split, computed via `cbm/evaluation/metrics.py` (macro F1 across classes).
- **Pipelines**:
  - **Baseline (PLMs)**: `cbm.pipelines.standard.get_cbm_standard`
  - **EssayCBM (CBE-PLMs)**: `cbm.pipelines.joint.get_cbm_joint`
- **Training (defaults)**: max_len 512, batch_size 8, up to 20 epochs with early stopping; Adam optimizer. Model backbones: LSTM, GPT2, `bert-base-uncased`, `roberta-base`.
- **Learning rates (tests’ dataset-optimal setting)**:
  - LSTM: 5e-4, GPT2: 5e-5, RoBERTa: 2e-5, BERT: 2e-5.
- **Result source**: `cbm/tests/test_results/result_essay.csv` (task scores column `score_fmted`).

