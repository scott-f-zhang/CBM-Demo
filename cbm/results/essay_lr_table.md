### Essay Learning Rates (from LR Finder)

Models | Baseline LR | EssayCBM LR
--- | --- | ---
BERT | 2e-05 | 5e-05
RoBERTa | 1e-05 | 5e-06
GPT2 | 5e-05 | 5e-04
LSTM | 1e-04 | 1e-03

Notes:
- Source: `cbm/tests/test_results/lr_finder_results.csv` (`best_lr` per pipeline).
- The loader file `cbm/lr_rate/essay_lr_rate.csv` currently lists only BERT/RoBERTa (1e-05) for general use.
- Test scripts also support two presets: `dataset_optimal` and `universal`; LR finder values above reflect empirical best per pipeline on Essay.

