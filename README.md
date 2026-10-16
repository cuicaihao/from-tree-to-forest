# From a Single Tree to a Forest: Companion Experiments

English | [简体中文](README.zh.md)

![A single decision tree evolving into an interconnected forest, representing the progression from individual trees to ensemble learning.](feature.webp)

This repository contains the reproducible companion materials for the article [“From a Single Tree to a Forest: Mathematical Principles, Algorithmic Evolution, and Engineering Practices of Tree Models”](https://cuicaihao.github.io/en/posts/2026-10-17-from-a-single-tree-to-a-forest-mathematical-principles-algorithmic-evolution-and-engineering-practices-of-tree-models/), including its synthetic datasets, data-generation script, and experimental notebooks.

The project follows two examples used throughout the article:

- Case A: bank transaction fraud classification, used to explain classification trees, class imbalance, gradients and Hessians, and categorical-feature handling;
- Case B: retail sales regression, used to explain regression trees, residuals, split gain, chronological partitioning, and forecast evaluation.

All data are generated from teaching rules and fixed random seeds. They do not come from a real bank or retailer and do not represent real-world industry distributions or model performance.

## Repository contents

| File | Purpose |
|---|---|
| `case-a-fraud-teaching.csv` | Eight deliberately designed transactions, S1–S8, for hand calculations in the article |
| `case-a-fraud.csv` | 6,000 synthetic transactions with stratified training, validation, and test partitions |
| `case-a-fraud.ipynb` | Fraud-classification experiments: calculation checks, ordered-statistic illustration, class imbalance, and comparisons of three boosting libraries |
| `case-b-retail-teaching.csv` | Eight deliberately designed sales records, D1–D8, for hand calculations in the article |
| `case-b-retail.csv` | A 4,800-row synthetic panel covering 20 products, two stores, and 120 days |
| `case-b-retail.ipynb` | Retail-regression experiments: split calculations, histogram illustration, chronological partitions, and comparisons of three boosting libraries |
| `generate_tree_cases.py` | Deterministically generates the four CSV files or verifies their current contents byte for byte |

The teaching tables and full experimental datasets serve different purposes. S1–S8 and D1–D8 exist only to make the mathematical derivations hand-checkable; the full datasets are used only for software training and evaluation. A gain calculated from eight teaching rows must not be used to explain full test-set metrics.

## Quick start

Install [uv](https://docs.astral.sh/uv/) first, then create the Python 3.12 project environment and install the locked dependencies:

```bash
uv sync
```

This creates a local `.venv`. Subsequent `uv run` commands automatically use that environment.

Enable the repository's pre-commit hook once after cloning:

```bash
uv run pre-commit install
```

The hook runs `nbstripout` on staged `.ipynb` files, removing cell outputs and execution counts before they are committed. To check every tracked file manually, run `uv run pre-commit run --all-files`.

Verify that the four fixed CSV files can be reproduced exactly:

```bash
uv run python generate_tree_cases.py --check
```

A successful run prints four `Verified` messages with 8, 6,000, 8, and 4,800 rows, respectively.

Run the fraud-classification notebook:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --output case-a-executed.ipynb --output-dir /private/tmp \
  case-a-fraud.ipynb
```

For the retail-regression notebook, replace the two occurrences of `case-a` at the end of the command with `case-b`:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --output case-b-executed.ipynb --output-dir /private/tmp \
  case-b-retail.ipynb
```

Executed notebooks are written to `/private/tmp`, leaving the source notebooks unchanged. Assertions in each notebook verify its CSV files, teaching statistics, full sample counts, and partition boundaries.

## Data-design boundaries

### Case A: fraud classification

- The teaching table contains four legitimate and four fraudulent transactions.
- The full dataset contains 6,000 transactions: 3,600 training, 1,200 validation, and 1,200 test records.
- Inputs include transaction amount, distance, merchant type, and merchant ID. The target `y` is observed later and must not be used as a prediction feature.
- Fraud labels in the full dataset are randomly drawn from specified probabilities. A single synthetic experiment cannot establish a general ranking of model libraries.

### Case B: retail sales regression

- Discount, `lag7`, and sales values in the teaching table are fixed inputs designed for hand calculations.
- In the full dataset, the first seven days form the history window, days 7–79 are used for training, days 80–99 for validation, and days 100–119 for testing.
- `lag7` is calculated by applying `shift(1)` to sales before taking the seven-day rolling mean, preventing the prediction day's sales from leaking into model inputs.
- The experiment assumes sufficient inventory, so the target `y` is fully observed sales. It does not simulate demand censoring caused by stockouts.
- The setup makes a next-day prediction after each daily close. Its fixed chronological holdout must not be described as multi-origin rolling backtesting.

## Regenerating the data

The `--check` option compares regenerated content with the current CSV files without overwriting them. To intentionally rebuild all four datasets, remove that option:

```bash
uv run python generate_tree_cases.py
```

This command overwrites all four CSV files. Save or commit any changes that you need to retain before running it.

## Suggested reading path

Read the article's case descriptions and hand calculations before opening the corresponding notebook:

1. Use the teaching CSV to map every sample, gradient, Hessian, and leaf statistic back to the formulas.
2. Use the full CSV to inspect the training, validation, and testing workflow.
3. When comparing XGBoost, LightGBM, and CatBoost, focus on input semantics, categorical handling, weighting, and evaluation definitions instead of treating one synthetic run as a framework leaderboard.
