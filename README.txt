# MLPiezoV2

MLPiezoV2 is a Python workflow for building surrogate models around GeoStudio slope-stability analyses driven by phreatic surface inputs. It supports model bootstrapping, dataset generation, ensemble training, active learning, workbook-driven batch runs, and downstream prediction and analysis tooling.

## What This Repo Does

This repository is meant to support a workflow like this:

1. Start from a GeoStudio model (`.gsz` or extracted `.xml`)
2. Generate a clean per-model workspace
3. Define control-point ranges for the phreatic surface
4. Run GeoStudio repeatedly to generate training data
5. Train an ensemble surrogate model to predict factor of safety (FOS)
6. Use active learning to intelligently add new samples
7. Run prediction, inverse design, sensitivity analysis, and visualization tools

## Main Workflows

### 1. Bootstrap a Model Workspace

Create a clean workspace from a single GeoStudio model:

```bash
python pipelines/bootstrap_single_model.py --model-path path/to/model.gsz

This creates a per-model workspace containing:

a generated model_config.py
a model catalog
an editable workbook of phreatic surface inputs
dataset and trained-model output folders
a copied or extracted template model
2. Generate a Dataset

Run GeoStudio repeatedly to build a dataset for either shallow or deep mode:

python pipelines/pipeline_generate_dataset.py --mode shallow --n_samples 100 --config-path path/to/model_config.py

This workflow:

samples control point elevations
applies physics validation
updates the GeoStudio XML
runs geocmd.exe
reads FOS results from GeoStudio CSV output
stores results in CSV and Parquet format
3. Train the Ensemble

Train an ensemble model on the generated dataset:

python pipelines/train_models.py --dataset path/to/dataset_dir --mode shallow --output path/to/trained_models --model-name my_model

The ensemble can include:

Ridge regression
Random forest
Gradient boosting
XGBoost
Neural network
4. Run Active Learning

Select uncertain samples, run them through GeoStudio, and append them to the dataset:

python scripts/active_learning.py --mode shallow --config-path path/to/model_config.py --model-dir path/to/trained_models --model-name my_model
5. Run the Hybrid Overnight Loop

Run dataset generation, training, and active learning in a timed loop:

python pipelines/run_hybrid_overnight.py --config-path path/to/model_config.py --mode shallow --hours 8
6. Prepare Models From a Workbook

Use an edited workbook to prepare multiple model runs, optionally executing them with GeoCMD:

python pipelines/run_models_from_inputs.py --catalog path/to/model_catalog.json --inputs path/to/phreatic_surface_inputs.xlsx --working-root output/prepared_model_runs

To also run GeoStudio:

python pipelines/run_models_from_inputs.py --catalog path/to/model_catalog.json --inputs path/to/phreatic_surface_inputs.xlsx --working-root output/prepared_model_runs --run --geocmd "C:/Program Files/Seequent/GeoStudio 2024.1/Bin/geocmd.exe"
Repository Layout
MLPiezoV2/
├── configs/
│   └── config.py
├── pipelines/
│   ├── bootstrap_single_model.py
│   ├── pipeline_generate_dataset.py
│   ├── train_models.py
│   ├── run_hybrid_overnight.py
│   └── run_models_from_inputs.py
├── scripts/
│   ├── active_learning.py
│   ├── quick_predict.py
│   ├── inverse_design.py
│   ├── dashboard.py
│   ├── sensitivity_analysis.py
│   └── visualize_results.py
├── src/
│   ├── geostudio_interface.py
│   ├── model_workflow.py
│   ├── dataset_manager.py
│   ├── data_loader.py
│   ├── ensemble.py
│   ├── ml_models.py
│   ├── model_catalog.py
│   ├── config_utils.py
│   └── xlsx_io.py
└── requirements.txt
Configuration

The repo includes a default config at:

configs/config.py

For real work, the preferred approach is:

bootstrap a model-specific workspace
edit the generated model_config.py
pass it explicitly with --config-path

Important config sections include:

CONTROL_POINTS
GEOSTUDIO
DATASET
ML_CONFIG
Requirements

Install dependencies with:

python -m pip install -r requirements.txt

Main Python dependencies include:

numpy
pandas
scipy
pydantic
pyarrow
openpyxl
scikit-learn
xgboost
matplotlib
seaborn
plotly
streamlit
pytest
External Requirements

This project assumes access to:

GeoStudio
geocmd.exe
valid GeoStudio .gsz or .xml models
appropriate model templates and output structure

The default config expects GeoStudio at:

C:/Program Files/Seequent/GeoStudio 2024.1/Bin

Update this path if your installation differs.

Outputs

Depending on the workflow, the project generates:

dataset CSV and Parquet files
training metadata
saved ensemble .pkl files
workbook exports
prepared GeoStudio working directories
run summaries
active learning plots
hybrid run status files
Typical End-to-End Flow
python pipelines/bootstrap_single_model.py --model-path path/to/model.gsz
python pipelines/pipeline_generate_dataset.py --mode shallow --n_samples 100 --config-path path/to/model_config.py
python pipelines/train_models.py --dataset path/to/dataset_dir --mode shallow --output path/to/trained_models --model-name my_model
python scripts/active_learning.py --mode shallow --config-path path/to/model_config.py --model-dir path/to/trained_models --model-name my_model
Notes
The default repository config is a generic template.
Per-model configs are the intended production path.
Large binary models, generated datasets, and trained artifacts are best kept out of version control.
This repo is focused on the Python and workflow side of the GeoStudio integration rather than storing source model assets.
Status

This repository appears to be an active working codebase for GeoStudio-driven surrogate modeling and active learning. Before production use, it is a good idea to validate the GeoStudio execution workflow, dataset integrity, and model outputs against known cases in your local environment.
