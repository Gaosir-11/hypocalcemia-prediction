# Binary prediction code

Generic R code for Boruta, repeated nested cross-validation, model tuning, optional single-predictor benchmarks, chronological evaluation, OOF-SHAP and a Shiny prototype.

Use the repository root as the working directory. Software versions are in `package_versions.csv`. Edit `config.R` to configure local inputs, categorical columns, optional reference predictors, seeds and the temporal cutoff. Defaults are generic examples.

The main CSV needs a complete 0/1 outcome and predictor columns. Keep row order consistent across scripts. Exclude identifiers and information unavailable at the intended prediction time. The optional temporal CSV also needs `calendar_year`. The optional benchmark CSV needs sequential `row_id`, binary `truth` and the predictor columns configured in `config.R`.

Run each analysis in a separate R session:

```r
source("run_analysis.R")
source("run_logistic.R")
source("run_benchmarks.R")
source("run_additional_benchmarks.R")
source("run_oof_shap.R")
source("run_temporal.R")
source("build_app_model.R")
shiny::runApp("app")
```

Run the main analysis first, logistic before benchmarks, and benchmarks before additional benchmarks. OOF-SHAP uses saved outer-fold fits. The single-predictor benchmarks have no hyperparameter search.

The App uses the retained predictors in a locally generated model bundle. Optional display metadata can be supplied locally through `FEATURE_METADATA_FILE` with `model_variable`, `english_name`, `unit`, `allowed_min` and `allowed_max`; otherwise column names and unrestricted numeric inputs are used.

No data, fitted objects or generated results are included. Keep these local. The App is a research prototype, not a standalone clinical decision tool.
