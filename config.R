INPUT_FILE <- file.path("data", "input.csv")

TEMPORAL_FILE <- file.path("data", "temporal.csv")

COMPONENT_FILE <- file.path("data", "benchmarks.csv")

OUTCOME_COLUMN <- "outcome"

EXCLUDED_COLUMNS <- character()

CATEGORICAL_COLUMNS <- character()

CATEGORICAL_LEVELS <- list()

REFERENCE_COLUMN <- "predictor_1"

ADDITIONAL_BENCHMARK_COLUMNS <- c("predictor_2", "predictor_3")

FEATURE_METADATA_FILE <- file.path("data", "feature_metadata.csv")

MAIN_DIR <- file.path("outputs", "main")

LOGISTIC_DIR <- file.path("outputs", "logistic")

BENCHMARK_DIR <- file.path("outputs", "benchmarks")

COMPONENT_DIR <- file.path("outputs", "additional_benchmarks")

SHAP_DIR <- file.path("outputs", "shap")

TEMPORAL_DIR <- file.path("outputs", "temporal")

REFIT_DIR <- file.path("outputs", "refit")

APP_BUNDLE <- file.path("app", "model_bundle.rds")

SEED_MAIN <- 42L

SEED_TEMPORAL <- 42L

TEMPORAL_END_YEAR <- NA_integer_

OUTER_K <- 5L

OUTER_R <- 5L

INNER_K <- 4L

TEMPORAL_INNER_K <- 5L

BORUTA_RUNS <- 100L

SEARCH_BUDGET <- 24L

CORES <- 1L

configure_engine <- function(data_file = INPUT_FILE, output_dir = MAIN_DIR, seed = SEED_MAIN, inner_k = INNER_K) {
    Sys.setenv(ML_DATA_FILE = data_file, ML_OUTCOME_COL = OUTCOME_COLUMN, ML_OUT_DIR = output_dir, ML_SEED = as.character(seed), 
        ML_OUTER_FOLDS = as.character(OUTER_K), ML_OUTER_REPEATS = as.character(OUTER_R), ML_INNER_FOLDS = as.character(inner_k), 
        ML_BORUTA_RUNS = as.character(BORUTA_RUNS), ML_TUNE_BUDGET = as.character(SEARCH_BUDGET), ML_CORES = as.character(CORES), 
        ML_SKIP_LASSO = "1", ML_SKIP_SHAP = "1", ML_INDEX_MODEL = "StableKernelNB", ML_EXCLUDE_PREDICTORS = paste(EXCLUDED_COLUMNS, 
            collapse = ","), ML_TARGET_SENSITIVITY = "0.85")
}
