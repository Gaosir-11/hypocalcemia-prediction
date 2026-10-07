source("config.R", encoding = "UTF-8")

run_temporal <- function() {
    if (!is.finite(TEMPORAL_END_YEAR)) 
        stop("Set TEMPORAL_END_YEAR in config.R before running.")
    configure_engine(TEMPORAL_FILE, TEMPORAL_DIR, seed = SEED_TEMPORAL, inner_k = TEMPORAL_INNER_K)
    Sys.setenv(ML_ANALYSIS_RNG = as.character(SEED_TEMPORAL), ML_TEMPORAL_DEVELOPMENT_END = as.character(TEMPORAL_END_YEAR), 
        ML_TEMPORAL_NB_ONLY = "1", ML_TEMPORAL_MAKE_PLOTS = "0")
    source(file.path("R", "engine.R"), local = TRUE, encoding = "UTF-8")
    on.exit({
        try(parallel::stopCluster(cl), silent = TRUE)
        foreach::registerDoSEQ()
    }, add = TRUE)
    source(file.path("R", "temporal.R"), local = TRUE, encoding = "UTF-8")
}

run_temporal()
