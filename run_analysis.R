source("config.R", encoding = "UTF-8")

run_analysis <- function() {
    configure_engine()
    source(file.path("R", "engine.R"), local = TRUE, encoding = "UTF-8")
    on.exit({
        try(parallel::stopCluster(cl), silent = TRUE)
        foreach::registerDoSEQ()
    }, add = TRUE)
    source(file.path("R", "nested_cv.R"), local = TRUE, encoding = "UTF-8")
}

run_analysis()
