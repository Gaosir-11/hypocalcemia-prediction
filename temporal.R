options(stringsAsFactors = FALSE, warn = 1)

raw_temporal <- read.csv(DATA_FILE, check.names = FALSE, fileEncoding = "UTF-8-BOM")

calendar_year <- as.integer(raw_temporal$calendar_year)

if (length(calendar_year) != nrow(dat) || anyNA(calendar_year)) {
    stop("Calendar year could not be recovered for every analysis record.")
}

metadata_cols <- c("row_id", "calendar_year", "calendar_period")

dat <- dat[, setdiff(names(dat), c(metadata_cols, EXCLUDED_COLUMNS)), drop = FALSE]

predictor_cols <- setdiff(names(dat), c(OUTCOME_COL, ".row_id"))

factor_cols <- intersect(factor_cols, predictor_cols)

numeric_cols <- intersect(numeric_cols, predictor_cols)

development_end_year <- as.integer(Sys.getenv("ML_TEMPORAL_DEVELOPMENT_END", unset = ""))

if (!is.finite(development_end_year)) stop("Invalid temporal cutoff.")

validation_start_year <- development_end_year + 1L

development_label <- paste0(min(calendar_year), "-", development_end_year)

validation_label <- paste0(validation_start_year, "-", max(calendar_year))

development_index <- which(calendar_year <= development_end_year)

temporal_validation_index <- which(calendar_year >= validation_start_year)

stopifnot(length(intersect(development_index, temporal_validation_index)) == 0L)

development <- dat[development_index, , drop = FALSE]

temporal_validation <- dat[temporal_validation_index, , drop = FALSE]

if (nrow(development) == 0L || nrow(temporal_validation) == 0L) {
    stop("Temporal split produced an empty cohort.")
}

if (length(unique(development[[OUTCOME_COL]])) < 2L || length(unique(temporal_validation[[OUTCOME_COL]])) < 
    2L) {
    stop("Both temporal cohorts must contain events and non-events.")
}

analysis_rng <- as.integer(Sys.getenv("ML_ANALYSIS_RNG", unset = "42"))

if (!is.finite(analysis_rng) || analysis_rng < 1L) stop("Invalid analysis RNG initialization.")

nb_only <- identical(Sys.getenv("ML_TEMPORAL_NB_ONLY", unset = "0"), "1")

make_selector_plots <- !identical(Sys.getenv("ML_TEMPORAL_MAKE_PLOTS", unset = "1"), "0")

selectors <- fit_selectors(development, analysis_rng, make_plots = make_selector_plots, plot_prefix = "01_temporal_development_")

selected_vars <- safe_feature_set(selectors$boruta, predictor_cols)

if (!length(selectors$boruta)) {
    stop("Boruta selected no predictor in the development-period cohort.")
}

write.csv(data.frame(variable = selected_vars, stringsAsFactors = FALSE), file.path(TAB_DIR, "01_temporal_development_Boruta_features.csv"), 
    row.names = FALSE, fileEncoding = "UTF-8")

write.csv(selectors$boruta_stats, file.path(TAB_DIR, "02_temporal_development_Boruta_statistics.csv"), 
    row.names = FALSE, fileEncoding = "UTF-8")

inner_control <- make_inner_control(development[[OUTCOME_COL]], analysis_rng + 100L)

inner_se <- function(fit) {
    z <- fit$resample$ROC
    z <- z[is.finite(z)]
    if (length(z) < 2L) 
        return(NA_real_)
    stats::sd(z)/sqrt(length(z))
}

corr_candidates <- c(0.85, 0.9, 0.95, 1)

corr_fits <- vector("list", length(corr_candidates))

corr_rows <- vector("list", length(corr_candidates))

for (i in seq_along(corr_candidates)) {
    cc <- corr_candidates[i]
    ans <- fit_one_model(development, development, selected_vars, "Ridge", "custom_ridge", analysis_rng + 
        200L + i, inner_control, corr_threshold = cc)
    corr_fits[[i]] <- ans
    corr_rows[[i]] <- data.frame(corr_threshold = cc, inner_AUROC = ans$inner_auc, inner_SE = inner_se(ans$fit))
}

corr_table <- dplyr::bind_rows(corr_rows)

best_i <- which.max(corr_table$inner_AUROC)

allowance <- corr_table$inner_SE[best_i]

if (!is.finite(allowance)) allowance <- 0.005

eligible <- corr_table[corr_table$inner_AUROC >= corr_table$inner_AUROC[best_i] - max(allowance, 0.005), 
    , drop = FALSE]

selected_corr <- min(eligible$corr_threshold)

corr_table$selected <- corr_table$corr_threshold == selected_corr

write.csv(corr_table, file.path(TAB_DIR, "03_development_inner_correlation_selection.csv"), row.names = FALSE)

fit_temporal_model <- function(label, implementation, vars, offset) {
    fit_one_model(development, temporal_validation, vars, label, implementation, analysis_rng + offset, 
        inner_control, corr_threshold = selected_corr)
}

model_objects <- list(StableKernelNB = fit_temporal_model("StableKernelNB", "custom_klar_nb", selected_vars, 
    1000L))

truth <- as.integer(temporal_validation[[OUTCOME_COL]] == "Yes")

predictions <- dplyr::bind_rows(lapply(names(model_objects), function(label) {
    data.frame(row_id = temporal_validation$.row_id, calendar_year = calendar_year[temporal_validation_index], 
        truth = truth, model = label, probability = model_objects[[label]]$prob, stringsAsFactors = FALSE)
}))

write.csv(predictions, file.path(PRED_DIR, "01_temporal_validation_predictions.csv"), row.names = FALSE)

average_precision <- function(y, p) {
    ok <- is.finite(p) & !is.na(y)
    y <- y[ok]
    p <- p[ok]
    ord <- order(p, decreasing = TRUE)
    y <- y[ord]
    if (!sum(y == 1L)) 
        return(NA_real_)
    precision <- cumsum(y == 1L)/seq_along(y)
    sum(precision[y == 1L])/sum(y == 1L)
}

calibration_metrics <- function(y, p) {
    p <- pmin(pmax(p, 1e-06), 1 - 1e-06)
    lp <- qlogis(p)
    cil <- tryCatch(unname(stats::coef(stats::glm(y ~ 1 + offset(lp), family = binomial()))[1]), error = function(e) NA_real_)
    fit <- tryCatch(stats::glm(y ~ lp, family = binomial()), error = function(e) NULL)
    c(calibration_intercept = cil, calibration_slope = if (is.null(fit)) NA_real_ else unname(stats::coef(fit)[2]))
}

metric_rows <- lapply(names(model_objects), function(label) {
    p <- model_objects[[label]]$prob
    roc_obj <- pROC::roc(truth, p, direction = "<", quiet = TRUE)
    ci <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))
    cal <- calibration_metrics(truth, p)
    data.frame(model = label, development_inner_AUROC = model_objects[[label]]$inner_auc, temporal_AUROC = as.numeric(pROC::auc(roc_obj)), 
        temporal_AUROC_CI_low = ci[1], temporal_AUROC_CI_high = ci[3], temporal_PR_AUC = average_precision(truth, 
            p), temporal_Brier = mean((truth - p)^2), calibration_intercept = cal["calibration_intercept"], 
        calibration_slope = cal["calibration_slope"], stringsAsFactors = FALSE)
})

metrics <- dplyr::bind_rows(metric_rows)

write.csv(metrics, file.path(TAB_DIR, "04_temporal_validation_model_metrics.csv"), row.names = FALSE)

tuning <- dplyr::bind_rows(lapply(names(model_objects), function(label) {
    bt <- model_objects[[label]]$best_tune
    bt$model <- label
    bt$selected_correlation_threshold <- selected_corr
    bt$development_inner_AUROC <- model_objects[[label]]$inner_auc
    bt
}))

write.csv(tuning, file.path(TAB_DIR, "05_temporal_validation_hyperparameters.csv"), row.names = FALSE)

saveRDS(list(selected_features = selected_vars, selected_correlation_threshold = selected_corr, models = lapply(model_objects, 
    `[[`, "fit")), file.path(RDS_DIR, "temporal_models.rds"))

capture.output(sessionInfo(), file = file.path(LOG_DIR, "sessionInfo.txt"))

try(parallel::stopCluster(cl), silent = TRUE)

foreach::registerDoSEQ()
