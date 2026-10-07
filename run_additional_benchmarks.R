source("config.R", encoding = "UTF-8")

RESULT <- file.path(COMPONENT_DIR, "results")

LOG <- file.path(COMPONENT_DIR, "logs")

dir.create(RESULT, recursive = TRUE, showWarnings = FALSE)

dir.create(LOG, recursive = TRUE, showWarnings = FALSE)

options(stringsAsFactors = FALSE, warn = 1)

invisible(NULL)

suppressPackageStartupMessages({
    library(pROC)
    library(dplyr)
})

dat <- read.csv(COMPONENT_FILE, check.names = FALSE, fileEncoding = "UTF-8-BOM")

manifest <- read.csv(file.path(MAIN_DIR, "01_Tables", "00_seed_outer_validation_manifest.csv"))

existing_oof <- read.csv(file.path(BENCHMARK_DIR, "02_Predictions", "02_benchmark_patient_level_repeated_OOF.csv"))

existing_metrics <- read.csv(file.path(BENCHMARK_DIR, "01_Tables", "01_benchmark_metrics_with_bootstrap_CI.csv"))

existing_delta <- read.csv(file.path(BENCHMARK_DIR, "01_Tables", "03_incremental_performance_vs_reference_predictor.csv"))

stopifnot(all(dat$truth %in% c(0L, 1L)), identical(dat$row_id, seq_len(nrow(dat))))

stopifnot(length(unique(manifest$fold)) == OUTER_K * OUTER_R, all(table(manifest$row_id) == OUTER_R))

BOOT_N <- 2000L

BOOT_SEED <- (SEED_MAIN + 700000L)

model_variables <- c(ADDITIONAL_BENCHMARK_COLUMNS, REFERENCE_COLUMN)

names(model_variables) <- c(paste0("Benchmark", seq_along(ADDITIONAL_BENCHMARK_COLUMNS), "OnlyLinear"), 
    "ReferenceOnlyLinear")

fold_names <- unique(manifest$fold)

pred_list <- list()

fit_log <- list()

for (fold in fold_names) {
    valid_ids <- sort(unique(manifest$row_id[manifest$fold == fold]))
    train_ids <- setdiff(dat$row_id, valid_ids)
    stopifnot(length(intersect(train_ids, valid_ids)) == 0L)
    for (model in names(model_variables)) {
        variable <- model_variables[[model]]
        train <- data.frame(y = dat$truth[train_ids], x = as.numeric(dat[[variable]][train_ids]))
        valid <- data.frame(x = as.numeric(dat[[variable]][valid_ids]))
        training_median <- median(train$x, na.rm = TRUE)
        if (!is.finite(training_median)) 
            stop("No usable training-fold predictor values")
        training_missing <- sum(is.na(train$x))
        validation_missing <- sum(is.na(valid$x))
        train$x[is.na(train$x)] <- training_median
        valid$x[is.na(valid$x)] <- training_median
        fit <- glm(y ~ x, data = train, family = binomial())
        probability <- as.numeric(predict(fit, newdata = valid, type = "response"))
        stopifnot(all(is.finite(probability)), all(probability >= 0 & probability <= 1))
        pred_list[[length(pred_list) + 1L]] <- data.frame(row_id = valid_ids, fold = fold, model = model, 
            truth = dat$truth[valid_ids], prob = probability)
        fit_log[[length(fit_log) + 1L]] <- data.frame(fold = fold, model = model, training_n = length(train_ids), 
            validation_n = length(valid_ids), training_imputation_median = training_median, training_missing_n = training_missing, 
            validation_missing_n = validation_missing, intercept = unname(coef(fit)[1]), slope = unname(coef(fit)[2]), 
            converged = fit$converged)
    }
}

fold_pred <- bind_rows(pred_list)

patient <- summarise(group_by(fold_pred, row_id, model), truth = first(truth), prob = mean(prob), predictions_n = n(), 
    .groups = "drop")

stopifnot(all(patient$predictions_n == OUTER_R))

recomputed <- arrange(filter(patient, model == "ReferenceOnlyLinear"), row_id)

reference <- arrange(filter(existing_oof, model == "ReferenceOnlyLinear"), row_id)

stopifnot(nrow(reference) == nrow(dat), identical(reference$row_id, recomputed$row_id))

stopifnot(all(reference$truth == recomputed$truth))

reference_prediction_max_difference <- max(abs(reference$prob - recomputed$prob))

if (reference_prediction_max_difference > 1e-12) stop("Reference-model reference cannot be reproduced on the locked folds")

cat("Reference-model reference reproduced; max probability difference:", reference_prediction_max_difference, 
    "\n")

all_oof <- bind_rows(existing_oof, filter(patient, model != "ReferenceOnlyLinear"))

stopifnot(!anyDuplicated(all_oof[c("row_id", "model")]))

auc_safe <- function(y, p) as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE)))

average_precision <- function(y, p) {
    ord <- order(p, decreasing = TRUE)
    y <- y[ord]
    mean((cumsum(y)/seq_along(y))[y == 1L])
}

average_precision_tie_aware <- function(y, p) {
    z <- arrange(summarise(group_by(data.frame(y = y, p = p), p), n = n(), positives = sum(y), .groups = "drop"), 
        desc(p))
    sum((cumsum(z$positives)/cumsum(z$n)) * z$positives)/sum(y)
}

calibration_metrics <- function(y, p) {
    p <- pmin(pmax(p, 1e-08), 1 - 1e-08)
    lp <- qlogis(p)
    intercept <- unname(coef(glm(y ~ offset(lp), family = binomial()))[1])
    slope <- unname(coef(glm(y ~ lp, family = binomial()))[2])
    c(Calibration_intercept = intercept, Calibration_slope = slope)
}

metric_vector <- function(y, p) c(AUROC = auc_safe(y, p), PR_AUC = average_precision(y, p), Brier = mean((p - 
    y)^2), calibration_metrics(y, p))

set.seed(BOOT_SEED)

boot_ids <- replicate(BOOT_N, sample(seq_len(nrow(dat)), nrow(dat), replace = TRUE), simplify = FALSE)

new_models <- setdiff(names(model_variables), "ReferenceOnlyLinear")

metrics_list <- list()

delta_list <- list()

ap_audit <- list()

for (model in new_models) {
    z <- arrange(filter(patient, .data$model == .env$model), row_id)
    stopifnot(nrow(z) == nrow(dat), all(z$truth == reference$truth))
    point <- metric_vector(z$truth, z$prob)
    cat("Bootstrap confidence intervals:", model, "AUROC", point[["AUROC"]], "\n")
    boot <- vapply(boot_ids, function(idx) {
        if (length(unique(z$truth[idx])) < 2L) 
            return(rep(NA_real_, length(point)))
        metric_vector(z$truth[idx], z$prob[idx])
    }, numeric(length(point)))
    metrics_list[[model]] <- bind_rows(lapply(seq_along(point), function(j) data.frame(model = model, 
        metric = names(point)[j], estimate = unname(point[j]), lower_95 = unname(quantile(boot[j, ], 
            0.025, na.rm = TRUE)), upper_95 = unname(quantile(boot[j, ], 0.975, na.rm = TRUE)), bootstrap_n = BOOT_N)))
    delta <- auc_safe(z$truth, z$prob) - auc_safe(reference$truth, reference$prob)
    delta_boot <- vapply(boot_ids, function(idx) {
        if (length(unique(z$truth[idx])) < 2L) 
            return(NA_real_)
        auc_safe(z$truth[idx], z$prob[idx]) - auc_safe(reference$truth[idx], reference$prob[idx])
    }, numeric(1))
    delta_list[[model]] <- data.frame(model = model, reference = "ReferenceOnlyLinear", metric = "AUROC", 
        difference = delta, lower_95 = unname(quantile(delta_boot, 0.025, na.rm = TRUE)), upper_95 = unname(quantile(delta_boot, 
            0.975, na.rm = TRUE)), bootstrap_n = BOOT_N)
    ap_audit[[model]] <- data.frame(model = model, existing_routine_average_precision = average_precision(z$truth, 
        z$prob), tie_aware_point_average_precision = average_precision_tie_aware(z$truth, z$prob), point_difference = average_precision(z$truth, 
        z$prob) - average_precision_tie_aware(z$truth, z$prob), duplicate_probability_n = sum(duplicated(z$prob)))
}

new_metrics <- bind_rows(metrics_list)

new_delta <- bind_rows(delta_list)

combined_metrics <- bind_rows(existing_metrics, new_metrics)

combined_delta <- bind_rows(existing_delta, new_delta)

pair_names <- c("ReferenceOnlyLinear", new_models, "StableKernelNB")

pair_results <- list()

for (i in seq_len(length(pair_names) - 1L)) for (j in (i + 1L):length(pair_names)) {
    a <- arrange(filter(all_oof, model == pair_names[i]), row_id)
    b <- arrange(filter(all_oof, model == pair_names[j]), row_id)
    point <- auc_safe(a$truth, a$prob) - auc_safe(b$truth, b$prob)
    boots <- vapply(boot_ids, function(idx) auc_safe(a$truth[idx], a$prob[idx]) - auc_safe(b$truth[idx], 
        b$prob[idx]), numeric(1))
    pair_results[[length(pair_results) + 1L]] <- data.frame(model_a = pair_names[i], model_b = pair_names[j], 
        AUROC_difference_a_minus_b = point, lower_95 = unname(quantile(boots, 0.025, na.rm = TRUE)), 
        upper_95 = unname(quantile(boots, 0.975, na.rm = TRUE)), bootstrap_n = BOOT_N)
}

write.csv(fold_pred, file.path(RESULT, "01_component_fold_OOF_predictions.csv"), row.names = FALSE)

write.csv(all_oof, file.path(RESULT, "02_combined_patient_level_OOF_predictions.csv"), row.names = FALSE)

write.csv(bind_rows(fit_log), file.path(RESULT, "03_training_fold_fit_and_preprocessing_audit.csv"), 
    row.names = FALSE)

write.csv(combined_metrics, file.path(RESULT, "04_extended_benchmark_metrics.csv"), row.names = FALSE)

write.csv(combined_delta, file.path(RESULT, "05_extended_incremental_metrics_vs_reference_predictor.csv"), 
    row.names = FALSE)

write.csv(bind_rows(pair_results), file.path(RESULT, "06_component_pairwise_AUROC_differences.csv"), 
    row.names = FALSE)

write.csv(bind_rows(ap_audit), file.path(LOG, "average_precision_tie_diagnostic.csv"), row.names = FALSE)

write.csv(data.frame(n = nrow(dat), events = sum(dat$truth), outer_folds = OUTER_K * OUTER_R, predictions_per_patient = OUTER_R, 
    bootstrap_n = BOOT_N, bootstrap_seed = BOOT_SEED, reference_prediction_max_difference = reference_prediction_max_difference), 
    file.path(LOG, "analysis_integrity.csv"), row.names = FALSE)

capture.output(sessionInfo(), file = file.path(LOG, "sessionInfo.txt"))

writeLines("COMPLETED", file.path(LOG, "COMPLETED.txt"))

print(combined_metrics[combined_metrics$model %in% new_models & combined_metrics$metric %in% c("AUROC", 
    "PR_AUC", "Brier"), ], row.names = FALSE)
