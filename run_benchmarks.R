options(stringsAsFactors = FALSE, warn = 1)

invisible(NULL)

suppressPackageStartupMessages({
    library(caret)
    library(dplyr)
    library(pROC)
    library(splines)
})

source("config.R", encoding = "UTF-8")

DATA_FILE <- INPUT_FILE

MANIFEST_FILE <- file.path(MAIN_DIR, "01_Tables", "00_seed_outer_validation_manifest.csv")

MODEL_OOF_FILE <- file.path(MAIN_DIR, "03_Predictions", "03_patient_level_repeated_OOF.csv")

LOGISTIC_OOF_FILE <- file.path(LOGISTIC_DIR, "03_Predictions", "02_patient_level_repeated_OOF_logistic.csv")

OUT_DIR <- BENCHMARK_DIR

TABLE_DIR <- file.path(OUT_DIR, "01_Tables")

PRED_DIR <- file.path(OUT_DIR, "02_Predictions")

LOG_DIR <- file.path(OUT_DIR, "03_Logs")

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(PRED_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

OUTCOME <- OUTCOME_COLUMN

REFERENCE <- REFERENCE_COLUMN

BOOT_N <- 2000L

BOOT_SEED <- (SEED_MAIN + 700000L)

dat <- read.csv(DATA_FILE, check.names = FALSE)

names(dat) <- sub("^﻿", "", names(dat), useBytes = TRUE)

stopifnot(OUTCOME %in% names(dat), REFERENCE %in% names(dat))

y_raw <- trimws(as.character(dat[[OUTCOME]]))

y_raw[y_raw %in% c("Yes", "yes")] <- "1"

y_raw[y_raw %in% c("No", "no")] <- "0"

if (anyNA(y_raw) || !all(unique(y_raw) %in% c("0", "1"))) stop("Invalid outcome coding")

dat$.y01 <- as.integer(y_raw == "1")

dat$.reference <- as.numeric(dat[[REFERENCE]])

if (anyNA(dat$.reference)) stop("Reference predictor contains missing values")

manifest <- read.csv(MANIFEST_FILE, check.names = FALSE)

fold_order <- unique(manifest$fold)

if (length(fold_order) != OUTER_K * OUTER_R) stop("Unexpected number of outer folds")

fold_predictions <- bind_rows(lapply(seq_along(fold_order), function(i) {
    fold <- fold_order[i]
    valid_idx <- sort(unique(manifest$row_id[manifest$fold == fold]))
    train_idx <- setdiff(seq_len(nrow(dat)), valid_idx)
    train <- dat[train_idx, , drop = FALSE]
    valid <- dat[valid_idx, , drop = FALSE]
    linear_fit <- glm(.y01 ~ .reference, data = train, family = binomial())
    linear_prob <- as.numeric(predict(linear_fit, newdata = valid, type = "response"))
    spline_fit <- glm(.y01 ~ ns(.reference, df = 3), data = train, family = binomial())
    spline_prob <- as.numeric(predict(spline_fit, newdata = valid, type = "response"))
    bind_rows(data.frame(row_id = valid_idx, fold = fold, model = "ReferenceOnlyLinear", truth = valid$.y01, 
        prob = linear_prob), data.frame(row_id = valid_idx, fold = fold, model = "ReferenceOnlySpline", 
        truth = valid$.y01, prob = spline_prob))
}))

patient_benchmarks <- summarise(group_by(fold_predictions, row_id, model), truth = first(truth), prob = mean(prob), 
    predictions_n = n(), .groups = "drop")

if (any(patient_benchmarks$predictions_n != OUTER_R)) stop("Repeated OOF aggregation failed")

model_oof <- select(filter(read.csv(MODEL_OOF_FILE, check.names = FALSE), model %in% c("StableKernelNB", 
    "GAM")), row_id, model, truth, prob, predictions_n)

logistic_oof <- select(read.csv(LOGISTIC_OOF_FILE, check.names = FALSE), row_id, model, truth, prob, 
    predictions_n)

all_oof <- bind_rows(patient_benchmarks, model_oof, logistic_oof)

auc_safe <- function(y, p) {
    as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE)))
}

average_precision <- function(y, p) {
    ord <- order(p, decreasing = TRUE)
    y <- y[ord]
    mean((cumsum(y)/seq_along(y))[y == 1L])
}

calibration_metrics <- function(y, p) {
    p <- pmin(pmax(p, 1e-08), 1 - 1e-08)
    lp <- qlogis(p)
    intercept <- unname(coef(glm(y ~ offset(lp), family = binomial()))[1])
    slope <- unname(coef(glm(y ~ lp, family = binomial()))[2])
    c(Calibration_intercept = intercept, Calibration_slope = slope)
}

metric_vector <- function(y, p) {
    c(AUROC = auc_safe(y, p), PR_AUC = average_precision(y, p), Brier = mean((p - y)^2), calibration_metrics(y, 
        p))
}

set.seed(BOOT_SEED)

eval_ids <- sort(unique(all_oof$row_id))

boot_ids <- replicate(BOOT_N, sample(seq_along(eval_ids), length(eval_ids), replace = TRUE), simplify = FALSE)

metric_rows <- bind_rows(lapply(split(all_oof, all_oof$model), function(z) {
    z <- z[match(eval_ids, z$row_id), , drop = FALSE]
    point <- metric_vector(z$truth, z$prob)
    boot <- vapply(boot_ids, function(idx) {
        if (length(unique(z$truth[idx])) < 2L) 
            return(rep(NA_real_, length(point)))
        metric_vector(z$truth[idx], z$prob[idx])
    }, numeric(length(point)))
    bind_rows(lapply(seq_along(point), function(j) {
        vals <- boot[j, ]
        data.frame(model = unique(z$model), metric = names(point)[j], estimate = unname(point[j]), lower_95 = unname(quantile(vals, 
            0.025, na.rm = TRUE)), upper_95 = unname(quantile(vals, 0.975, na.rm = TRUE)), bootstrap_n = BOOT_N)
    }))
}))

wide_oof <- arrange(tidyr::pivot_wider(select(all_oof, row_id, truth, model, prob), names_from = model, 
    values_from = prob), row_id)

reference_name <- "ReferenceOnlyLinear"

comparison_names <- c("ConventionalLogistic", "GAM", "StableKernelNB")

incremental_rows <- bind_rows(lapply(comparison_names, function(model_name) {
    point_reference <- metric_vector(wide_oof$truth, wide_oof[[reference_name]])
    point_model <- metric_vector(wide_oof$truth, wide_oof[[model_name]])
    point_difference <- point_model - point_reference
    boot_difference <- vapply(boot_ids, function(idx) {
        metric_vector(wide_oof$truth[idx], wide_oof[[model_name]][idx]) - metric_vector(wide_oof$truth[idx], 
            wide_oof[[reference_name]][idx])
    }, numeric(length(point_difference)))
    bind_rows(lapply(seq_along(point_difference), function(j) {
        data.frame(model = model_name, reference = reference_name, metric = names(point_difference)[j], 
            difference = unname(point_difference[j]), lower_95 = unname(quantile(boot_difference[j, ], 
                0.025, na.rm = TRUE)), upper_95 = unname(quantile(boot_difference[j, ], 0.975, na.rm = TRUE)), 
            bootstrap_n = BOOT_N)
    }))
}))

thresholds <- seq(0.01, 0.8, by = 0.01)

dca_rows <- bind_rows(lapply(split(all_oof, all_oof$model), function(z) {
    prevalence <- mean(z$truth)
    bind_rows(lapply(thresholds, function(pt) {
        pred <- z$prob >= pt
        tp <- sum(pred & z$truth == 1L)
        fp <- sum(pred & z$truth == 0L)
        data.frame(model = unique(z$model), threshold = pt, net_benefit = tp/nrow(z) - fp/nrow(z) * pt/(1 - 
            pt), treat_all = prevalence - (1 - prevalence) * pt/(1 - pt), treat_none = 0)
    }))
}))

write.csv(fold_predictions, file.path(PRED_DIR, "01_reference_predictor_fold_predictions.csv"), row.names = FALSE)

write.csv(all_oof, file.path(PRED_DIR, "02_benchmark_patient_level_repeated_OOF.csv"), row.names = FALSE)

write.csv(metric_rows, file.path(TABLE_DIR, "01_benchmark_metrics_with_bootstrap_CI.csv"), row.names = FALSE)

write.csv(dca_rows, file.path(TABLE_DIR, "02_benchmark_decision_curve_data.csv"), row.names = FALSE)

write.csv(incremental_rows, file.path(TABLE_DIR, "03_incremental_performance_vs_reference_predictor.csv"), 
    row.names = FALSE)

capture.output(sessionInfo(), file = file.path(LOG_DIR, "sessionInfo.txt"))

writeLines("COMPLETED", file.path(LOG_DIR, "COMPLETED.txt"))

print(metric_rows[metric_rows$metric %in% c("AUROC", "PR_AUC", "Brier"), ], row.names = FALSE)
