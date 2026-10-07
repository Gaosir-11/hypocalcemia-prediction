options(stringsAsFactors = FALSE, warn = 1)

if (.Platform$OS.type == "windows") {
    invisible(NULL)
}

suppressPackageStartupMessages({
    library(caret)
    library(recipes)
    library(pROC)
    library(dplyr)
})

source("config.R", encoding = "UTF-8")

DATA_FILE <- INPUT_FILE

MANIFEST_FILE <- file.path(MAIN_DIR, "01_Tables", "00_seed_outer_validation_manifest.csv")

FEATURE_FILE <- file.path(MAIN_DIR, "01_Tables", "04_foldwise_feature_sets.csv")

OUT_DIR <- LOGISTIC_DIR

CODE_DIR <- file.path(OUT_DIR, "01_Code")

TABLE_DIR <- file.path(OUT_DIR, "02_Tables")

PRED_DIR <- file.path(OUT_DIR, "03_Predictions")

LOG_DIR <- file.path(OUT_DIR, "04_Logs")

dir.create(CODE_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(PRED_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

OUTCOME_COL <- OUTCOME_COLUMN

EXCLUDED_PREDICTORS <- EXCLUDED_COLUMNS

BOOT_N <- 2000L

dat <- read.csv(DATA_FILE, check.names = FALSE)

clean_names <- sub("^﻿", "", names(dat), useBytes = TRUE)

Encoding(clean_names) <- "UTF-8"

names(dat) <- clean_names

stopifnot(OUTCOME_COL %in% names(dat), EXCLUDED_PREDICTORS %in% names(dat))

for (j in seq_along(dat)) {
    if (is.character(dat[[j]])) {
        z <- trimws(dat[[j]])
        z[z %in% c("", "NA", "N/A", "#N/A", "NULL", "null")] <- NA
        dat[[j]] <- z
    }
}

y_raw <- trimws(as.character(dat[[OUTCOME_COL]]))

y_raw[y_raw %in% c("Yes", "yes")] <- "1"

y_raw[y_raw %in% c("No", "no")] <- "0"

if (!all(na.omit(unique(y_raw)) %in% c("0", "1")) || anyNA(y_raw)) {
    stop("Outcome must be complete and coded as 0/1.")
}

dat[[OUTCOME_COL]] <- factor(ifelse(y_raw == "1", "Yes", "No"), levels = c("Yes", "No"))

predictor_cols <- setdiff(names(dat), c(OUTCOME_COL, EXCLUDED_PREDICTORS))

is_binary01 <- function(x) {
    ux <- unique(na.omit(as.character(x)))
    length(ux) >= 1L && length(ux) <= 2L && all(ux %in% c("0", "1"))
}

factor_cols <- unique(c(intersect(CATEGORICAL_COLUMNS, predictor_cols), predictor_cols[vapply(dat[predictor_cols], 
    is.character, logical(1))], predictor_cols[vapply(dat[predictor_cols], is_binary01, logical(1))]))

numeric_cols <- setdiff(predictor_cols, factor_cols)

for (v in factor_cols) {
    custom_levels <- CATEGORICAL_LEVELS[[v]]
    dat[[v]] <- if (is.null(custom_levels)) 
        factor(dat[[v]])
    else factor(dat[[v]], levels = custom_levels)
}

for (v in numeric_cols) dat[[v]] <- suppressWarnings(as.numeric(dat[[v]]))

dat$.row_id <- seq_len(nrow(dat))

manifest <- read.csv(MANIFEST_FILE, check.names = FALSE)

features <- read.csv(FEATURE_FILE, check.names = FALSE)

features <- features[features$strategy == "Boruta", , drop = FALSE]

fold_order <- unique(manifest$fold)

if (length(fold_order) != OUTER_K * OUTER_R) stop("Unexpected number of outer folds.")

if (!setequal(unique(features$fold), fold_order)) stop("Feature/fold manifest mismatch.")

make_recipe <- function(train_df, vars) {
    vars <- intersect(vars, predictor_cols)
    rec_dat <- train_df[, c(OUTCOME_COL, vars), drop = FALSE]
    recipes::step_normalize(recipes::step_zv(recipes::step_dummy(recipes::step_novel(recipes::step_unknown(recipes::step_YeoJohnson(recipes::step_impute_mode(recipes::step_impute_median(recipes::recipe(stats::reformulate(vars, 
        response = OUTCOME_COL), data = rec_dat), recipes::all_numeric_predictors()), recipes::all_nominal_predictors()), 
        recipes::all_numeric_predictors()), recipes::all_nominal_predictors(), new_level = "missing"), 
        recipes::all_nominal_predictors(), new_level = "novel"), recipes::all_nominal_predictors(), one_hot = FALSE), 
        recipes::all_predictors()), recipes::all_numeric_predictors())
}

fit_logistic_once <- function(train_df, valid_df, vars, seed) {
    set.seed(seed)
    rec <- make_recipe(train_df, vars)
    ctl <- caret::trainControl(method = "none", classProbs = TRUE)
    fit <- caret::train(rec, data = train_df[, c(OUTCOME_COL, vars), drop = FALSE], method = "glm", family = binomial(), 
        trControl = ctl, metric = "ROC")
    list(prob = as.numeric(predict(fit, newdata = valid_df[, vars, drop = FALSE], type = "prob")[, "Yes"]), 
        fit = fit)
}

auc_safe <- function(y, p) {
    ok <- is.finite(p) & !is.na(y)
    y <- y[ok]
    p <- p[ok]
    if (length(unique(y)) < 2L) 
        return(NA_real_)
    as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE)))
}

average_precision <- function(y, p) {
    ok <- is.finite(p) & !is.na(y)
    y <- y[ok]
    p <- p[ok]
    o <- order(p, decreasing = TRUE)
    y <- y[o]
    if (!sum(y == 1L)) 
        return(NA_real_)
    mean((cumsum(y)/seq_along(y))[y == 1L])
}

metric_once <- function(y, p) {
    c(AUROC = auc_safe(y, p), PR_AUC = average_precision(y, p), Brier = mean((p - y)^2))
}

pred_rows <- vector("list", length(fold_order))

fold_rows <- vector("list", length(fold_order))

fit_notes <- vector("list", length(fold_order))

for (i in seq_along(fold_order)) {
    fold <- fold_order[i]
    valid_idx <- sort(unique(manifest$row_id[manifest$fold == fold]))
    train_idx <- setdiff(seq_len(nrow(dat)), valid_idx)
    vars <- unique(features$variable[features$fold == fold])
    vars <- intersect(vars, predictor_cols)
    if (!length(vars)) 
        stop("No Boruta variables in ", fold)
    warning_messages <- character()
    ans <- withCallingHandlers(fit_logistic_once(dat[train_idx, , drop = FALSE], dat[valid_idx, , drop = FALSE], 
        vars, seed = 190000L + i), warning = function(w) {
        warning_messages <<- c(warning_messages, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    p <- pmin(pmax(ans$prob, 0), 1)
    y <- as.integer(dat[[OUTCOME_COL]][valid_idx] == "Yes")
    pred_rows[[i]] <- data.frame(row_id = valid_idx, fold = fold, model = "ConventionalLogistic", strategy = "Boruta", 
        truth = y, prob = p, stringsAsFactors = FALSE)
    fm <- metric_once(y, p)
    fold_rows[[i]] <- data.frame(fold = fold, model = "ConventionalLogistic", n_validation = length(valid_idx), 
        events_validation = sum(y), n_features = length(vars), AUROC = unname(fm["AUROC"]), PR_AUC = unname(fm["PR_AUC"]), 
        Brier = unname(fm["Brier"]), stringsAsFactors = FALSE)
    fit_notes[[i]] <- data.frame(fold = fold, n_features = length(vars), convergence = isTRUE(ans$fit$finalModel$converged), 
        warning_n = length(unique(warning_messages)), warning_text = paste(unique(warning_messages), 
            collapse = " | "), stringsAsFactors = FALSE)
    cat(sprintf("Completed %s (%d/%d); features=%d; fold AUROC=%.4f\n", fold, i, length(fold_order), 
        length(vars), fm["AUROC"]))
}

all_oof <- bind_rows(pred_rows)

fold_metrics <- bind_rows(fold_rows)

fit_diagnostics <- bind_rows(fit_notes)

patient_oof <- summarise(group_by(all_oof, row_id, model), truth = first(truth), prob = mean(prob), predictions_n = n(), 
    .groups = "drop")

if (nrow(patient_oof) != nrow(dat) || any(patient_oof$predictions_n != OUTER_R)) {
    stop("Repeated OOF aggregation failed.")
}

point <- metric_once(patient_oof$truth, patient_oof$prob)

set.seed((SEED_MAIN + 700000L))

boot <- replicate(BOOT_N, {
    idx <- sample.int(nrow(patient_oof), nrow(patient_oof), replace = TRUE)
    metric_once(patient_oof$truth[idx], patient_oof$prob[idx])
})

summary_long <- bind_rows(lapply(rownames(boot), function(metric) {
    vals <- boot[metric, ]
    data.frame(model = "ConventionalLogistic", metric = metric, estimate = unname(point[metric]), lower_95 = unname(stats::quantile(vals, 
        0.025, na.rm = TRUE)), upper_95 = unname(stats::quantile(vals, 0.975, na.rm = TRUE)), bootstrap_n = BOOT_N, 
        stringsAsFactors = FALSE)
}))

write.csv(all_oof, file.path(PRED_DIR, "01_fold_level_repeated_OOF_logistic.csv"), row.names = FALSE)

write.csv(patient_oof, file.path(PRED_DIR, "02_patient_level_repeated_OOF_logistic.csv"), row.names = FALSE)

write.csv(fold_metrics, file.path(TABLE_DIR, "01_outer_fold_metrics_logistic.csv"), row.names = FALSE)

write.csv(summary_long, file.path(TABLE_DIR, "02_metrics_bootstrap_long_logistic.csv"), row.names = FALSE)

write.csv(fit_diagnostics, file.path(TABLE_DIR, "03_fit_diagnostics_logistic.csv"), row.names = FALSE)

existing_long <- read.csv(file.path(MAIN_DIR, "01_Tables", "06_metrics_bootstrap_long.csv"), check.names = FALSE)

logistic_standard <- transmute(summary_long, Model = model, Metric = metric, Estimate = estimate, Lower = lower_95, 
    Upper = upper_95, BootstrapValid = bootstrap_n)

combined_long <- bind_rows(existing_long, logistic_standard)

auc_order <- pull(arrange(filter(combined_long, Metric == "AUROC"), desc(Estimate)), Model)

combined_long <- mutate(arrange(mutate(combined_long, Model = factor(Model, levels = auc_order)), Model, 
    match(Metric, c("AUROC", "PR_AUC", "Brier"))), Model = as.character(Model))

combined_wide <- data.frame(Model = auc_order, stringsAsFactors = FALSE)

for (metric_name in c("AUROC", "PR_AUC", "Brier")) {
    z <- combined_long[combined_long$Metric == metric_name, , drop = FALSE]
    z <- z[match(auc_order, z$Model), , drop = FALSE]
    combined_wide[[metric_name]] <- sprintf("%.3f (%.3f-%.3f)", z$Estimate, z$Lower, z$Upper)
}

write.csv(combined_long, file.path(TABLE_DIR, "04_all_10_models_bootstrap_long.csv"), row.names = FALSE)

write.csv(combined_wide, file.path(TABLE_DIR, "05_all_10_models_bootstrap_wide.csv"), row.names = FALSE)

capture.output(sessionInfo(), file = file.path(LOG_DIR, "sessionInfo.txt"))

cat("\nPatient-level repeated OOF results:\n")

print(summary_long, row.names = FALSE, digits = 6)

cat(sprintf("\nFold AUROC mean (SD): %.4f (%.4f); range %.4f-%.4f\n", mean(fold_metrics$AUROC, na.rm = TRUE), 
    sd(fold_metrics$AUROC, na.rm = TRUE), min(fold_metrics$AUROC, na.rm = TRUE), max(fold_metrics$AUROC, 
        na.rm = TRUE)))

cat(sprintf("Converged fits: %d/%d; folds with warnings: %d\n", sum(fit_diagnostics$convergence), nrow(fit_diagnostics), 
    sum(fit_diagnostics$warning_n > 0L)))
