source("config.R", encoding = "UTF-8")

configure_engine(INPUT_FILE, REFIT_DIR)

source(file.path("R", "engine.R"), encoding = "UTF-8")

source(file.path("app", "predict.R"), encoding = "UTF-8")

options(stringsAsFactors = FALSE, warn = 1)

invisible(NULL)

safe_seed <- function(x) as.integer(((as.numeric(x) - 1)%%.Machine$integer.max) + 1)

inner_se <- function(fit) {
    z <- fit$resample$ROC
    z <- z[is.finite(z)]
    if (length(z) < 2L) 
        return(NA_real_)
    stats::sd(z)/sqrt(length(z))
}

choose_one_se <- function(x, simpler_cols) {
    ok <- x[x$status == "OK" & is.finite(x$inner_auc), , drop = FALSE]
    if (!nrow(ok)) 
        stop("No valid candidate configuration was available.")
    best <- ok[which.max(ok$inner_auc), , drop = FALSE]
    allowance <- best$inner_se
    if (!is.finite(allowance)) 
        allowance <- 0.005
    eligible <- ok[ok$inner_auc >= best$inner_auc - max(allowance, 0.005), , drop = FALSE]
    ord_args <- lapply(simpler_cols, function(v) eligible[[v]])
    ord_args[[length(ord_args) + 1L]] <- -eligible$inner_auc
    eligible <- eligible[do.call(order, ord_args), , drop = FALSE]
    eligible$config_id[1]
}

set.seed(safe_seed(SEED + 999999))

selectors <- fit_selectors(dat, safe_seed(SEED + 999999), make_plots = TRUE, plot_prefix = "final_full_cohort_")

final_features <- safe_feature_set(selectors$boruta, predictor_cols)

if (!length(final_features)) stop("Full-cohort Boruta returned no predictors.")

corr_candidates <- c(0.85, 0.9, 0.95, 1)

inner_ctl <- make_inner_control(dat[[OUTCOME_COL]], safe_seed(SEED + 888888))

corr_rows <- list()

for (cc in corr_candidates) {
    ans <- tryCatch(fit_one_model(dat, dat, final_features, "Ridge", "custom_ridge", safe_seed(SEED + 
        7e+05 + round(cc * 100)), inner_ctl, corr_threshold = cc), error = function(e) structure(list(error = conditionMessage(e)), 
        class = "model_error"))
    cfg <- sprintf("Boruta__corr%.2f", cc)
    if (inherits(ans, "model_error")) {
        corr_rows[[cfg]] <- data.frame(config_id = cfg, corr_threshold = cc, n_features = length(final_features), 
            inner_auc = NA_real_, inner_se = NA_real_, status = ans$error)
    }
    else {
        corr_rows[[cfg]] <- data.frame(config_id = cfg, corr_threshold = cc, n_features = length(final_features), 
            inner_auc = ans$inner_auc, inner_se = inner_se(ans$fit), status = "OK")
    }
}

corr_table <- do.call(rbind, corr_rows)

chosen_cfg <- choose_one_se(corr_table, c("corr_threshold"))

chosen_corr <- corr_table$corr_threshold[match(chosen_cfg, corr_table$config_id)]

corr_table$selected <- corr_table$config_id == chosen_cfg

final_answer <- fit_one_model(dat, dat, final_features, "StableKernelNB", "custom_klar_nb", safe_seed(SEED + 
    777777), inner_ctl, corr_threshold = chosen_corr)

recipe_input_features <- final_features

final_model <- final_answer$fit

final_features <- intersect(recipe_input_features, final_model$finalModel$varnames)

stopifnot(length(final_features) > 0)

final_preprocessor <- final_model$recipe

final_threshold <- final_answer$threshold

metadata <- data.frame(model_variable = final_features, english_name = final_features, type = vapply(final_features, 
    function(v) if (is.numeric(dat[[v]])) "continuous" else "categorical", character(1)), unit = "", 
    allowed_min = -Inf, allowed_max = Inf, stringsAsFactors = FALSE)

if (file.exists(FEATURE_METADATA_FILE)) {
    supplied <- read.csv(FEATURE_METADATA_FILE, check.names = FALSE)
    matched <- match(metadata$model_variable, supplied$model_variable)
    for (column in intersect(c("english_name", "unit", "allowed_min", "allowed_max"), names(supplied))) {
        ok <- !is.na(matched)
        metadata[[column]][ok] <- supplied[[column]][matched[ok]]
    }
}

bundle <- export_bundle(final_model, dat, recipe_input_features, final_features, metadata)

dir.create(dirname(APP_BUNDLE), recursive = TRUE, showWarnings = FALSE)

saveRDS(bundle, APP_BUNDLE)

write.csv(data.frame(variable = final_features), file.path(REFIT_DIR, "final_features.csv"), row.names = FALSE)

write.csv(metadata, file.path(REFIT_DIR, "feature_dictionary.csv"), row.names = FALSE)

write.csv(corr_table, file.path(REFIT_DIR, "correlation_selection.csv"), row.names = FALSE)

write.csv(final_answer$best_tune, file.path(REFIT_DIR, "hyperparameters.csv"), row.names = FALSE)

p1 <- as.numeric(predict(final_model, newdata = dat[, recipe_input_features, drop = FALSE], type = "prob")[, 
    "Yes"])

p2 <- predict_bundle(bundle, dat[, recipe_input_features, drop = FALSE])

stopifnot(all(is.finite(p2)), max(abs(p1 - p2)) < 1e-12)

try(parallel::stopCluster(cl), silent = TRUE)

foreach::registerDoSEQ()
