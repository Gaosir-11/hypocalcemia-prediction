options(stringsAsFactors = FALSE, warn = 1)

BOOT_N <- 2000L

ONE_SE_FLOOR <- 0.005

CORR_CANDIDATES <- c(0.85, 0.9, 0.95, 1)

safe_seed <- function(x) {
    as.integer(((as.numeric(x) - 1)%%.Machine$integer.max) + 1)
}

{
    model_settings <- data.frame(AlgorithmName = c("StableKernelNB", "Ridge", "LASSO", "ElasticNet", 
        "GAM", "RegularizedRF", "ConservativeXGBoost", "ConservativeCatBoost"), Implementation = c("custom_klar_nb", 
        "custom_ridge", "custom_lasso", "custom_elastic", "custom_gam", "custom_rf_regularized", "custom_xgboost", 
        "custom_catboost"), stringsAsFactors = FALSE)
    penalized_models <- c("Ridge", "LASSO", "ElasticNet")
}

requested_models <- trimws(strsplit(Sys.getenv("ML_OPTIMIZATION_MODELS", unset = Sys.getenv("ML_MODELS", 
    unset = "")), ",", fixed = TRUE)[[1]])

requested_models <- requested_models[nzchar(requested_models)]

if (length(requested_models)) {
    model_settings <- model_settings[model_settings$AlgorithmName %in% requested_models, , drop = FALSE]
    if (!nrow(model_settings)) 
        stop("ML_MODELS did not match any optimization-driver model.")
    penalized_models <- intersect(penalized_models, model_settings$AlgorithmName)
}

set.seed(SEED)

{
    outer_train_indices <- caret::createMultiFolds(dat[[OUTCOME_COL]], k = OUTER_FOLDS, times = OUTER_REPEATS)
    fold_order <- names(outer_train_indices)
}

manifest <- bind_rows(lapply(seq_along(outer_train_indices), function(i) {
    data.frame(fold = fold_order[i], row_id = sort(setdiff(seq_len(nrow(dat)), outer_train_indices[[i]])), 
        stringsAsFactors = FALSE)
}))

if (length(fold_order) != OUTER_FOLDS * OUTER_REPEATS) {
    stop("Unexpected number of outer folds for seed ", SEED, ".")
}

if (!all(table(manifest$row_id) == OUTER_REPEATS)) {
    stop("Each patient must occur once per repeat in outer validation.")
}

write.csv(manifest, file.path(TAB_DIR, "00_seed_outer_validation_manifest.csv"), row.names = FALSE)

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
        return(NA_character_)
    best <- ok[which.max(ok$inner_auc), , drop = FALSE]
    allowance <- best$inner_se
    if (!is.finite(allowance)) 
        allowance <- ONE_SE_FLOOR
    eligible <- ok[ok$inner_auc >= best$inner_auc - max(allowance, ONE_SE_FLOOR), , drop = FALSE]
    ord_args <- lapply(simpler_cols, function(v) eligible[[v]])
    ord_args[[length(ord_args) + 1L]] <- -eligible$inner_auc
    eligible <- eligible[do.call(order, ord_args), , drop = FALSE]
    eligible$config_id[1]
}

auc_safe <- function(y, p) {
    ok <- is.finite(p) & !is.na(y)
    y <- y[ok]
    p <- p[ok]
    if (length(unique(y)) < 2L) 
        return(NA_real_)
    as.numeric(pROC::auc(pROC::roc(y, p, direction = "<", quiet = TRUE)))
}

param_signature <- function(bt) {
    paste(paste(names(bt), vapply(bt, function(z) as.character(z[1]), character(1)), sep = "="), collapse = ";")
}

make_ensemble <- function(chosen_answers, outer_probabilities, y, seed) {
    inner_parts <- list()
    for (nm in names(chosen_answers)) {
        pp <- best_tune_predictions(chosen_answers[[nm]]$fit)
        if (is.null(pp) || !all(c("rowIndex", "obs", "Yes") %in% names(pp))) 
            next
        inner_parts[[nm]] <- pp[, c("rowIndex", "obs", "Yes"), drop = FALSE]
        names(inner_parts[[nm]])[3] <- nm
    }
    if (length(inner_parts) < 2L) 
        return(NULL)
    z <- Reduce(function(a, b) merge(a, b, by = c("rowIndex", "obs"), all = FALSE), inner_parts)
    model_cols <- intersect(names(chosen_answers), names(z))
    z <- z[stats::complete.cases(z[, model_cols, drop = FALSE]), , drop = FALSE]
    if (nrow(z) < 20L || length(model_cols) < 2L) 
        return(NULL)
    yy <- as.integer(z$obs == "Yes")
    base_auc <- vapply(model_cols, function(v) auc_safe(yy, z[[v]]), numeric(1))
    model_cols <- names(sort(base_auc, decreasing = TRUE))
    model_cols <- head(model_cols[is.finite(base_auc[model_cols])], 4L)
    if (length(model_cols) < 2L) 
        return(NULL)
    kept <- character()
    for (v in model_cols) {
        if (!length(kept) || all(abs(stats::cor(z[[v]], z[, kept, drop = FALSE])) < 0.98)) 
            kept <- c(kept, v)
    }
    model_cols <- kept
    if (length(model_cols) < 2L) 
        return(NULL)
    candidates <- list()
    m2 <- head(model_cols, 2L)
    candidates$EqualTop2 <- list(inner = rowMeans(z[, m2, drop = FALSE]), outer = rowMeans(as.data.frame(outer_probabilities[m2]), 
        na.rm = TRUE), members = m2, weights = rep(1/length(m2), length(m2)))
    if (length(model_cols) >= 3L) {
        m3 <- head(model_cols, 3L)
        candidates$EqualTop3 <- list(inner = rowMeans(z[, m3, drop = FALSE]), outer = rowMeans(as.data.frame(outer_probabilities[m3]), 
            na.rm = TRUE), members = m3, weights = rep(1/length(m3), length(m3)))
    }
    w <- pmax(base_auc[model_cols] - 0.5, 0.001)
    w <- w/sum(w)
    candidates$AUCWeighted <- list(inner = as.numeric(as.matrix(z[, model_cols, drop = FALSE]) %*% w), 
        outer = as.numeric(as.matrix(as.data.frame(outer_probabilities[model_cols])) %*% w), members = model_cols, 
        weights = w)
    x_meta <- qlogis(pmax(1e-06, pmin(1 - 1e-06, as.matrix(z[, model_cols, drop = FALSE]))))
    set.seed(seed)
    meta_folds <- caret::createFolds(factor(yy, levels = c(1, 0)), k = 4, returnTrain = FALSE)
    foldid <- integer(length(yy))
    for (i in seq_along(meta_folds)) foldid[meta_folds[[i]]] <- i
    meta_cv <- tryCatch(glmnet::cv.glmnet(x_meta, yy, family = "binomial", alpha = 0, foldid = foldid, 
        type.measure = "auc", keep = TRUE, standardize = TRUE), error = function(e) NULL)
    if (!is.null(meta_cv)) {
        li <- which.min(abs(meta_cv$lambda - meta_cv$lambda.1se))
        inner_meta <- as.numeric(meta_cv$fit.preval[, li])
        if (any(inner_meta < 0 | inner_meta > 1, na.rm = TRUE)) 
            inner_meta <- stats::plogis(inner_meta)
        outer_x <- qlogis(pmax(1e-06, pmin(1 - 1e-06, as.matrix(as.data.frame(outer_probabilities[model_cols])))))
        outer_meta <- as.numeric(stats::predict(meta_cv$glmnet.fit, newx = outer_x, s = meta_cv$lambda.1se, 
            type = "response"))
        candidates$RidgeStack <- list(inner = inner_meta, outer = outer_meta, members = model_cols, weights = NA_real_)
    }
    cand_auc <- vapply(candidates, function(a) auc_safe(yy, a$inner), numeric(1))
    best_auc <- max(cand_auc, na.rm = TRUE)
    eligible <- names(cand_auc)[cand_auc >= best_auc - ONE_SE_FLOOR]
    preference <- c("EqualTop2", "EqualTop3", "AUCWeighted", "RidgeStack")
    selected <- preference[preference %in% eligible][1]
    ans <- candidates[[selected]]
    list(method = selected, inner_auc = cand_auc[selected], outer_prob = ans$outer, members = paste(ans$members, 
        collapse = "+"), weights = if (all(is.finite(ans$weights))) paste(round(ans$weights, 6), collapse = "+") else "ridge_stack", 
        candidate_auc = paste(paste(names(cand_auc), round(cand_auc, 6), sep = "="), collapse = ";"))
}

strict_oof_rows <- list()

sensitivity_oof_rows <- list()

config_rows <- list()

threshold_rows <- list()

tuning_rows <- list()

feature_rows <- list()

ensemble_rows <- list()

normalize_strict_oof <- function(z) {
    required <- c("row_id", "fold", "model", "strategy", "corr_threshold", "truth", "prob")
    has_nested_models <- any(vapply(intersect(model_settings$AlgorithmName, names(z)), function(nm) is.data.frame(z[[nm]]), 
        logical(1)))
    if (all(required %in% names(z)) && !has_nested_models && !any(grepl("\\.row_id$", names(z)))) {
        return(z[, required, drop = FALSE])
    }
    parts <- list()
    for (md in model_settings$AlgorithmName) {
        if (md %in% names(z) && is.data.frame(z[[md]])) {
            zz <- z[[md]]
        }
        else {
            prefix <- paste0(md, ".")
            cols <- names(z)[startsWith(names(z), prefix)]
            if (!length(cols)) 
                next
            zz <- z[, cols, drop = FALSE]
            names(zz) <- substring(names(zz), nchar(prefix) + 1L)
        }
        if (all(required %in% names(zz))) {
            zz <- zz[!is.na(zz$row_id), required, drop = FALSE]
            if (nrow(zz)) 
                parts[[md]] <- zz
        }
    }
    if (all(required %in% names(z))) {
        ens <- z[!is.na(z$row_id) & z$model == "NestedEnsemble", required, drop = FALSE]
        if (nrow(ens)) 
            parts[["NestedEnsemble"]] <- ens
    }
    bind_rows(unname(parts))
}

for (fold_i in seq_along(fold_order)) {
    fold_name <- fold_order[fold_i]
    checkpoint <- file.path(RDS_DIR, paste0("limited_outer_", sprintf("%02d", fold_i), ".rds"))
    if (RESUME && file.exists(checkpoint)) {
        cat(sprintf("[%02d/%02d] Resume %s\n", fold_i, length(fold_order), fold_name))
        obj <- readRDS(checkpoint)
    }
    else {
        cat(sprintf("\n[%02d/%02d] %s\n", fold_i, length(fold_order), fold_name))
        va_idx <- sort(unique(manifest$row_id[manifest$fold == fold_name]))
        tr_idx <- setdiff(seq_len(nrow(dat)), va_idx)
        tr <- dat[tr_idx, , drop = FALSE]
        va <- dat[va_idx, , drop = FALSE]
        selectors <- fit_selectors(tr, safe_seed(SEED + fold_i * 100))
        write.csv(selectors$boruta_stats, file.path(TAB_DIR, paste0("Boruta_status_", sprintf("%02d", 
            fold_i), ".csv")), row.names = FALSE, fileEncoding = "UTF-8")
        boruta_vars <- safe_feature_set(selectors$boruta, predictor_cols)
        lasso_vars <- safe_feature_set(selectors$lasso, boruta_vars)
        feature_sets <- {
            list(Boruta = boruta_vars)
        }
        fold_features <- bind_rows(lapply(names(feature_sets), function(st) data.frame(fold = fold_name, 
            strategy = st, variable = feature_sets[[st]])))
        inner_ctl <- make_inner_control(tr[[OUTCOME_COL]], safe_seed(SEED + fold_i * 10000))
        threshold_answers <- list()
        threshold_cmp <- list()
        selected_corr <- list()
        for (st in names(feature_sets)) {
            for (cc in CORR_CANDIDATES) {
                cfg <- paste(st, sprintf("corr%.2f", cc), sep = "__")
                ans <- tryCatch(fit_one_model(tr, va, feature_sets[[st]], "Ridge", "custom_ridge", safe_seed(SEED + 
                  fold_i * 1e+05 + match(st, names(feature_sets)) * 100 + round(cc * 100)), inner_ctl, 
                  corr_threshold = cc), error = function(e) structure(list(error = conditionMessage(e)), 
                  class = "model_error"))
                if (inherits(ans, "model_error")) {
                  threshold_cmp[[cfg]] <- data.frame(fold = fold_name, strategy = st, corr_threshold = cc, 
                    config_id = cfg, n_features = length(feature_sets[[st]]), inner_auc = NA_real_, inner_se = NA_real_, 
                    status = ans$error)
                }
                else {
                  threshold_answers[[cfg]] <- ans
                  threshold_cmp[[cfg]] <- data.frame(fold = fold_name, strategy = st, corr_threshold = cc, 
                    config_id = cfg, n_features = length(feature_sets[[st]]), inner_auc = ans$inner_auc, 
                    inner_se = inner_se(ans$fit), status = "OK")
                }
            }
            cmp_st <- bind_rows(threshold_cmp)[bind_rows(threshold_cmp)$strategy == st, , drop = FALSE]
            chosen_cfg <- choose_one_se(cmp_st, c("corr_threshold"))
            selected_corr[[st]] <- cmp_st$corr_threshold[match(chosen_cfg, cmp_st$config_id)]
            cmp_st$selected <- cmp_st$config_id == chosen_cfg
            threshold_cmp[names(threshold_cmp) %in% cmp_st$config_id] <- split(cmp_st, cmp_st$config_id)
        }
        fold_strict <- list()
        fold_sensitivity <- list()
        fold_configs <- list()
        fold_tuning <- list()
        chosen_answers <- list()
        outer_probs <- list()
        for (m in seq_len(nrow(model_settings))) {
            algo <- model_settings$AlgorithmName[m]
            impl <- model_settings$Implementation[m]
            strategies <- "Boruta"
            answers <- list()
            comparisons <- list()
            cat(sprintf("  [%d/%d] %s\n", m, nrow(model_settings), algo))
            for (st in strategies) {
                cc <- selected_corr[[st]]
                cfg <- paste(algo, st, sprintf("corr%.2f", cc), sep = "__")
                ridge_cfg <- paste(st, sprintf("corr%.2f", cc), sep = "__")
                ans <- if (algo == "Ridge") 
                  threshold_answers[[ridge_cfg]]
                else tryCatch(fit_one_model(tr, va, feature_sets[[st]], algo, impl, safe_seed(SEED + 
                  fold_i * 1e+06 + m * 10000 + match(st, names(feature_sets)) * 100), inner_ctl, corr_threshold = cc), 
                  error = function(e) structure(list(error = conditionMessage(e)), class = "model_error"))
                if (is.null(ans) || inherits(ans, "model_error")) {
                  msg <- if (is.null(ans)) 
                    "Missing cached Ridge fit"
                  else ans$error
                  comparisons[[cfg]] <- data.frame(fold = fold_name, model = algo, strategy = st, corr_threshold = cc, 
                    config_id = cfg, n_features = length(feature_sets[[st]]), inner_auc = NA_real_, inner_se = NA_real_, 
                    status = msg)
                }
                else {
                  answers[[cfg]] <- ans
                  comparisons[[cfg]] <- data.frame(fold = fold_name, model = algo, strategy = st, corr_threshold = cc, 
                    config_id = cfg, n_features = length(feature_sets[[st]]), inner_auc = ans$inner_auc, 
                    inner_se = inner_se(ans$fit), status = "OK")
                  fold_tuning[[cfg]] <- data.frame(fold = fold_name, model = algo, strategy = st, corr_threshold = cc, 
                    param_signature = param_signature(ans$best_tune), inner_auc = ans$inner_auc)
                }
            }
            cmp <- bind_rows(comparisons)
            selected_cfg <- choose_one_se(cmp, c("n_features", "corr_threshold"))
            cmp$selected <- cmp$config_id == selected_cfg
            fold_configs[[algo]] <- cmp
            if (!is.na(selected_cfg) && selected_cfg %in% names(answers)) {
                chosen <- answers[[selected_cfg]]
                chosen_row <- cmp[cmp$config_id == selected_cfg, , drop = FALSE]
                fold_strict[[algo]] <- data.frame(row_id = va$.row_id, fold = fold_name, model = algo, 
                  strategy = chosen_row$strategy, corr_threshold = chosen_row$corr_threshold, truth = as.integer(va[[OUTCOME_COL]] == 
                    "Yes"), prob = chosen$prob)
                chosen_answers[[algo]] <- chosen
                outer_probs[[algo]] <- chosen$prob
            }
        }
        ensemble <- make_ensemble(chosen_answers, outer_probs, tr[[OUTCOME_COL]], safe_seed(SEED + fold_i * 
            2e+06))
        fold_ensemble <- NULL
        ensemble_meta <- NULL
        if (!is.null(ensemble)) {
            fold_ensemble <- data.frame(row_id = va$.row_id, fold = fold_name, model = "NestedEnsemble", 
                strategy = "InnerOOFWeighted", corr_threshold = NA_real_, truth = as.integer(va[[OUTCOME_COL]] == 
                  "Yes"), prob = ensemble$outer_prob)
            ensemble_meta <- data.frame(fold = fold_name, method = ensemble$method, inner_auc = ensemble$inner_auc, 
                members = ensemble$members, weights = ensemble$weights, candidate_auc = ensemble$candidate_auc)
        }
        obj <- list(strict_oof = bind_rows(unname(fold_strict), fold_ensemble), sensitivity_oof = bind_rows(fold_sensitivity), 
            configs = bind_rows(fold_configs), thresholds = bind_rows(threshold_cmp), tuning = bind_rows(fold_tuning), 
            features = fold_features, ensemble = ensemble_meta)
        saveRDS(obj, checkpoint)
        saveRDS(list(fold = fold_name, train_ids = tr_idx, validation_ids = va_idx, selectors = selectors, 
            chosen_answers = chosen_answers, inner_control = inner_ctl, configs = bind_rows(fold_configs)), 
            file.path(RDS_DIR, paste0("outer_fitted_", sprintf("%02d", fold_i), ".rds")))
    }
    normalized_strict <- normalize_strict_oof(obj$strict_oof)
    if (!any(normalized_strict$model == "NestedEnsemble")) {
        cmp_ok <- obj$configs[obj$configs$selected & obj$configs$status == "OK" & is.finite(obj$configs$inner_auc), 
            , drop = FALSE]
        cmp_ok <- cmp_ok[order(-cmp_ok$inner_auc, cmp_ok$n_features), , drop = FALSE]
        top2 <- unique(cmp_ok$model)[seq_len(min(2L, length(unique(cmp_ok$model))))]
        if (length(top2) == 2L) {
            a <- normalized_strict[normalized_strict$model == top2[1], c("row_id", "fold", "truth", "prob"), 
                drop = FALSE]
            b <- normalized_strict[normalized_strict$model == top2[2], c("row_id", "prob"), drop = FALSE]
            names(b)[2] <- "prob2"
            ee <- merge(a, b, by = "row_id", all = FALSE)
            fallback <- data.frame(row_id = ee$row_id, fold = ee$fold, model = "NestedEnsemble", strategy = "InnerOOFWeighted", 
                corr_threshold = NA_real_, truth = ee$truth, prob = rowMeans(ee[, c("prob", "prob2")]))
            normalized_strict <- bind_rows(normalized_strict, fallback)
            obj$ensemble <- data.frame(fold = fold_name, method = "EqualTop2Fallback", inner_auc = mean(cmp_ok$inner_auc[match(top2, 
                cmp_ok$model)], na.rm = TRUE), members = paste(top2, collapse = "+"), weights = "0.5+0.5", 
                candidate_auc = "fallback_selected_by_inner_auc")
        }
    }
    strict_oof_rows[[fold_i]] <- normalized_strict
    sensitivity_oof_rows[[fold_i]] <- obj$sensitivity_oof
    config_rows[[fold_i]] <- obj$configs
    threshold_rows[[fold_i]] <- obj$thresholds
    tuning_rows[[fold_i]] <- obj$tuning
    feature_rows[[fold_i]] <- obj$features
    ensemble_rows[[fold_i]] <- obj$ensemble
}

if ("package:plyr" %in% search()) detach("package:plyr", unload = FALSE, character.only = TRUE)

suppressPackageStartupMessages(library(dplyr))

strict_oof <- bind_rows(strict_oof_rows)

sensitivity_oof <- bind_rows(sensitivity_oof_rows)

configs <- bind_rows(config_rows)

thresholds <- bind_rows(threshold_rows)

tuning <- bind_rows(tuning_rows)

features <- bind_rows(feature_rows)

ensembles <- bind_rows(ensemble_rows)

{
    non_boruta_strategies <- unique(strict_oof$strategy[strict_oof$model != "NestedEnsemble" & strict_oof$strategy != 
        "Boruta"])
    if (length(non_boruta_strategies)) {
        stop("Boruta-only integrity check failed: non-Boruta feature strategy detected")
    }
}

write.csv(strict_oof, file.path(PRED_DIR, "01_strict_primary_OOF.csv"), row.names = FALSE)

write.csv(configs, file.path(TAB_DIR, "01_inner_model_configuration_selection.csv"), row.names = FALSE)

write.csv(thresholds, file.path(TAB_DIR, "02_inner_correlation_threshold_selection.csv"), row.names = FALSE)

write.csv(tuning, file.path(TAB_DIR, "03_selected_hyperparameters.csv"), row.names = FALSE)

write.csv(features, file.path(TAB_DIR, "04_foldwise_feature_sets.csv"), row.names = FALSE, fileEncoding = "UTF-8")

write.csv(ensembles, file.path(TAB_DIR, "05_ensemble_selection.csv"), row.names = FALSE)

all_oof <- bind_rows(strict_oof, sensitivity_oof)

aggregate_oof <- summarise(group_by(filter(all_oof, is.finite(prob)), row_id, model), truth = first(truth), 
    prob = mean(prob), predictions_n = n(), .groups = "drop")

write.csv(aggregate_oof, file.path(PRED_DIR, "03_patient_level_repeated_OOF.csv"), row.names = FALSE)

average_precision <- function(y, p) {
    o <- order(p, decreasing = TRUE)
    y <- y[o]
    if (!sum(y == 1)) 
        return(NA_real_)
    mean((cumsum(y)/seq_along(y))[y == 1])
}

metric_once <- function(y, p) c(AUROC = auc_safe(y, p), PR_AUC = average_precision(y, p), Brier = mean((p - 
    y)^2))

eval_ids <- sort(unique(strict_oof$row_id))

set.seed(safe_seed(SEED + 7e+05))

boot_ids <- replicate(BOOT_N, sample(seq_along(eval_ids), length(eval_ids), replace = TRUE), simplify = FALSE)

metric_rows <- lapply(unique(aggregate_oof$model), function(md) {
    z <- aggregate_oof[aggregate_oof$model == md, ]
    z <- z[match(eval_ids, z$row_id), ]
    if (anyNA(z$row_id)) 
        return(NULL)
    point <- metric_once(z$truth, z$prob)
    boot <- do.call(rbind, lapply(boot_ids, function(idx) metric_once(z$truth[idx], z$prob[idx])))
    data.frame(Model = md, Metric = names(point), Estimate = as.numeric(point), Lower = apply(boot, 2, 
        quantile, 0.025, na.rm = TRUE), Upper = apply(boot, 2, quantile, 0.975, na.rm = TRUE), BootstrapValid = colSums(is.finite(boot)))
})

metrics_long <- bind_rows(metric_rows)

write.csv(metrics_long, file.path(TAB_DIR, "06_metrics_bootstrap_long.csv"), row.names = FALSE)

metrics_wide <- tidyr::pivot_wider(dplyr::select(mutate(metrics_long, value = sprintf("%.3f (%.3f-%.3f)", 
    Estimate, Lower, Upper)), Model, Metric, value), names_from = Metric, values_from = value)

write.csv(metrics_wide, file.path(TAB_DIR, "07_metrics_bootstrap_wide.csv"), row.names = FALSE)

inner_selected <- dplyr::select(filter(configs, selected, status == "OK"), fold, model, strategy, corr_threshold, 
    inner_auc, inner_se)

inner_ensemble <- if (nrow(ensembles)) {
    transmute(ensembles, fold, model = "NestedEnsemble", strategy = "InnerOOFWeighted", corr_threshold = NA_real_, 
        inner_auc, inner_se = NA_real_)
} else {
    data.frame(fold = character(), model = character(), strategy = character(), corr_threshold = numeric(), 
        inner_auc = numeric(), inner_se = numeric())
}

inner_all <- bind_rows(inner_selected, inner_ensemble)

outer_fold <- summarise(group_by(strict_oof, fold, model, strategy, corr_threshold), outer_auc = auc_safe(truth, 
    prob), outer_n = n(), outer_events = sum(truth), .groups = "drop")

fold_report <- left_join(inner_all, outer_fold, by = c("fold", "model", "strategy", "corr_threshold"))

write.csv(fold_report, file.path(TAB_DIR, "08_foldwise_inner_outer_AUROC.csv"), row.names = FALSE)

strategy_freq <- ungroup(mutate(group_by(count(filter(configs, selected, status == "OK"), model, strategy, 
    name = "selected_folds"), model), selection_frequency = selected_folds/sum(selected_folds)))

corr_freq <- ungroup(mutate(group_by(count(filter(configs, selected, status == "OK"), model, corr_threshold, 
    name = "selected_folds"), model), selection_frequency = selected_folds/sum(selected_folds)))

selected_keys <- dplyr::select(filter(configs, selected, status == "OK"), fold, model, strategy, corr_threshold)

tuning_for_frequency <- inner_join(tuning, selected_keys, by = c("fold", "model", "strategy", "corr_threshold"))

param_freq <- ungroup(mutate(group_by(count(tuning_for_frequency, model, strategy, corr_threshold, param_signature, 
    name = "selected_folds"), model), selection_frequency = selected_folds/sum(selected_folds)))

write.csv(strategy_freq, file.path(TAB_DIR, "09_feature_strategy_frequency.csv"), row.names = FALSE)

write.csv(corr_freq, file.path(TAB_DIR, "10_correlation_threshold_frequency.csv"), row.names = FALSE)

write.csv(param_freq, file.path(TAB_DIR, "11_hyperparameter_frequency.csv"), row.names = FALSE)

integrity <- summarise(group_by(strict_oof, model), raw_rows = n(), patients = n_distinct(row_id), folds = n_distinct(fold), 
    missing_probabilities = sum(!is.finite(prob)), .groups = "drop")

write.csv(integrity, file.path(TAB_DIR, "13_OOF_integrity.csv"), row.names = FALSE)

capture.output(sessionInfo(), file = file.path(LOG_DIR, "sessionInfo.txt"))

try(parallel::stopCluster(cl), silent = TRUE)

foreach::registerDoSEQ()
