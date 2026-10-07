options(stringsAsFactors = FALSE)

invisible(NULL)

suppressPackageStartupMessages({
    library(recipes)
    library(klaR)
    library(kernelshap)
    library(ggplot2)
    library(shapviz)
    library(dplyr)
    library(tidyr)
    library(readr)
    library(patchwork)
})

source("config.R", encoding = "UTF-8")

work_dir <- "."

out_dir <- file.path(SHAP_DIR, "figures")

supp_dir <- file.path(SHAP_DIR, "results")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dir.create(supp_dir, recursive = TRUE, showWarnings = FALSE)

dat <- read.csv(INPUT_FILE, check.names = FALSE)

names(dat)[match(OUTCOME_COLUMN, names(dat))] <- "outcome"

dat$outcome <- factor(ifelse(dat$outcome == 1, "Yes", "No"), levels = c("Yes", "No"))

predictor_cols <- setdiff(names(dat), c("outcome", EXCLUDED_COLUMNS))

is_binary01 <- function(x) {
    z <- unique(na.omit(as.character(x)))
    length(z) <= 2 && all(z %in% c("0", "1"))
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

strict <- filter(read.csv(file.path(MAIN_DIR, "03_Predictions", "01_strict_primary_OOF.csv"), check.names = FALSE), 
    model == "StableKernelNB")

features <- read.csv(file.path(MAIN_DIR, "01_Tables", "04_foldwise_feature_sets.csv"), check.names = FALSE)

stable_prob <- function(model_fit, nd) {
    nd <- if (is.data.frame(nd)) 
        nd
    else as.data.frame(nd, check.names = FALSE)
    nd <- nd[, model_fit$varnames, drop = FALSE]
    lev <- model_fit$levels
    n <- nrow(nd)
    k_num <- length(lev)
    log_post <- matrix(log(pmax(as.numeric(model_fit$apriori), 1e-12)), nrow = n, ncol = k_num, byrow = TRUE, 
        dimnames = list(NULL, lev))
    for (j in seq_along(model_fit$varnames)) {
        v <- model_fit$varnames[j]
        train_v <- model_fit$x[[v]]
        tab <- model_fit$tables[[v]]
        if (is.numeric(train_v)) {
            vals <- as.numeric(nd[[v]])
            for (kk in seq_along(lev)) {
                dens <- if (isTRUE(model_fit$usekernel)) {
                  ker <- tab[[lev[kk]]]
                  vapply(vals, function(z) klaR:::dkernel(x = z, kernel = ker), numeric(1))
                }
                else {
                  stats::dnorm(vals, mean = tab[lev[kk], 1], sd = tab[lev[kk], 2])
                }
                dens[!is.finite(dens) | dens <= 0] <- 1e-12
                log_post[, kk] <- log_post[, kk] + log(dens)
            }
        }
        else {
            vals <- as.character(nd[[v]])
            col_idx <- match(vals, colnames(tab))
            for (kk in seq_along(lev)) {
                dens <- rep(1e-12, n)
                ok <- !is.na(col_idx)
                dens[ok] <- as.numeric(tab[lev[kk], col_idx[ok]])
                dens[!is.finite(dens) | dens <= 0] <- 1e-12
                log_post[, kk] <- log_post[, kk] + log(dens)
            }
        }
    }
    row_max <- apply(log_post, 1, max)
    out <- exp(log_post - row_max)
    out <- out/rowSums(out)
    as.numeric(out[, "Yes"])
}

predict_raw <- function(object, newdata) {
    baked <- bake(object$recipe, new_data = newdata)
    stable_prob(object$model, baked[, object$baked_vars, drop = FALSE])
}

all_explained <- sort(unique(features$variable))

folds <- unique(strict$fold)

long_parts <- vector("list", length(folds))

check_rows <- vector("list", length(folds))

base_parts <- vector("list", length(folds))

for (ii in seq_along(folds)) {
    fold_name <- folds[ii]
    valid_ids <- strict$row_id[strict$fold == fold_name]
    snapshot <- readRDS(file.path(MAIN_DIR, "04_Models", sprintf("outer_fitted_%02d.rds", ii)))
    stopifnot(snapshot$fold == fold_name, identical(as.integer(snapshot$validation_ids), as.integer(valid_ids)))
    train_ids <- snapshot$train_ids
    vars <- snapshot$selectors$boruta
    saved_fit <- snapshot$chosen_answers$StableKernelNB$fit
    fitted <- list(recipe = saved_fit$recipe, model = saved_fit$finalModel, baked_vars = saved_fit$finalModel$varnames)
    valid_raw <- dat[valid_ids, vars, drop = FALSE]
    pred <- predict_raw(fitted, valid_raw)
    locked <- strict$prob[strict$fold == fold_name]
    check_rows[[ii]] <- data.frame(fold = fold_name, n = length(valid_ids), max_abs_prediction_difference = max(abs(pred - 
        locked)), mean_abs_prediction_difference = mean(abs(pred - locked)))
    stopifnot(max(abs(pred - locked)) < 1e-08)
    set.seed(1000 + ii)
    bg_ids <- sample(train_ids, min(50, length(train_ids)))
    bg <- dat[bg_ids, vars, drop = FALSE]
    ks <- kernelshap(fitted, X = valid_raw, bg_X = bg, pred_fun = predict_raw, verbose = FALSE)
    S <- as.matrix(ks$S)
    base_parts[[ii]] <- data.frame(row_id = valid_ids, fold = fold_name, baseline = rep(as.numeric(ks$baseline), 
        length(valid_ids)))
    part <- expand.grid(row_id = valid_ids, variable = all_explained, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    part$fold <- fold_name
    part$shap <- 0
    for (v in colnames(S)) {
        part$shap[part$variable == v] <- S[, v]
    }
    long_parts[[ii]] <- part
    message(sprintf("Completed %s (%d/%d); maximum prediction difference %.3g", fold_name, ii, length(folds), 
        check_rows[[ii]]$max_abs_prediction_difference))
}

shap_long <- bind_rows(long_parts)

checks <- bind_rows(check_rows)

bases <- arrange(summarise(group_by(bind_rows(base_parts), row_id), baseline = mean(baseline), .groups = "drop"), 
    row_id)

write.csv(checks, file.path(supp_dir, "OOF_SHAP_prediction_reproduction_check.csv"), row.names = FALSE)

write.csv(shap_long, file.path(supp_dir, "OOF_SHAP_fold_level_values.csv"), row.names = FALSE)

agg <- summarise(group_by(shap_long, row_id, variable), shap = mean(shap), .groups = "drop")

wide <- pivot_wider(agg, names_from = variable, values_from = shap, values_fill = 0)

wide <- wide[match(seq_len(nrow(dat)), wide$row_id), , drop = FALSE]

S <- as.matrix(wide[, all_explained, drop = FALSE])

X <- dat[, all_explained, drop = FALSE]

pred <- arrange(filter(read.csv(file.path(work_dir, "03_Results/03_Predictions/03_patient_level_repeated_OOF.csv"), 
    check.names = FALSE), model == "StableKernelNB"), row_id)

write.csv(data.frame(row_id = seq_len(nrow(dat)), truth = as.integer(dat$outcome == "Yes"), oof_probability = pred$prob, 
    oof_shap_baseline = bases$baseline, S, check.names = FALSE), file.path(supp_dir, "OOF_SHAP_patient_level_values.csv"), 
    row.names = FALSE)

new_names <- all_explained

colnames(S) <- new_names

names(X) <- new_names

sv <- shapviz(S, X = X)

theme_pub <- theme_minimal(base_size = 11) + theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", 
    color = "#183B56"))

p1 <- sv_importance(sv, kind = "bar", max_display = min(14, ncol(S)), fill = "#2F6B9A") + ggtitle("A  Global OOF-SHAP importance") + 
    theme_pub

p2 <- sv_importance(sv, kind = "beeswarm", max_display = min(14, ncol(S))) + ggtitle("B  Distribution of held-out feature attributions") + 
    theme_pub

g6 <- p1 + p2 + plot_layout(widths = c(0.9, 1.25))

ggsave(file.path(out_dir, "Figure_6_OOF_SHAP_importance_and_beeswarm.jpg"), g6, width = 12, height = 6.6, 
    dpi = 400, bg = "white")

ggsave(file.path(out_dir, "Figure_6_OOF_SHAP_importance_and_beeswarm.pdf"), g6, width = 12, height = 6.6, 
    bg = "white")

imp <- sort(colMeans(abs(S)), decreasing = TRUE)

top_dep <- names(imp)[seq_len(min(4, length(imp)))]

dep_plots <- lapply(seq_along(top_dep), function(i) {
    sv_dependence(sv, v = top_dep[i]) + ggtitle(paste0(LETTERS[i], "  ", top_dep[i])) + theme_pub
})

g7 <- wrap_plots(dep_plots, ncol = 2)

ggsave(file.path(out_dir, "Figure_7_OOF_SHAP_dependence.jpg"), g7, width = 10, height = 7.5, dpi = 400, 
    bg = "white")

ggsave(file.path(out_dir, "Figure_7_OOF_SHAP_dependence.pdf"), g7, width = 10, height = 7.5, bg = "white")

target_q <- quantile(pred$prob, c(0.25, 0.5, 0.75))

case_ids <- unique(vapply(target_q, function(q) which.min(abs(pred$prob - q)), integer(1)))

wf <- lapply(seq_along(case_ids), function(i) {
    id <- case_ids[i]
    one_sv <- shapviz(matrix(S[id, ], nrow = 1, dimnames = list(NULL, colnames(S))), X = X[id, , drop = FALSE], 
        baseline = bases$baseline[id])
    sv_waterfall(one_sv, row_id = 1, max_display = min(10, ncol(S))) + ggtitle(sprintf("%s  Individual prediction (predicted probability %.3f)", 
        LETTERS[i], pred$prob[id])) + theme_pub
})

g8 <- wrap_plots(wf, ncol = 1)

ggsave(file.path(out_dir, "Figure_8_Individual_OOF_SHAP_explanations.jpg"), g8, width = 9, height = 3.8 * 
    length(wf), dpi = 400, bg = "white")

ggsave(file.path(out_dir, "Figure_8_Individual_OOF_SHAP_explanations.pdf"), g8, width = 9, height = 3.8 * 
    length(wf), bg = "white")

write.csv(data.frame(Predictor = names(imp), Mean_absolute_OOF_SHAP = as.numeric(imp)), file.path(supp_dir, 
    "OOF_SHAP_Importance.csv"), row.names = FALSE)

writeLines(c(sprintf("Maximum absolute difference between retained-fit and locked-fold probabilities: %.12g", 
    max(checks$max_abs_prediction_difference)), sprintf("Mean of fold-level mean absolute differences: %.12g", 
    mean(checks$mean_abs_prediction_difference)), sprintf("Individual cases (25th, 50th, 75th predicted-risk quantiles): %s", 
    paste(case_ids, collapse = ", "))), file.path(supp_dir, "OOF_SHAP_analysis_summary.txt"))

message("OOF-SHAP analysis completed.")

writeLines(as.character(Sys.time()), file.path(supp_dir, "COMPLETED.txt"))
