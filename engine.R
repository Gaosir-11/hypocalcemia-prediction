options(stringsAsFactors = FALSE, warn = 1)

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x[1])) y else x

env_or_default <- function(name, default) {
    value <- Sys.getenv(name, unset = "")
    if (nzchar(value)) 
        value
    else default
}

DATA_FILE <- env_or_default("ML_DATA_FILE", file.path("data", "input.csv"))

OUTCOME_COL <- env_or_default("ML_OUTCOME_COL", "outcome")

DEFAULT_OUT_DIR <- file.path("outputs", "main")

OUT_DIR <- Sys.getenv("ML_OUT_DIR", unset = DEFAULT_OUT_DIR)

SEED <- as.integer(Sys.getenv("ML_SEED", unset = "42"))

OUTER_FOLDS <- as.integer(Sys.getenv("ML_OUTER_FOLDS", unset = "5"))

OUTER_REPEATS <- as.integer(Sys.getenv("ML_OUTER_REPEATS", unset = "5"))

INNER_FOLDS <- as.integer(Sys.getenv("ML_INNER_FOLDS", unset = "4"))

FEATURE_INNER_FOLDS <- INNER_FOLDS

BORUTA_MAX_RUNS <- as.integer(Sys.getenv("ML_BORUTA_RUNS", unset = "100"))

LASSO_LAMBDA_CHOICE <- "lambda.1se"

TUNE_BUDGET <- as.integer(Sys.getenv("ML_TUNE_BUDGET", unset = "24"))

SKIP_SHAP <- tolower(trimws(Sys.getenv("ML_SKIP_SHAP", unset = "0"))) %in% c("1", "true", "yes", "y")

SKIP_LASSO <- tolower(trimws(Sys.getenv("ML_SKIP_LASSO", unset = "0"))) %in% c("1", "true", "yes", "y")

INDEX_MODEL <- trimws(Sys.getenv("ML_INDEX_MODEL", unset = "StableKernelNB"))

TARGET_SENSITIVITY <- as.numeric(Sys.getenv("ML_TARGET_SENSITIVITY", unset = "0.85"))

if (!is.finite(TARGET_SENSITIVITY) || TARGET_SENSITIVITY <= 0 || TARGET_SENSITIVITY >= 1) {
    stop("Target sensitivity must be between 0 and 1.")
}

N_CORES <- as.integer(Sys.getenv("ML_CORES", unset = as.character(max(1, min(6, parallel::detectCores(logical = TRUE) - 
    1)))))

RESUME <- TRUE

USE_ONE_SE_RULE <- TRUE

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

FIG_DIR <- file.path(OUT_DIR, "02_Figures")

TAB_DIR <- file.path(OUT_DIR, "01_Tables")

RDS_DIR <- file.path(OUT_DIR, "04_Models")

LOG_DIR <- file.path(OUT_DIR, "05_Logs")

PRED_DIR <- file.path(OUT_DIR, "03_Predictions")

dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(RDS_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

dir.create(PRED_DIR, recursive = TRUE, showWarnings = FALSE)

required_pkgs <- c("caret", "recipes", "glmnet", "Boruta", "randomForest", "xgboost", "klaR", "MASS", 
    "catboost", "pROC", "ggplot2", "dplyr", "reshape2", "shapviz", "kernelshap", "patchwork", "gridExtra", 
    "tidyr", "doParallel", "foreach", "rpart", "mgcv")

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_pkgs)) {
    stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "), ". Install dependencies before running.")
}

suppressPackageStartupMessages({
    library(caret)
    library(recipes)
    library(glmnet)
    library(Boruta)
    library(pROC)
    library(ggplot2)
    library(dplyr)
    library(reshape2)
    library(shapviz)
    library(kernelshap)
    library(patchwork)
    library(gridExtra)
    library(doParallel)
})

set.seed(SEED)

cl <- NULL

if (N_CORES > 1L) {
    cl <- parallel::makePSOCKcluster(N_CORES)
    parallel::clusterCall(cl, function() {
        invisible(NULL)
    })
    doParallel::registerDoParallel(cl)
} else {
    foreach::registerDoSEQ()
}

okabe_ito <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442", 
    "#999999", "#332288", "#88CCEE", "#44AA99", "#117733", "#DDCC77")

theme_pub <- theme_bw(base_size = 11) + theme(text = element_text(family = if (.Platform$OS.type == "windows") "sans" else "sans"), 
    panel.grid.minor = element_blank(), panel.grid.major = element_line(color = "grey92", linewidth = 0.3), 
    plot.title = element_text(face = "bold", hjust = 0.5), axis.title = element_text(face = "bold"), 
    legend.title = element_text(face = "bold"), strip.background = element_rect(fill = "grey95", color = "grey70"), 
    strip.text = element_text(face = "bold"))

theme_set(theme_pub)

save_plot2 <- function(plot, stem, width = 8, height = 6) {
    ggsave(file.path(FIG_DIR, paste0(stem, ".pdf")), plot, width = width, height = height, device = cairo_pdf, 
        limitsize = FALSE)
    grDevices::png(file.path(FIG_DIR, paste0(stem, ".png")), width = width, height = height, units = "in", 
        res = 300, bg = "white")
    print(plot)
    grDevices::dev.off()
}

if (!file.exists(DATA_FILE)) stop("Input file not found: ", DATA_FILE)

dat <- read.csv(DATA_FILE, check.names = FALSE)

clean_names <- sub("^﻿", "", names(dat), useBytes = TRUE)

Encoding(clean_names) <- "UTF-8"

names(dat) <- clean_names

if (!OUTCOME_COL %in% names(dat)) stop("Outcome column not found: ", OUTCOME_COL)

for (j in seq_along(dat)) {
    if (is.character(dat[[j]])) {
        z <- trimws(dat[[j]])
        z[z %in% c("", "NA", "N/A", "#N/A", "NULL", "null")] <- NA
        dat[[j]] <- z
    }
}

y_raw <- dat[[OUTCOME_COL]]

if (is.factor(y_raw)) y_raw <- as.character(y_raw)

y_raw <- trimws(as.character(y_raw))

y_raw[y_raw %in% c("Yes", "yes")] <- "1"

y_raw[y_raw %in% c("No", "no")] <- "0"

if (!all(na.omit(unique(y_raw)) %in% c("0", "1"))) stop("Outcome must be coded as 0/1.")

if (anyNA(y_raw)) stop("Outcome values must be complete.")

dat[[OUTCOME_COL]] <- factor(ifelse(y_raw == "1", "Yes", "No"), levels = c("Yes", "No"))

excluded_predictors <- trimws(strsplit(Sys.getenv("ML_EXCLUDE_PREDICTORS", unset = ""), ",", fixed = TRUE)[[1]])

excluded_predictors <- excluded_predictors[nzchar(excluded_predictors)]

unknown_exclusions <- setdiff(excluded_predictors, names(dat))

if (length(unknown_exclusions)) {
    stop("Requested excluded predictors were not found: ", paste(unknown_exclusions, collapse = ", "))
}

predictor_cols <- setdiff(names(dat), c(OUTCOME_COL, excluded_predictors))

if (length(excluded_predictors)) {
    cat("Excluded before imputation and feature selection: ", paste(excluded_predictors, collapse = ", "), 
        "\n", sep = "")
}

is_binary01 <- function(x) {
    ux <- unique(na.omit(as.character(x)))
    length(ux) >= 1 && length(ux) <= 2 && all(ux %in% c("0", "1"))
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

mode_value <- function(x) {
    x <- x[!is.na(x)]
    if (!length(x)) 
        return(NA)
    names(sort(table(x), decreasing = TRUE))[1]
}

fit_imputer <- function(x) {
    list(numeric = lapply(x[vapply(x, is.numeric, logical(1))], function(z) median(z, na.rm = TRUE)), 
        factor = lapply(x[vapply(x, function(z) is.factor(z) || is.character(z), logical(1))], mode_value), 
        levels = lapply(x[vapply(x, is.factor, logical(1))], levels))
}

apply_imputer <- function(x, imp) {
    x <- as.data.frame(x, check.names = FALSE)
    for (v in intersect(names(imp$numeric), names(x))) {
        fill <- imp$numeric[[v]]
        if (!is.finite(fill)) 
            fill <- 0
        x[[v]][is.na(x[[v]])] <- fill
    }
    for (v in intersect(names(imp$factor), names(x))) {
        z <- as.character(x[[v]])
        z[is.na(z) | z == ""] <- as.character(imp$factor[[v]])
        lev <- unique(c(imp$levels[[v]] %||% character(), z))
        x[[v]] <- factor(z, levels = lev)
    }
    x
}

active_predictors <- function(x) {
    names(x)[vapply(x, function(z) length(unique(na.omit(z))) > 1, logical(1))]
}

fit_selectors <- function(train_df, seed, make_plots = FALSE, plot_prefix = "") {
    x0 <- train_df[, predictor_cols, drop = FALSE]
    y <- train_df[[OUTCOME_COL]]
    imp <- fit_imputer(x0)
    x <- apply_imputer(x0, imp)
    active <- active_predictors(x)
    x <- x[, active, drop = FALSE]
    y01 <- as.integer(y == "Yes")
    lasso_vars <- character()
    lasso_fit <- NULL
    lasso_cv <- NULL
    lambda_used <- NA_real_
    cf <- matrix(numeric(), nrow = 0, ncol = 1)
    set.seed(seed + 2)
    boruta_raw <- Boruta::Boruta(x = x, y = y, doTrace = 0, maxRuns = BORUTA_MAX_RUNS)
    boruta_fit <- tryCatch(Boruta::TentativeRoughFix(boruta_raw), error = function(e) boruta_raw)
    boruta_vars <- Boruta::getSelectedAttributes(boruta_fit, withTentative = FALSE)
    boruta_stats <- Boruta::attStats(boruta_fit)
    boruta_stats$variable <- rownames(boruta_stats)
    if (make_plots) {
        grDevices::cairo_pdf(file.path(FIG_DIR, paste0(plot_prefix, "Boruta_importance.pdf")), width = 12, 
            height = 7)
        par(mar = c(10, 4.5, 2.5, 1) + 0.1, family = if (.Platform$OS.type == "windows") 
            "sans"
        else "sans")
        plot(boruta_fit, las = 2, cex.axis = 0.65, xlab = "", main = "Boruta variable importance")
        dev.off()
        png(file.path(FIG_DIR, paste0(plot_prefix, "Boruta_importance.png")), width = 12, height = 7, 
            units = "in", res = 300, type = "cairo")
        par(mar = c(10, 4.5, 2.5, 1) + 0.1, family = if (.Platform$OS.type == "windows") 
            "sans"
        else "sans")
        plot(boruta_fit, las = 2, cex.axis = 0.65, xlab = "", main = "Boruta variable importance")
        dev.off()
    }
    list(lasso = lasso_vars, boruta = boruta_vars, intersection = intersect(lasso_vars, boruta_vars), 
        imputer = imp, lasso_fit = lasso_fit, lasso_cv = lasso_cv, lambda = lambda_used, lasso_coef = cf, 
        boruta_fit = boruta_fit, boruta_stats = boruta_stats)
}

safe_feature_set <- function(vars, fallback) {
    vars <- intersect(vars, predictor_cols)
    if (!length(vars)) 
        fallback
    else vars
}

make_recipe <- function(train_df, vars, corr_threshold = 1, keep_nominal = FALSE) {
    vars <- intersect(vars, predictor_cols)
    rec_dat <- train_df[, c(OUTCOME_COL, vars), drop = FALSE]
    rec <- recipes::step_novel(recipes::step_unknown(recipes::step_YeoJohnson(recipes::step_impute_mode(recipes::step_impute_median(recipes::recipe(stats::reformulate(vars, 
        response = OUTCOME_COL), data = rec_dat), recipes::all_numeric_predictors()), recipes::all_nominal_predictors()), 
        recipes::all_numeric_predictors()), recipes::all_nominal_predictors(), new_level = "missing"), 
        recipes::all_nominal_predictors(), new_level = "novel")
    if (!isTRUE(keep_nominal)) {
        rec <- recipes::step_dummy(rec, recipes::all_nominal_predictors(), one_hot = FALSE)
    }
    rec <- recipes::step_zv(rec, recipes::all_predictors())
    if (is.finite(corr_threshold) && corr_threshold < 1) {
        rec <- recipes::step_corr(rec, recipes::all_numeric_predictors(), threshold = corr_threshold)
    }
    recipes::step_normalize(rec, recipes::all_numeric_predictors())
}

two_class_summary_safe <- function(data, lev = NULL, model = NULL) {
    out <- caret::twoClassSummary(data, lev = lev, model = model)
    if (!all(is.finite(out))) 
        out[!is.finite(out)] <- NA_real_
    out
}

make_inner_control <- function(y, seed) {
    set.seed(seed)
    idx <- caret::createFolds(y, k = min(INNER_FOLDS, min(table(y))), returnTrain = TRUE)
    idx_out <- lapply(idx, function(i) setdiff(seq_along(y), i))
    caret::trainControl(method = "cv", number = length(idx), index = idx, indexOut = idx_out, classProbs = TRUE, 
        summaryFunction = two_class_summary_safe, savePredictions = "final", allowParallel = TRUE, verboseIter = FALSE)
}

custom_catboost <- list(label = "CatBoost", library = "catboost", type = c("Classification"), parameters = data.frame(parameter = c("iterations", 
    "depth", "learning_rate", "l2_leaf_reg"), class = rep("numeric", 4), label = c("Iterations", "Depth", 
    "Learning rate", "L2 regularization")), grid = function(x, y, len = NULL, search = "grid") data.frame(), 
    fit = function(x, y, wts, param, lev, last, classProbs, ...) {
        label <- as.integer(y == lev[1])
        pool <- catboost::catboost.load_pool(as.matrix(x), label = label)
        fit <- catboost::catboost.train(pool, NULL, params = list(loss_function = "Logloss", eval_metric = "AUC", 
            verbose = 0, random_seed = SEED, thread_count = 1, iterations = as.integer(param$iterations), 
            depth = as.integer(param$depth), learning_rate = param$learning_rate, l2_leaf_reg = param$l2_leaf_reg))
        list(model = fit, obsLevels = lev)
    }, predict = function(modelFit, newdata, submodels = NULL) {
        lev <- modelFit$obsLevels
        p <- catboost::catboost.predict(modelFit$model, catboost::catboost.load_pool(as.matrix(newdata)), 
            prediction_type = "Probability")
        factor(ifelse(p >= 0.5, lev[1], lev[2]), levels = lev)
    }, prob = function(modelFit, newdata, submodels = NULL) {
        lev <- modelFit$obsLevels
        p <- catboost::catboost.predict(modelFit$model, catboost::catboost.load_pool(as.matrix(newdata)), 
            prediction_type = "Probability")
        out <- data.frame(p, 1 - p)
        names(out) <- lev
        out
    }, levels = function(x) x$obsLevels, sort = function(x) x[order(x$iterations, x$depth), ])

custom_xgboost <- list(label = "XGBoost", library = "xgboost", type = c("Classification"), parameters = data.frame(parameter = c("nrounds", 
    "max_depth", "eta", "gamma", "colsample_bytree", "min_child_weight", "subsample"), class = rep("numeric", 
    7), label = c("Iterations", "Depth", "Learning rate", "Gamma", "Column sample", "Min child weight", 
    "Row sample")), grid = function(x, y, len = NULL, search = "grid") data.frame(), fit = function(x, 
    y, wts, param, lev, last, classProbs, ...) {
    label <- as.integer(y == lev[1])
    ds <- xgboost::xgb.DMatrix(as.matrix(x), label = label)
    fit <- xgboost::xgb.train(params = list(objective = "binary:logistic", eval_metric = "auc", nthread = 1, 
        max_depth = as.integer(param$max_depth), eta = param$eta, gamma = param$gamma, colsample_bytree = param$colsample_bytree, 
        min_child_weight = param$min_child_weight, subsample = param$subsample), data = ds, nrounds = as.integer(param$nrounds), 
        verbose = 0)
    list(model = fit, obsLevels = lev)
}, predict = function(modelFit, newdata, submodels = NULL) {
    lev <- modelFit$obsLevels
    p <- predict(modelFit$model, xgboost::xgb.DMatrix(as.matrix(newdata)))
    factor(ifelse(p >= 0.5, lev[1], lev[2]), levels = lev)
}, prob = function(modelFit, newdata, submodels = NULL) {
    lev <- modelFit$obsLevels
    p <- predict(modelFit$model, xgboost::xgb.DMatrix(as.matrix(newdata)))
    out <- data.frame(p, 1 - p)
    names(out) <- lev
    out
}, levels = function(x) x$obsLevels, sort = function(x) x[order(x$nrounds, x$max_depth), ])

custom_glmnet <- list(label = "Elastic net logistic regression", library = "glmnet", type = c("Classification"), 
    parameters = data.frame(parameter = c("alpha", "lambda"), class = rep("numeric", 2), label = c("Mixing percentage", 
        "Regularization")), grid = function(x, y, len = NULL, search = "grid") data.frame(), fit = function(x, 
        y, wts, param, lev, last, classProbs, ...) {
        xx <- as.matrix(x)
        augmented <- ncol(xx) == 1
        if (augmented) xx <- cbind(xx, .aux_zero = 0)
        fit <- glmnet::glmnet(xx, as.integer(y == lev[1]), family = "binomial", alpha = param$alpha, 
            lambda = param$lambda, standardize = FALSE)
        list(model = fit, obsLevels = lev, lambda_used = param$lambda, augmented = augmented)
    }, predict = function(modelFit, newdata, submodels = NULL) {
        lev <- modelFit$obsLevels
        xx <- as.matrix(newdata)
        if (isTRUE(modelFit$augmented)) xx <- cbind(xx, .aux_zero = 0)
        p <- as.numeric(predict(modelFit$model, newx = xx, s = modelFit$lambda_used, type = "response"))
        factor(ifelse(p >= 0.5, lev[1], lev[2]), levels = lev)
    }, prob = function(modelFit, newdata, submodels = NULL) {
        lev <- modelFit$obsLevels
        xx <- as.matrix(newdata)
        if (isTRUE(modelFit$augmented)) xx <- cbind(xx, .aux_zero = 0)
        p <- as.numeric(predict(modelFit$model, newx = xx, s = modelFit$lambda_used, type = "response"))
        out <- data.frame(p, 1 - p)
        names(out) <- lev
        out
    }, levels = function(x) x$obsLevels, sort = function(x) x[order(x$alpha, x$lambda), ])

custom_klar_nb <- list(label = "Stable kernel Naive Bayes", library = "klaR", type = c("Classification"), 
    parameters = data.frame(parameter = c("fL", "usekernel", "adjust"), class = c("numeric", "logical", 
        "numeric"), label = c("Laplace correction", "Kernel density", "Bandwidth adjustment")), grid = function(x, 
        y, len = NULL, search = "grid") {
        rbind(expand.grid(fL = c(0, 1), usekernel = TRUE, adjust = c(1.1, 1.25, 1.4, 1.55, 1.7, 1.9)), 
            expand.grid(fL = c(0, 1), usekernel = FALSE, adjust = 1))
    }, fit = function(x, y, wts, param, lev, last, classProbs, ...) {
        xx <- if (is.data.frame(x)) x else as.data.frame(x, check.names = FALSE)
        fit <- klaR::NaiveBayes(x = xx, grouping = y, usekernel = isTRUE(param$usekernel), fL = as.numeric(param$fL), 
            adjust = as.numeric(param$adjust))
        fit$obsLevels <- lev
        fit
    }, prob = function(modelFit, newdata, submodels = NULL) {
        nd <- if (is.data.frame(newdata)) newdata else as.data.frame(newdata, check.names = FALSE)
        nd <- nd[, modelFit$varnames, drop = FALSE]
        lev <- modelFit$levels
        n <- nrow(nd)
        k_num <- length(lev)
        log_post <- matrix(log(pmax(as.numeric(modelFit$apriori), 1e-12)), nrow = n, ncol = k_num, byrow = TRUE, 
            dimnames = list(NULL, lev))
        for (j in seq_along(modelFit$varnames)) {
            v <- modelFit$varnames[j]
            train_v <- modelFit$x[[v]]
            tab <- modelFit$tables[[v]]
            if (is.numeric(train_v)) {
                vals <- as.numeric(nd[[v]])
                for (kk in seq_along(lev)) {
                  dens <- if (isTRUE(modelFit$usekernel)) {
                    ker <- tab[[lev[kk]]]
                    vapply(vals, function(z) klaR:::dkernel(x = z, kernel = ker), numeric(1))
                  } else {
                    stats::dnorm(vals, mean = tab[lev[kk], 1], sd = tab[lev[kk], 2])
                  }
                  dens[!is.finite(dens) | dens <= 0] <- 1e-12
                  log_post[, kk] <- log_post[, kk] + log(dens)
                }
            } else {
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
        out <- matrix(pmax(1e-12, pmin(1 - 1e-12, as.numeric(out))), nrow = n, ncol = k_num, dimnames = list(NULL, 
            lev))
        out <- out/rowSums(out)
        as.data.frame(out, check.names = FALSE)
    }, predict = function(modelFit, newdata, submodels = NULL) {
        pp <- custom_klar_nb$prob(modelFit, newdata)
        factor(modelFit$levels[max.col(as.matrix(pp), ties.method = "first")], levels = modelFit$levels)
    }, levels = function(x) x$obsLevels, sort = function(x) x[order(!x$usekernel, x$fL, x$adjust), , 
        drop = FALSE])

custom_gam <- list(label = "Regularized generalized additive logistic model", library = "mgcv", type = c("Classification"), 
    parameters = data.frame(parameter = c("k", "gamma"), class = c("numeric", "numeric"), label = c("Maximum smooth basis dimension", 
        "Smoothness penalty multiplier")), grid = function(x, y, len = NULL, search = "grid") expand.grid(k = c(3, 
        4), gamma = c(1, 1.25, 1.5)), fit = function(x, y, wts, param, lev, last, classProbs, ...) {
        xx <- if (is.data.frame(x)) x else as.data.frame(x, check.names = FALSE)
        dat_gam <- xx
        dat_gam$.outcome <- as.integer(y == lev[1])
        quote_name <- function(z) paste0("`", gsub("`", "", z), "`")
        terms <- vapply(names(xx), function(v) {
            z <- xx[[v]]
            if (is.numeric(z) && length(unique(z[is.finite(z)])) > 4L) {
                paste0("s(", quote_name(v), ", k=", as.integer(param$k), ", bs='cr')")
            } else quote_name(v)
        }, character(1))
        form <- stats::as.formula(paste(".outcome ~", paste(terms, collapse = " + ")))
        fit <- mgcv::gam(form, data = dat_gam, family = stats::binomial(), method = "REML", select = TRUE, 
            gamma = as.numeric(param$gamma))
        fit$obsLevels <- lev
        fit
    }, prob = function(modelFit, newdata, submodels = NULL) {
        p <- as.numeric(stats::predict(modelFit, newdata = as.data.frame(newdata), type = "response"))
        p <- pmax(1e-12, pmin(1 - 1e-12, p))
        out <- data.frame(p, 1 - p, check.names = FALSE)
        names(out) <- modelFit$obsLevels
        out
    }, predict = function(modelFit, newdata, submodels = NULL) {
        p <- custom_gam$prob(modelFit, newdata)[[modelFit$obsLevels[1]]]
        factor(ifelse(p >= 0.5, modelFit$obsLevels[1], modelFit$obsLevels[2]), levels = modelFit$obsLevels)
    }, levels = function(x) x$obsLevels, sort = function(x) x[order(x$k, x$gamma), , drop = FALSE])

custom_rf_regularized <- list(label = "Regularized random forest", library = "randomForest", type = c("Classification"), 
    parameters = data.frame(parameter = c("mtry", "nodesize", "maxnodes"), class = rep("numeric", 3), 
        label = c("Variables per split", "Minimum terminal node size", "Maximum terminal nodes")), grid = function(x, 
        y, len = NULL, search = "grid") expand.grid(mtry = seq_len(min(3, ncol(x))), nodesize = c(8, 
        15, 25), maxnodes = c(6, 10, 16)), fit = function(x, y, wts, param, lev, last, classProbs, ...) {
        fit <- randomForest::randomForest(x = x, y = y, ntree = 1000, mtry = min(as.integer(param$mtry), 
            ncol(x)), nodesize = as.integer(param$nodesize), maxnodes = as.integer(param$maxnodes), importance = TRUE)
        fit$obsLevels <- lev
        fit
    }, prob = function(modelFit, newdata, submodels = NULL) as.data.frame(stats::predict(modelFit, newdata, 
        type = "prob"), check.names = FALSE), predict = function(modelFit, newdata, submodels = NULL) stats::predict(modelFit, 
        newdata, type = "response"), levels = function(x) x$obsLevels, sort = function(x) x[order(x$mtry, 
        x$nodesize, x$maxnodes), , drop = FALSE])

sample_rows <- function(df, n, seed) {
    if (nrow(df) <= n) 
        return(df)
    set.seed(seed)
    df[sample(seq_len(nrow(df)), n), , drop = FALSE]
}

build_grid <- function(impl, p, seed) {
    set.seed(seed)
    if (impl == "custom_rf_regularized") {
        return(sample_rows(expand.grid(mtry = seq_len(min(3, p)), nodesize = c(8, 15, 25), maxnodes = c(6, 
            10, 16)), min(TUNE_BUDGET, 27), seed))
    }
    if (impl == "rf") {
        return(data.frame(mtry = unique(pmax(1, pmin(p, round(seq(1, p, length.out = min(TUNE_BUDGET, 
            p))))))))
    }
    if (impl == "custom_xgboost") {
        return(sample_rows(expand.grid(nrounds = c(120, 220, 350), max_depth = c(1, 2), eta = c(0.01, 
            0.025, 0.05), gamma = c(1, 3, 5), colsample_bytree = c(0.6, 0.8), min_child_weight = c(5, 
            10, 15), subsample = c(0.7, 0.85)), min(TUNE_BUDGET, 24), seed))
    }
    if (impl == "svmRadial") {
        return(data.frame(sigma = 2^seq(-8, -2, length.out = TUNE_BUDGET), C = 2^seq(-2, 5, length.out = TUNE_BUDGET)))
    }
    if (impl == "glm") 
        return(data.frame(parameter = "none"))
    if (impl == "knn") 
        return(data.frame(k = unique(seq(3, min(31, max(3, p * 3)), by = 2))))
    if (impl == "custom_pls") 
        return(data.frame(ncomp = seq_len(max(1, min(p, TUNE_BUDGET)))))
    if (impl == "gbm") {
        return(sample_rows(expand.grid(n.trees = c(80, 150, 250), interaction.depth = 1:3, shrinkage = c(0.03, 
            0.08), n.minobsinnode = c(10, 20)), TUNE_BUDGET, seed))
    }
    if (impl == "nnet") {
        return(sample_rows(expand.grid(size = c(1, 3, 5, 7), decay = c(0.001, 0.01, 0.1, 1)), TUNE_BUDGET, 
            seed))
    }
    if (impl == "nb") {
        return(sample_rows(expand.grid(fL = c(0, 1), usekernel = c(TRUE, FALSE), adjust = c(0.5, 0.75, 
            1, 1.25, 1.5)), TUNE_BUDGET, seed))
    }
    if (impl == "custom_klar_nb") {
        return(rbind(expand.grid(fL = c(0, 1), usekernel = TRUE, adjust = c(1.1, 1.25, 1.4, 1.55, 1.7, 
            1.9)), expand.grid(fL = c(0, 1), usekernel = FALSE, adjust = 1)))
    }
    if (impl == "custom_stable_nb") {
        return(expand.grid(var_smoothing = c(1e-06, 1e-04, 0.001, 0.01, 0.05), variance_shrinkage = c(0, 
            0.25, 0.5, 0.75, 1), prior_shrinkage = c(0, 0.25, 0.5)))
    }
    if (impl == "custom_gam") {
        return(expand.grid(k = c(3, 4), gamma = c(1, 1.25, 1.5)))
    }
    if (impl == "lda") 
        return(data.frame(parameter = "none"))
    if (impl %in% c("custom_glmnet", "custom_ridge", "custom_lasso", "custom_elastic")) {
        alpha_values <- switch(impl, custom_ridge = 0, custom_lasso = 1, custom_elastic = c(0.25, 0.5, 
            0.75), c(0, 0.25, 0.5, 0.75, 1))
        base_grid <- expand.grid(alpha = alpha_values, lambda = 10^seq(-4, 0.5, length.out = 14))
        return(base_grid)
    }
    if (impl == "AdaBoost.M1") {
        return(sample_rows(expand.grid(mfinal = c(50, 100, 150), maxdepth = 1:2, coeflearn = c("Breiman", 
            "Freund", "Zhu")), TUNE_BUDGET, seed))
    }
    if (impl == "custom_catboost") {
        return(sample_rows(expand.grid(iterations = c(150, 300, 500), depth = c(2, 3), learning_rate = c(0.01, 
            0.03, 0.05), l2_leaf_reg = c(10, 20, 30)), min(12, TUNE_BUDGET), seed))
    }
    if (impl == "custom_lightgbm") {
        return(sample_rows(expand.grid(nrounds = c(150, 300), num_leaves = c(3, 7), learning_rate = c(0.02, 
            0.05), feature_fraction = c(0.6, 0.8), min_data_in_leaf = c(20, 30, 40)), min(12, TUNE_BUDGET), 
            seed))
    }
}

best_tune_predictions <- function(fit) {
    z <- fit$pred
    if (is.null(z) || !nrow(z)) 
        return(NULL)
    bt <- fit$bestTune
    for (v in intersect(names(bt), names(z))) {
        if (is.numeric(bt[[v]])) 
            z <- z[abs(z[[v]] - bt[[v]]) < 1e-12, , drop = FALSE]
        else z <- z[as.character(z[[v]]) == as.character(bt[[v]]), , drop = FALSE]
    }
    z
}

sensitivity_target_threshold <- function(truth, prob, target = TARGET_SENSITIVITY) {
    truth01 <- as.integer(truth == "Yes")
    if (length(unique(truth01)) < 2 || anyNA(prob)) 
        return(0.5)
    candidates <- sort(unique(prob), decreasing = TRUE)
    perf <- do.call(rbind, lapply(candidates, function(th) {
        pred <- prob >= th
        tp <- sum(pred & truth01 == 1)
        fn <- sum(!pred & truth01 == 1)
        tn <- sum(!pred & truth01 == 0)
        fp <- sum(pred & truth01 == 0)
        data.frame(threshold = th, sensitivity = if ((tp + fn) > 0) 
            tp/(tp + fn)
        else NA_real_, specificity = if ((tn + fp) > 0) 
            tn/(tn + fp)
        else NA_real_)
    }))
    eligible <- perf[is.finite(perf$sensitivity) & perf$sensitivity >= target, , drop = FALSE]
    if (!nrow(eligible)) 
        return(min(prob, na.rm = TRUE))
    eligible <- eligible[order(-eligible$specificity, -eligible$threshold), , drop = FALSE]
    max(0.01, min(0.99, eligible$threshold[1]))
}

fit_one_model <- function(train_df, valid_df, vars, algo, impl, seed, inner_control, corr_threshold = 1) {
    p_est <- max(2, length(vars))
    grid <- build_grid(impl, p_est, seed)
    method <- switch(impl, custom_xgboost = custom_xgboost, custom_glmnet = custom_glmnet, custom_ridge = custom_glmnet, 
        custom_lasso = custom_glmnet, custom_elastic = custom_glmnet, custom_catboost = custom_catboost, 
        custom_klar_nb = custom_klar_nb, custom_gam = custom_gam, custom_rf_regularized = custom_rf_regularized, 
        impl)
    if (startsWith(impl, "custom_")) 
        inner_control$allowParallel <- FALSE
    extra <- list()
    if (impl == "rf") 
        extra <- list(ntree = 500, importance = TRUE)
    if (impl == "gbm") 
        extra <- list(verbose = FALSE)
    if (impl == "nnet") 
        extra <- list(trace = FALSE, MaxNWts = 10000, maxit = 500)
    keep_nominal <- impl %in% c("custom_klar_nb", "custom_gam")
    rec <- make_recipe(train_df, vars, corr_threshold = corr_threshold, keep_nominal = keep_nominal)
    args <- c(list(x = rec, data = train_df[, c(OUTCOME_COL, vars), drop = FALSE], method = method, metric = "ROC", 
        trControl = inner_control, tuneGrid = grid), extra)
    set.seed(seed)
    fit <- do.call(caret::train, args)
    prob <- as.numeric(predict(fit, newdata = valid_df[, vars, drop = FALSE], type = "prob")[, "Yes"])
    inner_pred <- best_tune_predictions(fit)
    threshold <- if (is.null(inner_pred)) 
        0.5
    else sensitivity_target_threshold(inner_pred$obs, inner_pred$Yes, TARGET_SENSITIVITY)
    list(fit = fit, prob = prob, threshold = threshold, best_tune = fit$bestTune, inner_auc = max(fit$results$ROC, 
        na.rm = TRUE))
}
