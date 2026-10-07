stable_nb_probability <- function(modelFit, newdata, submodels = NULL) {
    nd <- if (is.data.frame(newdata)) 
        newdata
    else as.data.frame(newdata, check.names = FALSE)
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
    out <- matrix(pmax(1e-12, pmin(1 - 1e-12, as.numeric(out))), nrow = n, ncol = k_num, dimnames = list(NULL, 
        lev))
    out <- out/rowSums(out)
    as.data.frame(out, check.names = FALSE)
}

predict_bundle <- function(bundle, newdata) {
    baked <- recipes::bake(bundle$preprocessor, new_data = newdata)
    probability <- stable_nb_probability(bundle$model, baked)
    as.numeric(probability[, "Yes"])
}

export_bundle <- function(fit, data, recipe_input_features, final_features, metadata) {
    model <- fit$finalModel[c("levels", "apriori", "tables", "varnames", "usekernel")]
    model$x <- fit$finalModel$x[0, , drop = FALSE]
    preprocessor <- fit$recipe
    preprocessor$retained <- NULL
    if (!is.null(preprocessor$template)) {
        preprocessor$template <- preprocessor$template[0, , drop = FALSE]
    }
    list(model = model, preprocessor = preprocessor, input_schema = data[0, recipe_input_features, drop = FALSE], 
        final_features = final_features, feature_metadata = metadata)
}
