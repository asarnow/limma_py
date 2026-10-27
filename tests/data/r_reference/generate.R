# Regenerate the R limma reference outputs used by tests/test_r_reference.py.
# Run from this directory:  Rscript generate.R
# Fixtures fixA.csv / fixB.csv come from make_fixtures.py.
# Generated with R 4.5.3, limma 3.68.5.
suppressMessages(library(limma))

# fixA: genes x samples with NaNs, two groups of 4 -> full pipeline
X <- as.matrix(read.csv("fixA.csv"))
design <- cbind(G1 = rep(1:0, each = 4), G2 = rep(0:1, each = 4))
fit <- lmFit(X, design)
fit2 <- contrasts.fit(fit, makeContrasts(G2 - G1, levels = design))
hyper <- NULL
for (leg in c(TRUE, FALSE)) {
  eb <- eBayes(fit2, legacy = leg)
  tag <- if (leg) "legacy" else "default"
  write.csv(data.frame(logFC = eb$coefficients[, 1], t = eb$t[, 1],
                       P.Value = eb$p.value[, 1], B = eb$lods[, 1],
                       s2.post = eb$s2.post, df.total = eb$df.total,
                       df.residual = eb$df.residual),
            sprintf("R_fixA_%s.csv", tag), row.names = FALSE)
  hyper <- rbind(hyper, data.frame(fixture = "fixA", method = tag,
                                   df2 = eb$df.prior, scale = eb$s2.prior))
}

# fixB: raw scaled-F variances with unequal df1 -> estimators directly
B <- read.csv("fixB.csv")
o <- limma:::fitFDistUnequalDF1(B$x, B$df1)
hyper <- rbind(hyper, data.frame(fixture = "fixB", method = "default", df2 = o$df2, scale = o$scale))
o <- fitFDist(B$x, B$df1)
hyper <- rbind(hyper, data.frame(fixture = "fixB", method = "legacy", df2 = o$df2, scale = o$scale))
write.csv(hyper, "R_hyperparameters.csv", row.names = FALSE)
