pkgs <- c("dplyr", "readr")
install.packages(pkgs)

missing <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(missing) > 0) {
  stop(paste("Failed to install:", paste(missing, collapse = ", ")))
}
cat("All packages installed:", paste(pkgs, collapse = ", "), "\n")