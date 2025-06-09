# This script installs R packages. When installing BiocManager, the script updates all R packages
# currently installed (options update=TRUE, ask=FALSE in BiocManager::install).

default_repo <- "http://cran.us.r-project.org"

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = default_repo)
}
BiocManager::install(version = "3.18", update = FALSE, ask = FALSE)

# PLIER
devtools::install_github("wgmao/PLIER@fe4e9b23c47ee199afc5c984692df79ac5aabe80")

