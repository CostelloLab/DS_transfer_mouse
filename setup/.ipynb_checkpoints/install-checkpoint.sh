#!/bin/bash
set -eo pipefail
IFS=$'\n\t'

# This script installs dependencies for the project.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--packages-group)
      PACKAGES_GROUP="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ -z "${CONDA_DEFAULT_ENV}" ]; then
  echo "ERROR: no conda environment was activated."
  exit 1
fi

if [ -z "${PACKAGES_GROUP}" ]; then
  echo "ERROR: specify a group of packages to install with -g [group]."
  exit 1
fi

if [ "$PACKAGES_GROUP" = "r" ]; then
  echo "Installing R packages"

  # Install other R packages using Bioconductor
  TAR=$(which tar) conda run \
    --no-capture-output \
    Rscript ${SCRIPT_DIR}/envs/r-deps-bioconductor.r
fi

