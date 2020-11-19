#!/bin/bash

NODE_PROBLEM_DETECTOR_VERSION="${NODE_PROBLEM_DETECTOR_VERSION:-v0.8.5}"
NODE_PROBLEM_DETECTOR_TAR_HASH="${NODE_PROBLEM_DETECTOR_TAR_HASH:-3fbf97a38c06d8fcc8c46f956a6e90aa1b47cb42d50ddcfd1a644a7e624e42ee523db2f81e08fbfb21b80142d4bafdbedce16e8b62d2274a5b2b703a56d9c015}"
NODE_PROBLEM_DETECTOR_RELEASE_PATH="${NODE_PROBLEM_DETECTOR_RELEASE_PATH:-https://storage.googleapis.com/kubernetes-release}"

NPD_CUSTOM_PLUGINS_VERSION="${NPD_CUSTOM_PLUGINS_VERSION:-v1.0.1}"
NPD_CUSTOM_PLUGINS_TAR_HASH="${NPD_CUSTOM_PLUGINS_TAR_HASH:-6d3cdcced8cab4c6631b32bff2241e81a86949163dde58c8464d3cfb1fa8508c5595ec18f36fc636f5db4bf5458bfc8668d850eb68298e3bc39f83e6355d2c83}"
NPD_CUSTOM_PLUGINS_RELEASE_PATH="${NPD_CUSTOM_PLUGINS_RELEASE_PATH:-https://storage.googleapis.com/gke-release}"

# Install node problem detector custom plugins.
function install-npd-custom-plugins {
  local -r version="${NPD_CUSTOM_PLUGINS_VERSION}"
  local -r hash="${NPD_CUSTOM_PLUGINS_TAR_HASH}"
  local -r release_path="${NPD_CUSTOM_PLUGINS_RELEASE_PATH}"
  local -r tar="npd-custom-plugins-${version}.tar.gz"

  echo "Downloading ${tar}."
  download-or-bust "${hash}" "${release_path}/npd-custom-plugins/${version}/${tar}"
  local -r dir="${KUBE_HOME}/npd-custom-plugins"
  mkdir -p "${dir}"
  tar xzf "${KUBE_HOME}/${tar}" -C "${dir}" --overwrite
}
