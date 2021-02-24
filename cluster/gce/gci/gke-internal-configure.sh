#!/bin/bash
NODE_PROBLEM_DETECTOR_RELEASE_PATH="${NODE_PROBLEM_DETECTOR_RELEASE_PATH:-https://storage.googleapis.com/kubernetes-release}"
NODE_PROBLEM_DETECTOR_LATEST_VERSION="${NODE_PROBLEM_DETECTOR_LATEST_VERSION:-v0.8.7}"
NODE_PROBLEM_DETECTOR_LATEST_TAR_HASH="${NODE_PROBLEM_DETECTOR_LATEST_TAR_HASH:-853576423077bf72e7bd8e96cd782cf272f7391379f8121650c1448531c0d3a0991dfbd0784a1157423976026806ceb14ca8fb35bac1249127dbf00af45b7eea}"
NODE_PROBLEM_DETECTOR_LATEST_RELEASE_PATH="${NODE_PROBLEM_DETECTOR_LATEST_RELEASE_PATH:-https://storage.googleapis.com/kubernetes-release}"

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
