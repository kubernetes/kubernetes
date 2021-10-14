#!/bin/bash
NPD_CUSTOM_PLUGINS_VERSION="${NPD_CUSTOM_PLUGINS_VERSION:-v1.0.4}"
NPD_CUSTOM_PLUGINS_TAR_HASH="${NPD_CUSTOM_PLUGINS_TAR_HASH:-b048ce6daf072a600d9d34997b1e23f8190976f902cf91e6e479aba89202c3ddc5116e2511ce95e842942ca93654f29fa377f1ad93d294f6d07c202d5352c9df}"
NPD_CUSTOM_PLUGINS_RELEASE_PATH="${NPD_CUSTOM_PLUGINS_RELEASE_PATH:-https://storage.googleapis.com/gke-release}"

M4A_APPARMOR_PROFILE_HASH="${M4A_APPARMOR_PROFILE_HASH:-cd84b52e756bee90b4a26612b372519ebf942bb3b6145d1928d3c1ae0faa4a17ea040f3e5f0429df9193dfcf84364d6a4ac56ebefb70420ae12579be5c5b5756}"
M4A_APPARMOR_RELEASE_PATH="${M4A_APPARMOR_RELEASE_PATH:-https://storage.googleapis.com/anthos-migrate-release}"
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

function install-m4a-apparmor-profile {
  local -r hash="${M4A_APPARMOR_PROFILE_HASH}"
  local -r release_path="${M4A_APPARMOR_RELEASE_PATH}"
  local -r profile="m4a-apparmor-profile"

  if type apparmor_parser; then
    echo "Downloading ${profile}."
    download-or-bust "${hash}" "${release_path}/artifacts/${profile}"
    # This call is expected to fail as the profile is likely not installed.
    # we are only using this as a safety measure against changes in Ubuntu
    # image and  installing a clean one
    sudo apparmor_parser --remove ${KUBE_HOME}/${profile} > /dev/null 2>&1 || true
    sudo apparmor_parser ${KUBE_HOME}/${profile}
  else
    echo "No apparmor_parser found, cannot install M4A apparmor profile"
  fi
}
