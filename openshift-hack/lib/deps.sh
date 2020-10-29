#!/usr/bin/env bash

# os::deps::path_with_shellcheck returns a path that includes shellcheck.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  The path that includes shellcheck.
function os::deps::path_with_shellcheck() {
  local path="${PATH}"
  if ! which shellcheck &> /dev/null; then
    local shellcheck_path="${TMPDIR:-/tmp}/shellcheck"
    mkdir -p "${shellcheck_path}"
    pushd "${shellcheck_path}" > /dev/null || exit 1
      # This version needs to match that required by
      # hack/verify-shellcheck.sh to avoid the use of docker.
      local version="v0.7.0"
      local tar_file="shellcheck-${version}.linux.x86_64.tar.xz"
      curl -LO "https://github.com/koalaman/shellcheck/releases/download/${version}/${tar_file}"
      tar xf "${tar_file}"
      path="${PATH}:$(pwd)/shellcheck-${version}"
    popd > /dev/null || exit 1
  fi
  echo "${path}"
}
readonly -f os::deps::path_with_shellcheck
