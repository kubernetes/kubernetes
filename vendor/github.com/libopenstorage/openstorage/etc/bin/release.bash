#!/bin/bash

set -e

DIR="$(cd "$(dirname "${0}")/../.." && pwd)"
cd "${DIR}"

error() {
  echo "error: $@" >&2
  exit 1
}

check_env_set() {
  if [ -z "${!1}" ]; then
    error "${1} not set"
  fi
}

execute() {
  echo "$@" >&2
  if [ -z "${DRY_RUN}" ]; then
    $@
  fi
}

check_env_set "GITHUB_TOKEN"
check_env_set "GITHUB_RELEASE_TAG"
check_env_set "GITHUB_RELEASE_DESCRIPTION"

execute go get -u -v github.com/aktau/github-release

execute git checkout release
execute git merge master
execute git push origin release
execute git tag "${GITHUB_RELEASE_TAG}"
execute git push --tags

execute github-release release \
  --user "libopenstorage" \
  --repo "openstorage" \
  --tag "${GITHUB_RELEASE_TAG}" \
  --description "${GITHUB_RELEASE_DESCRIPTION}"
