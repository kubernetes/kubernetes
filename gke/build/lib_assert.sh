#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

__script_dir="$(dirname "$(realpath "$0")")"

# shellcheck source=./lib_log.sh
source "${__script_dir}"/lib_log.sh

assert_variable_equality()
{
  if [[ "${1}" != "${2}" ]]; then
    log.fail "assertion failure: \`${1}' is not equal to \`${2}'"
  fi
}

assert_variable_inequality()
{
  if [[ "${1}" == "${2}" ]]; then
    log.fail "assertion failure: \`${1}' is equal to \`${2}'"
  fi
}

assert_variable_not_empty()
{
  if [[ -z "${1}" ]]; then
    log.fail "assertion failure: variable \`${1}' cannot be empty"
  fi
}

assert_path_exists()
{
  if [[ ! -f "${1}" ]]; then
    log.fail "assertion failure: path \`${1}' does not exist"
  fi
}

assert_pathregex_exists_in_tar()
{
  local tarball="${1}"
  local pathregex="${2}"
  log.info "checking for ${pathregex} in ${tarball}"
  if ! tar tf "${tarball}" | grep "${pathregex}" >/dev/null 2>&1; then
    log.fail "could not detect ${pathregex} in ${tarball}"
  fi
}
