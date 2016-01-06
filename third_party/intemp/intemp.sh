#!/usr/bin/env bash

# Runs the supplied bash command string in a temporary workspace directory.
# Usage: intemp.sh [-t prefix] <command>
# Requires: mktemp

set -o errexit
set -o nounset
set -o pipefail

opt_flag=${1:-}
[ -z "${opt_flag}" ] && echo "No command supplied" >&2 && exit 1

if [ "${opt_flag}" == "-t" ]; then
  shift
  prefix=${1:-}
  [ -z "${prefix}" ] && echo "No prefix supplied" >&2 && exit 1
  shift
else
  prefix='temp'
fi

cmd="$1"
[ -z "${cmd}" ] && echo "No command supplied" >&2 && exit 1

workspace=$(mktemp -d "${TMPDIR:-/tmp}/${prefix}.XXXXXX")
echo "Workspace created: ${workspace}" 1>&2

cleanup() {
  local -r workspace="$1"
  rm -rf "${workspace}"
  echo "Workspace deleted: ${workspace}" 1>&2
}
trap "cleanup '${workspace}'" EXIT

pushd "${workspace}" > /dev/null
bash -ceu "${cmd}"
popd > /dev/null

trap - EXIT
cleanup "${workspace}"
