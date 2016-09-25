#!/usr/bin/env bash
# Copyright 2015 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Update the Godeps/LICENSES document.
# Generates a table of Godep dependencies and their license.
#
# Usage:
#    $0 [--create-missing] [/path/to/licenses]
#
#    --create-missing will write the files that only exist upstream, locally.
#    This option is mostly used for testing as we cannot check-in any of the
#    additionally created files into the godep auto-generated tree.
#
#    Run every time a license file is added/modified within /Godeps to
#    update /Godeps/LICENSES

set -o errexit
set -o nounset
set -o pipefail

export LANG=C
export LC_ALL=C

###############################################################################
# Process package content
#
# @param package  The incoming package name
# @param type     The type of content (LICENSE or COPYRIGHT)
#
process_content () {
  local package=$1
  local type=$2

  local package_root
  local ensure_pattern
  local dir_root
  local find_maxdepth
  local find_names
  local -a local_files=()

  # Necessary to expand {}
  case ${type} in
      LICENSE) find_names=(-iname 'licen[sc]e*')
               find_maxdepth=1
               # Sadly inconsistent in the wild, but mostly license files
               # containing copyrights, but no readme/notice files containing
               # licenses (except to "see license file")
               ensure_pattern="License|Copyright"
               ;;
    # We search READMEs for copyrights and this includes notice files as well
    # Look in as many places as we find files matching
    COPYRIGHT) find_names=(-iname 'notice*' -o -iname 'readme*')
               find_maxdepth=3
               ensure_pattern="Copyright"
               ;;
  esac

  # Start search at package root
  case ${package} in
    github.com/*|golang.org/*|bitbucket.org/*)
     package_root=$(echo ${package} |awk -F/ '{ print $1"/"$2"/"$3 }')
     ;;
    go4.org/*)
     package_root=$(echo ${package} |awk -F/ '{ print $1 }')
     ;;
    *)
     package_root=$(echo ${package} |awk -F/ '{ print $1"/"$2 }')
     ;;
  esac

  # Find files - only root and package level
  local_files=($(
    for dir_root in ${package} ${package_root}; do
      [[ -d ${DEPS_DIR}/${dir_root} ]] || continue

      # One (set) of these is fine
      find ${DEPS_DIR}/${dir_root} \
          -xdev -follow -maxdepth ${find_maxdepth} \
          -type f "${find_names[@]}"
    done | sort -u))

  local index
  local f
  index="${package}-${type}"
  if [[ -z "${CONTENT[${index}]-}" ]]; then
    for f in ${local_files[@]-}; do
      # Find some copyright info in any file and break
      if egrep -wq "${ensure_pattern}" "${f}"; then
        CONTENT[${index}]="${f}"
        break
      fi
    done
  fi
}


#############################################################################
# MAIN
#############################################################################
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# This variable can be injected, as in the verify script.
LICENSE_ROOT="${LICENSE_ROOT:-${KUBE_ROOT}}"
cd "${LICENSE_ROOT}"

GODEPS_LICENSE_FILE="Godeps/LICENSES"
TMP_LICENSE_FILE="/tmp/Godeps.LICENSES.$$"
DEPS_DIR="vendor"
declare -Ag CONTENT

# Put the K8S LICENSE on top
(
echo "================================================================================"
echo "= Kubernetes licensed under: ="
echo
cat ${LICENSE_ROOT}/LICENSE
echo
echo "= LICENSE $(cat ${LICENSE_ROOT}/LICENSE | md5sum)"
echo "================================================================================"
) > ${TMP_LICENSE_FILE}

# Loop through every package in Godeps.json
for PACKAGE in $(cat Godeps/Godeps.json | \
                 jq -r ".Deps[].ImportPath" | \
                 sort -f); do
  process_content ${PACKAGE} LICENSE
  process_content ${PACKAGE} COPYRIGHT

  # display content
  echo
  echo "================================================================================"
  echo "= ${DEPS_DIR}/${PACKAGE} licensed under: ="
  echo

  file=""
  if [[ -n "${CONTENT[${PACKAGE}-LICENSE]-}" ]]; then
      file="${CONTENT[${PACKAGE}-LICENSE]-}"
  elif [[ -n "${CONTENT[${PACKAGE}-COPYRIGHT]-}" ]]; then
      file="${CONTENT[${PACKAGE}-COPYRIGHT]-}"
  fi
  if [[ -z "${file}" ]]; then
      cat > /dev/stderr << __EOF__
No license could be found for ${PACKAGE} - aborting.

Options:
1. Check if the upstream repository has a newer version with LICENSE and/or
   COPYRIGHT files.
2. Contact the author of the package to ensure there is a LICENSE and/or
   COPYRIGHT file present.
3. Do not use this package in Kubernetes.
__EOF__
      exit 9
  fi
  cat "${file}"

  echo
  echo "= ${file} $(cat ${file} | md5sum)"
  echo "================================================================================"
  echo
done >> ${TMP_LICENSE_FILE}

cat ${TMP_LICENSE_FILE} > ${GODEPS_LICENSE_FILE}
