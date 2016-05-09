#!/usr/bin/env bash
# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Update the Godeps/LICENSES.md document.
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
# Manage the state of LICENSE/COPYRIGHT files
# Default operation is to check to see if a file is in the state file.
#
# @optparam -a    Add the file to the state file
# @param    file  The file to check or add
# @return 1 when no file is found in state file
#
file_state () {
  local add=0
  case "$1" in
    -a) add=1;shift ;;
  esac
  local file=$1

  # If we're ignoring state, then return 1
  ((CREATE_MISSING)) && return 1

  # initialize if step 1
  if ((add)); then
    echo "${file}" >> ${GODEPS_STATE}
    return 0
  fi

  # Get return code from grep itself
  # Redirect stderr so that a missing state file returns 1 quietly
  egrep -wq "^${file}$" ${GODEPS_STATE} 2>/dev/null
}

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
  local package_root_url
  local dir_root
  local find_maxdepth
  local find_names
  local -a local_files=()
  local -a remote_files=()

  # Necessary to expand {}
  case ${type} in
      LICENSE) remote_files=(LICENSE{,.code,.txt,.md})
               find_names=(-iname 'licen[sc]e*')
               find_maxdepth=1
               # Sadly inconsistent in the wild, but mostly license files
               # containing copyrights, but no readme/notice files containing
               # licenses (except to "see license file")
               ensure_pattern="License|Copyright"
               ;;
    # We search readmes for copyrights and this includes notice files as well
    # Look in as many places as we find files matching
    COPYRIGHT) remote_files=(NOTICE{,.txt} README{,.md})
               find_names=(-iname 'notice*' -o -iname 'readme*')
               find_maxdepth=3
               ensure_pattern="Copyright"
               ;;
  esac

  # Start search at package root
  case ${package} in
    github.com/*|golang.org/*|bitbucket.org/*)
     package_root=$(echo ${package} |awk -F/ '{print $1"/"$2"/"$3 }')
     ;;
    *)
     package_root=$(echo ${package} |awk -F/ '{print $1"/"$2 }')
     ;;
  esac
  # if github.com, rewrite package root url, otherwise take as is
  package_root_url="${package_root/github.com/raw.githubusercontent.com}"

  # Find LOCAL files first - only root and package level
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
  FILE_CONTENT[${index}]=""
  for f in ${local_files[@]-}; do
    # Find some copyright info in any file and break
    if egrep -wq "${ensure_pattern}" "${f}"; then
      FILE_CONTENT[${index}]=$(cat "${f}")
      break
    fi
  done

  if [[ -z "${FILE_CONTENT[${index}]-}" ]]; then
    # When nothing is set at the package level, try package_root
    FILE_CONTENT[${index}]="${FILE_CONTENT[${package_root}-${type}]-}"
  fi

  if [[ -z "${FILE_CONTENT[${index}]-}" ]]; then
    # Last ditch attempt - see if we can get it from version control
    for f in ${remote_files[@]}; do
      file_state "${package_root_url}/master/${f}" && continue
      if ! FILE_CONTENT[${index}]="$(\
          curl --fail --retry 10 -s \
              https://${package_root_url}/master/${f})" || \
         ! $(echo "${FILE_CONTENT[${index}]-}" |\
          egrep -qw "${ensure_pattern}") ||
         [[ "${FILE_CONTENT[${index}]-}" =~ \<\ *html ]] ; then

        ((CREATE_MISSING)) || file_state -a "${package_root_url}/master/${f}"
        continue
      fi
    done
  fi
}


#############################################################################
# MAIN
#############################################################################
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

LICENSE_ROOT="${LICENSE_ROOT:-${KUBE_ROOT}}"
cd "${LICENSE_ROOT}"

# If CREATE_MISSING=1, the state file is ignored
CREATE_MISSING=0
if [[ ${1-} == "--create-missing" ]]; then
  CREATE_MISSING=1
  shift
fi

# Place to store the state of not-found files so we don't curl too much
GODEPS_STATE="Godeps/.license_file_state"

GODEPS_LICENSE_FILE=${1:-"Godeps/LICENSES"}
DEPS_DIR="vendor"
declare -Ag FILE_CONTENT


# Put the K8S LICENSE on top
(
echo "================================================================================"
echo "= Kubernetes licensed under: ="
echo
cat ${LICENSE_ROOT}/LICENSE
) > ${GODEPS_LICENSE_FILE}

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

  if [[ -z "${FILE_CONTENT[${PACKAGE}-LICENSE]-}" &&
        -z "${FILE_CONTENT[${PACKAGE}-COPYRIGHT]-}" ]]; then
    echo "UNKNOWN"
  else
    if [[ -n "${FILE_CONTENT[${PACKAGE}-LICENSE]-}" ]]; then
      echo "${FILE_CONTENT[${PACKAGE}-LICENSE]-}"
      echo
    fi
    if [[ -n "${FILE_CONTENT[${PACKAGE}-COPYRIGHT]-}" ]]; then
      echo "${FILE_CONTENT[${PACKAGE}-COPYRIGHT]-}" | sed -n '/Copyright /,$p'
    fi
  fi
done >> ${GODEPS_LICENSE_FILE}
