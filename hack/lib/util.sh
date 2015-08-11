#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

kube::util::sortable_date() {
  date "+%Y%m%d-%H%M%S"
}

# this mimics the behavior of linux realpath which is not shipped by default with
# mac OS X 
kube::util::realpath() {
  [[ $1 = /* ]] && echo "$1" | sed 's/\/$//' || echo "$PWD/${1#./}"  | sed 's/\/$//'
}

kube::util::wait_for_url() {
  local url=$1
  local prefix=${2:-}
  local wait=${3:-0.5}
  local times=${4:-25}

  which curl >/dev/null || {
    kube::log::usage "curl must be installed"
    exit 1
  }

  local i
  for i in $(seq 1 $times); do
    local out
    if out=$(curl -fs $url 2>/dev/null); then
      kube::log::status "On try ${i}, ${prefix}: ${out}"
      return 0
    fi
    sleep ${wait}
  done
  kube::log::error "Timed out waiting for ${prefix} to answer at ${url}; tried ${times} waiting ${wait} between each"
  return 1
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
kube::util::ensure-temp-dir() {
  if [[ -z ${KUBE_TEMP-} ]]; then
    KUBE_TEMP=$(mktemp -d 2>/dev/null || mktemp -d -t kubernetes.XXXXXX)
  fi
}

# This figures out the host platform without relying on golang.  We need this as
# we don't want a golang install to be a prerequisite to building yet we need
# this info to figure out where the final binaries are placed.
kube::util::host_platform() {
  local host_os
  local host_arch
  case "$(uname -s)" in
    Darwin)
      host_os=darwin
      ;;
    Linux)
      host_os=linux
      ;;
    *)
      kube::log::error "Unsupported host OS.  Must be Linux or Mac OS X."
      exit 1
      ;;
  esac

  case "$(uname -m)" in
    x86_64*)
      host_arch=amd64
      ;;
    i?86_64*)
      host_arch=amd64
      ;;
    amd64*)
      host_arch=amd64
      ;;
    arm*)
      host_arch=arm
      ;;
    i?86*)
      host_arch=x86
      ;;
    *)
      kube::log::error "Unsupported host arch. Must be x86_64, 386 or arm."
      exit 1
      ;;
  esac
  echo "${host_os}/${host_arch}"
}

kube::util::find-binary() {
  local lookfor="${1}"
  local host_platform="$(kube::util::host_platform)"
  local locations=(
    "${KUBE_ROOT}/_output/dockerized/bin/${host_platform}/${lookfor}"
    "${KUBE_ROOT}/_output/local/bin/${host_platform}/${lookfor}"
    "${KUBE_ROOT}/platforms/${host_platform}/${lookfor}"
  )
  local bin=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )
  echo -n "${bin}"
}

# Wait for background jobs to finish. Return with
# an error status if any of the jobs failed.
kube::util::wait-for-jobs() {
  local fail=0
  local job
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done
  return ${fail}
}

# Takes a binary to run $1 and then copies the results to $2.
# If the generated and original files are the same after filtering lines
# that match $3, copy is skipped.
kube::util::gen-doc() {
  local cmd="$1"
  local base_dest="$(kube::util::realpath $2)"
  local relative_doc_dest="$(echo $3 | sed 's/\/$//')"
  local dest="${base_dest}/${relative_doc_dest}"
  local skipprefix="${4:-}"

  # We do this in a tmpdir in case the dest has other non-autogenned files
  # We don't want to include them in the list of gen'd files
  local tmpdir="${KUBE_ROOT}/doc_tmp"
  mkdir -p "${tmpdir}"
  # generate the new files
  ${cmd} "${tmpdir}"
  # create the list of generated files
  ls "${tmpdir}" | LC_ALL=C sort > "${tmpdir}/.files_generated"

  while read file; do
    # Don't add analytics link to generated .md files -- that is done (and
    # checked) by mungedocs.

    # Remove all old generated files from the destination
    if [[ -e "${tmpdir}/${file}" ]]; then
      local original generated
      # Filter all munges from original content.
      original=$(cat "${dest}/${file}" | sed '/^<!-- BEGIN MUNGE:.*/,/^<!-- END MUNGE:.*/d')
      generated=$(cat "${tmpdir}/${file}")
      # If this function was asked to filter lines with a prefix, do so.
      if [[ -n "${skipprefix}" ]]; then
        original=$(echo "${original}" | grep -v "^${skipprefix}" || :)
        generated=$(echo "${generated}" | grep -v "^${skipprefix}" || :)
      fi
      # By now, the contents should be normalized and stripped of any
      # auto-managed content.  We also ignore whitespace here because of
      # markdown strictness fixups later in the pipeline.
      if diff -Bw <(echo "${original}") <(echo "${generated}") > /dev/null; then
        # actual contents same, overwrite generated with original.
        mv "${dest}/${file}" "${tmpdir}/${file}"
      fi
    else
      rm "${dest}/${file}" || true
    fi
  done <"${dest}/.files_generated"

  # put the new generated file into the destination
  # the shopt is so that we get .files_generated from the glob.
  shopt -s dotglob
  cp -af "${tmpdir}"/* "${dest}"
  shopt -u dotglob

  #cleanup
  rm -rf "${tmpdir}"
}

# Takes a path $1 to traverse for md files to append the ga-beacon tracking
# link to, if needed. If $2 is set, just print files that are missing
# the link.
kube::util::gen-analytics() {
  local path="$1"
  local dryrun="${2:-}"
  local mdfiles dir link
  # find has some strange inconsistencies between darwin/linux. The
  # path to search must end in '/' for linux, but darwin will put an extra
  # slash in results if there is a trailing '/'.
  if [[ $( uname ) == 'Linux' ]]; then
    dir="${path}/"
  else
    dir="${path}"
  fi
  # We don't touch files in Godeps|third_party, and the kubectl
  # docs are autogenerated by gendocs.
  mdfiles=($( find "${dir}" -name "*.md" -type f \
              -not -path "${path}/Godeps/*" \
              -not -path "${path}/third_party/*" \
              -not -path "${path}/_output/*" \
              -not -path "${path}/docs/user-guide/kubectl/kubectl*" ))
  for f in "${mdfiles[@]}"; do
    link=$(kube::util::analytics-link "${f#${path}/}")
    if grep -q -F -x "${link}" "${f}"; then
      continue
    elif [[ -z "${dryrun}" ]]; then
      echo -e "\n\n${link}" >> "${f}"
    else
      echo "$f"
    fi
  done
}

# Prints analytics link to append to a file at path $1.
kube::util::analytics-link() {
  local path="$1"
  echo "[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/${path}?pixel)]()"
}

# ex: ts=2 sw=2 et filetype=sh
