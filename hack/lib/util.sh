#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

kube::util::wait_for_url() {
  local url=$1
  local prefix=${2:-}
  local wait=${3:-1}
  local times=${4:-30}

  which curl >/dev/null || {
    kube::log::usage "curl must be installed"
    exit 1
  }

  local i
  for i in $(seq 1 $times); do
    local out
    if out=$(curl -gfs $url 2>/dev/null); then
      kube::log::status "On try ${i}, ${prefix}: ${out}"
      return 0
    fi
    sleep ${wait}
  done
  kube::log::error "Timed out waiting for ${prefix} to answer at ${url}; tried ${times} waiting ${wait} between each"
  return 1
}

# returns a random port
kube::util::get_random_port() {
  awk -v min=1024 -v max=65535 'BEGIN{srand(); print int(min+rand()*(max-min+1))}'
}

# use netcat to check if the host($1):port($2) is free (return 0 means free, 1 means used)
kube::util::test_host_port_free() {
  local host=$1
  local port=$2
  local success=0
  local fail=1

  which nc >/dev/null || {
    kube::log::usage "netcat isn't installed, can't verify if ${host}:${port} is free, skipping the check..."
    return ${success}
  }

  if [ ! $(nc -vz "${host}" "${port}") ]; then
    kube::log::status "${host}:${port} is free, proceeding..."
    return ${success}
  else
    kube::log::status "${host}:${port} is already used"
    return ${fail}
  fi
}

# Example:  kube::util::trap_add 'echo "in trap DEBUG"' DEBUG
# See: http://stackoverflow.com/questions/3338030/multiple-bash-traps-for-the-same-signal
kube::util::trap_add() {
  local trap_add_cmd
  trap_add_cmd=$1
  shift

  for trap_add_name in "$@"; do
    local existing_cmd
    local new_cmd

    # Grab the currently defined trap commands for this trap
    existing_cmd=`trap -p "${trap_add_name}" |  awk -F"'" '{print $2}'`

    if [[ -z "${existing_cmd}" ]]; then
      new_cmd="${trap_add_cmd}"
    else
      new_cmd="${existing_cmd};${trap_add_cmd}"
    fi

    # Assign the test
    trap "${new_cmd}" "${trap_add_name}"
  done
}

# Opposite of kube::util::ensure-temp-dir()
kube::util::cleanup-temp-dir() {
  rm -rf "${KUBE_TEMP}"
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
kube::util::ensure-temp-dir() {
  if [[ -z ${KUBE_TEMP-} ]]; then
    KUBE_TEMP=$(mktemp -d 2>/dev/null || mktemp -d -t kubernetes.XXXXXX)
    kube::util::trap_add kube::util::cleanup-temp-dir EXIT
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
    aarch64*)
      host_arch=arm64
      ;;
    arm64*)
      host_arch=arm64
      ;;
    arm*)
      host_arch=arm
      ;;
    i?86*)
      host_arch=x86
      ;;
    s390x*)
      host_arch=s390x
      ;;
    ppc64le*)
      host_arch=ppc64le
      ;;
    *)
      kube::log::error "Unsupported host arch. Must be x86_64, 386, arm, arm64, s390x or ppc64le."
      exit 1
      ;;
  esac
  echo "${host_os}/${host_arch}"
}

kube::util::find-binary() {
  local lookfor="${1}"
  local host_platform="$(kube::util::host_platform)"
  local locations=(
    "${KUBE_ROOT}/_output/bin/${lookfor}"
    "${KUBE_ROOT}/_output/dockerized/bin/${host_platform}/${lookfor}"
    "${KUBE_ROOT}/_output/local/bin/${host_platform}/${lookfor}"
    "${KUBE_ROOT}/platforms/${host_platform}/${lookfor}"
  )
  local bin=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )
  echo -n "${bin}"
}

# Run all known doc generators (today gendocs and genman for kubectl)
# $1 is the directory to put those generated documents
kube::util::gen-docs() {
  local dest="$1"

  # Find binary
  gendocs=$(kube::util::find-binary "gendocs")
  genkubedocs=$(kube::util::find-binary "genkubedocs")
  genman=$(kube::util::find-binary "genman")
  genyaml=$(kube::util::find-binary "genyaml")
  genfeddocs=$(kube::util::find-binary "genfeddocs")

  mkdir -p "${dest}/docs/user-guide/kubectl/"
  "${gendocs}" "${dest}/docs/user-guide/kubectl/"
  mkdir -p "${dest}/docs/admin/"
  "${genkubedocs}" "${dest}/docs/admin/" "kube-apiserver"
  "${genkubedocs}" "${dest}/docs/admin/" "kube-controller-manager"
  "${genkubedocs}" "${dest}/docs/admin/" "kube-proxy"
  "${genkubedocs}" "${dest}/docs/admin/" "kube-scheduler"
  "${genkubedocs}" "${dest}/docs/admin/" "kubelet"

  # We don't really need federation-apiserver and federation-controller-manager
  # binaries to generate the docs. We just pass their names to decide which docs
  # to generate. The actual binary for running federation is hyperkube.
  "${genfeddocs}" "${dest}/docs/admin/" "federation-apiserver"
  "${genfeddocs}" "${dest}/docs/admin/" "federation-controller-manager"

  mkdir -p "${dest}/docs/man/man1/"
  "${genman}" "${dest}/docs/man/man1/"
  mkdir -p "${dest}/docs/yaml/kubectl/"
  "${genyaml}" "${dest}/docs/yaml/kubectl/"

  # create the list of generated files
  pushd "${dest}" > /dev/null
  touch .generated_docs
  find . -type f | cut -sd / -f 2- | LC_ALL=C sort > .generated_docs
  popd > /dev/null
}

# Puts a placeholder for every generated doc. This makes the link checker work.
kube::util::set-placeholder-gen-docs() {
  local list_file="${KUBE_ROOT}/.generated_docs"
  if [ -e ${list_file} ]; then
    # remove all of the old docs; we don't want to check them in.
    while read file; do
      if [[ "${list_file}" != "${KUBE_ROOT}/${file}" ]]; then
        cp "${KUBE_ROOT}/hack/autogenerated_placeholder.txt" "${KUBE_ROOT}/${file}"
      fi
    done <"${list_file}"
    # The .generated_docs file lists itself, so we don't need to explicitly
    # delete it.
  fi
}

# Removes previously generated docs-- we don't want to check them in. $KUBE_ROOT
# must be set.
kube::util::remove-gen-docs() {
  if [ -e "${KUBE_ROOT}/.generated_docs" ]; then
    # remove all of the old docs; we don't want to check them in.
    while read file; do
      rm "${KUBE_ROOT}/${file}" 2>/dev/null || true
    done <"${KUBE_ROOT}/.generated_docs"
    # The .generated_docs file lists itself, so we don't need to explicitly
    # delete it.
  fi
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
  # We don't touch files in special dirs, and the kubectl docs are
  # autogenerated by gendocs.
  # Don't descend into .directories
  mdfiles=($( find "${dir}" -name "*.md" -type f \
              -not -path '*/\.*' \
              -not -path "${path}/vendor/*" \
              -not -path "${path}/staging/*" \
              -not -path "${path}/third_party/*" \
              -not -path "${path}/_gopath/*" \
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

# Takes a group/version and returns the path to its location on disk, sans
# "pkg". E.g.:
# * default behavior: extensions/v1beta1 -> apis/extensions/v1beta1
# * default behavior for only a group: experimental -> apis/experimental
# * Special handling for empty group: v1 -> api/v1, unversioned -> api/unversioned
# * Special handling for groups suffixed with ".k8s.io": foo.k8s.io/v1 -> apis/foo/v1
# * Very special handling for when both group and version are "": / -> api
kube::util::group-version-to-pkg-path() {
  local group_version="$1"
  # Special cases first.
  # TODO(lavalamp): Simplify this by moving pkg/api/v1 and splitting pkg/api,
  # moving the results to pkg/apis/api.
  case "${group_version}" in
    # both group and version are "", this occurs when we generate deep copies for internal objects of the legacy v1 API.
    __internal)
      echo "api"
      ;;
    v1)
      echo "api/v1"
      ;;
    unversioned)
      echo "api/unversioned"
      ;;
    *.k8s.io)
      echo "apis/${group_version%.k8s.io}"
      ;;
    *.k8s.io/*)
      echo "apis/${group_version/.k8s.io/}"
      ;;
    *)
      echo "apis/${group_version%__internal}"
      ;;
  esac
}

# Takes a group/version and returns the swagger-spec file name.
# default behavior: extensions/v1beta1 -> extensions_v1beta1
# special case for v1: v1 -> v1
kube::util::gv-to-swagger-name() {
  local group_version="$1"
  case "${group_version}" in
    v1)
      echo "v1"
      ;;
    *)
      echo "${group_version%/*}_${group_version#*/}"
      ;;
  esac
}


# Fetches swagger spec from apiserver.
# Assumed vars:
# SWAGGER_API_PATH: Base path for swaggerapi on apiserver. Ex:
# http://localhost:8080/swaggerapi.
# SWAGGER_ROOT_DIR: Root dir where we want to to save the fetched spec.
# VERSIONS: Array of group versions to include in swagger spec.
kube::util::fetch-swagger-spec() {
  for ver in ${VERSIONS}; do
    # fetch the swagger spec for each group version.
    if [[ ${ver} == "v1" ]]; then
      SUBPATH="api"
    else
      SUBPATH="apis"
    fi
    SUBPATH="${SUBPATH}/${ver}"
    SWAGGER_JSON_NAME="$(kube::util::gv-to-swagger-name ${ver}).json"
    curl -w "\n" -fs "${SWAGGER_API_PATH}${SUBPATH}" > "${SWAGGER_ROOT_DIR}/${SWAGGER_JSON_NAME}"

    # fetch the swagger spec for the discovery mechanism at group level.
    if [[ ${ver} == "v1" ]]; then
      continue
    fi
    SUBPATH="apis/"${ver%/*}
    SWAGGER_JSON_NAME="${ver%/*}.json"
    curl -w "\n" -fs "${SWAGGER_API_PATH}${SUBPATH}" > "${SWAGGER_ROOT_DIR}/${SWAGGER_JSON_NAME}"
  done

  # fetch swagger specs for other discovery mechanism.
  curl -w "\n" -fs "${SWAGGER_API_PATH}" > "${SWAGGER_ROOT_DIR}/resourceListing.json"
  curl -w "\n" -fs "${SWAGGER_API_PATH}version" > "${SWAGGER_ROOT_DIR}/version.json"
  curl -w "\n" -fs "${SWAGGER_API_PATH}api" > "${SWAGGER_ROOT_DIR}/api.json"
  curl -w "\n" -fs "${SWAGGER_API_PATH}apis" > "${SWAGGER_ROOT_DIR}/apis.json"
  curl -w "\n" -fs "${SWAGGER_API_PATH}logs" > "${SWAGGER_ROOT_DIR}/logs.json"
}


# Returns the name of the upstream remote repository name for the local git
# repo, e.g. "upstream" or "origin".
kube::util::git_upstream_remote_name() {
  git remote -v | grep fetch |\
    grep -E 'github.com[/:]kubernetes/kubernetes|k8s.io/kubernetes' |\
    head -n 1 | awk '{print $1}'
}

# Checks whether there are any files matching pattern $2 changed between the
# current branch and upstream branch named by $1.
# Returns 1 (false) if there are no changes, 0 (true) if there are changes
# detected.
kube::util::has_changes_against_upstream_branch() {
  local -r git_branch=$1
  local -r pattern=$2
  local full_branch

  full_branch="$(kube::util::git_upstream_remote_name)/${git_branch}"
  echo "Checking for '${pattern}' changes against '${full_branch}'"
  # make sure the branch is valid, otherwise the check will pass erroneously.
  if ! git describe "${full_branch}" >/dev/null; then
    # abort!
    exit 1
  fi
  # notice this uses ... to find the first shared ancestor
  if git diff --name-only "${full_branch}...HEAD" | grep "${pattern}" > /dev/null; then
    return 0
  fi
  # also check for pending changes
  if git status --porcelain | grep "${pattern}" > /dev/null; then
    echo "Detected '${pattern}' uncommitted changes."
    return 0
  fi
  echo "No '${pattern}' changes detected."
  return 1
}

# ex: ts=2 sw=2 et filetype=sh
