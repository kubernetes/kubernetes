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
    if out=$(curl --max-time 1 -gkfs $url 2>/dev/null); then
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

kube::util::find-binary-for-platform() {
  local -r lookfor="$1"
  local -r platform="$2"
  local locations=(
    "${KUBE_ROOT}/_output/bin/${lookfor}"
    "${KUBE_ROOT}/_output/dockerized/bin/${platform}/${lookfor}"
    "${KUBE_ROOT}/_output/local/bin/${platform}/${lookfor}"
    "${KUBE_ROOT}/platforms/${platform}/${lookfor}"
  )
  # Also search for binary in bazel build tree.
  # In some cases we have to name the binary $BINARY_bin, since there was a
  # directory named $BINARY next to it.
  locations+=($(find "${KUBE_ROOT}/bazel-bin/" -type f -executable \
    \( -name "${lookfor}" -o -name "${lookfor}_bin" \) 2>/dev/null || true) )

  # List most recently-updated location.
  local -r bin=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )
  echo -n "${bin}"
}

kube::util::find-binary() {
  kube::util::find-binary-for-platform "$1" "$(kube::util::host_platform)"
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
  "${genkubedocs}" "${dest}/docs/admin/" "cloud-controller-manager"
  "${genkubedocs}" "${dest}/docs/admin/" "kube-proxy"
  "${genkubedocs}" "${dest}/docs/admin/" "kube-scheduler"
  "${genkubedocs}" "${dest}/docs/admin/" "kubelet"

  # We don't really need federation-apiserver and federation-controller-manager
  # binaries to generate the docs. We just pass their names to decide which docs
  # to generate. The actual binary for running federation is hyperkube.
  "${genfeddocs}" "${dest}/docs/admin/" "federation-apiserver"
  "${genfeddocs}" "${dest}/docs/admin/" "federation-controller-manager"
  "${genfeddocs}" "${dest}/docs/admin/" "kubefed"

  mkdir -p "${dest}/docs/man/man1/"
  "${genman}" "${dest}/docs/man/man1/" "kube-apiserver"
  "${genman}" "${dest}/docs/man/man1/" "kube-controller-manager"
  "${genman}" "${dest}/docs/man/man1/" "cloud-controller-manager"
  "${genman}" "${dest}/docs/man/man1/" "kube-proxy"
  "${genman}" "${dest}/docs/man/man1/" "kube-scheduler"
  "${genman}" "${dest}/docs/man/man1/" "kubelet"
  "${genman}" "${dest}/docs/man/man1/" "kubectl"

  mkdir -p "${dest}/docs/yaml/kubectl/"
  "${genyaml}" "${dest}/docs/yaml/kubectl/"

  # create the list of generated files
  pushd "${dest}" > /dev/null
  touch docs/.generated_docs
  find . -type f | cut -sd / -f 2- | LC_ALL=C sort > docs/.generated_docs
  popd > /dev/null
}

# Puts a placeholder for every generated doc. This makes the link checker work.
kube::util::set-placeholder-gen-docs() {
  local list_file="${KUBE_ROOT}/docs/.generated_docs"
  if [ -e ${list_file} ]; then
    # remove all of the old docs; we don't want to check them in.
    while read file; do
      if [[ "${list_file}" != "${KUBE_ROOT}/${file}" ]]; then
        cp "${KUBE_ROOT}/hack/autogenerated_placeholder.txt" "${KUBE_ROOT}/${file}"
      fi
    done <"${list_file}"
    # The docs/.generated_docs file lists itself, so we don't need to explicitly
    # delete it.
  fi
}

# Removes previously generated docs-- we don't want to check them in. $KUBE_ROOT
# must be set.
kube::util::remove-gen-docs() {
  if [ -e "${KUBE_ROOT}/docs/.generated_docs" ]; then
    # remove all of the old docs; we don't want to check them in.
    while read file; do
      rm "${KUBE_ROOT}/${file}" 2>/dev/null || true
    done <"${KUBE_ROOT}/docs/.generated_docs"
    # The docs/.generated_docs file lists itself, so we don't need to explicitly
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
  staging_apis=(
  $(
    pushd ${KUBE_ROOT}/staging/src/k8s.io/api > /dev/null
      find . -name types.go | xargs -n1 dirname | sed "s|\./||g" | sort
    popd > /dev/null
  )
  )

  local group_version="$1"

  if [[ " ${staging_apis[@]} " =~ " ${group_version/.*k8s.io/} " ]]; then
    echo "vendor/k8s.io/api/${group_version/.*k8s.io/}"
    return
  fi

  # "v1" is the API GroupVersion 
  if [[ "${group_version}" == "v1" ]]; then
    echo "vendor/k8s.io/api/core/v1"
    return
  fi

  # Special cases first.
  # TODO(lavalamp): Simplify this by moving pkg/api/v1 and splitting pkg/api,
  # moving the results to pkg/apis/api.
  case "${group_version}" in
    # both group and version are "", this occurs when we generate deep copies for internal objects of the legacy v1 API.
    __internal)
      echo "pkg/api"
      ;;
    federation/v1beta1)
      echo "federation/apis/federation/v1beta1"
      ;;
    meta/v1)
      echo "vendor/k8s.io/apimachinery/pkg/apis/meta/v1"
      ;;
    meta/v1)
      echo "../vendor/k8s.io/apimachinery/pkg/apis/meta/v1"
      ;;
    meta/v1alpha1)
      echo "vendor/k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
      ;;
    meta/v1alpha1)
      echo "../vendor/k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
      ;;
    unversioned)
      echo "pkg/api/unversioned"
      ;;
    *.k8s.io)
      echo "pkg/apis/${group_version%.*k8s.io}"
      ;;
    *.k8s.io/*)
      echo "pkg/apis/${group_version/.*k8s.io/}"
      ;;
    *)
      echo "pkg/apis/${group_version%__internal}"
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
    if [[ " ${KUBE_NONSERVER_GROUP_VERSIONS} " == *" ${ver} "* ]]; then
      continue
    fi
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

# Checks whether godep restore was run in the current GOPATH, i.e. that all referenced repos exist
# and are checked out to the referenced rev.
kube::util::godep_restored() {
  local -r godeps_json=${1:-Godeps/Godeps.json}
  local -r gopath=${2:-${GOPATH%:*}}
  if ! which jq &>/dev/null; then
    echo "jq not found. Please install." 1>&2
    return 1
  fi
  local root
  local old_rev=""
  while read path rev; do
    rev=$(echo "${rev}" | sed "s/['\"]//g") # remove quotes which are around revs sometimes

    if [[ "${rev}" == "${old_rev}" ]] && [[ "${path}" == "${root}"* ]]; then
      # avoid checking the same git/hg root again
      continue
    fi

    root="${path}"
    while [ "${root}" != "." -a ! -d "${gopath}/src/${root}/.git" -a ! -d "${gopath}/src/${root}/.hg" ]; do
      root=$(dirname "${root}")
    done
    if [ "${root}" == "." ]; then
      echo "No checkout of ${path} found in GOPATH \"${gopath}\"." 1>&2
      return 1
    fi
    local head
    if [ -d "${gopath}/src/${root}/.git" ]; then
      head="$(cd "${gopath}/src/${root}" && git rev-parse HEAD)"
    else
      head="$(cd "${gopath}/src/${root}" && hg parent --template '{node}')"
    fi
    if [ "${head}" != "${rev}" ]; then
      echo "Unexpected HEAD '${head}' at ${gopath}/src/${root}, expected '${rev}'." 1>&2
      return 1
    fi
    old_rev="${rev}"
  done < <(jq '.Deps|.[]|.ImportPath + " " + .Rev' -r < "${godeps_json}")
  return 0
}

# Exits script if working directory is dirty. If it's run interactively in the terminal
# the user can commit changes in a second terminal. This script will wait.
kube::util::ensure_clean_working_dir() {
  while ! git diff HEAD --exit-code &>/dev/null; do
    echo -e "\nUnexpected dirty working directory:\n"
    if tty -s; then
        git status -s
    else
        git diff -a # be more verbose in log files without tty
        exit 1
    fi | sed 's/^/  /'
    echo -e "\nCommit your changes in another terminal and then continue here by pressing enter."
    read
  done 1>&2
}

# Ensure that the given godep version is installed and in the path
kube::util::ensure_godep_version() {
  GODEP_VERSION=${1:-"v79"}
  if [[ "$(godep version 2>/dev/null)" == *"godep ${GODEP_VERSION}"* ]]; then
    return
  fi

  kube::util::ensure-temp-dir
  mkdir -p "${KUBE_TEMP}/go/src"

  GOPATH="${KUBE_TEMP}/go" go get -d -u github.com/tools/godep 2>/dev/null
  pushd "${KUBE_TEMP}/go/src/github.com/tools/godep" >/dev/null
    git checkout -q "${GODEP_VERSION}"
    GOPATH="${KUBE_TEMP}/go" go install .
  popd >/dev/null

  PATH="${KUBE_TEMP}/go/bin:${PATH}"
  hash -r # force bash to clear PATH cache
  godep version
}

# Ensure that none of the staging repos is checked out in the GOPATH because this
# easily confused godep.
kube::util::ensure_no_staging_repos_in_gopath() {
  kube::util::ensure_single_dir_gopath
  local error=0
  for repo in $(ls ${KUBE_ROOT}/staging/src/k8s.io); do
    if [ -e "${GOPATH}/src/k8s.io/${repo}" ]; then
      echo "k8s.io/${repo} exists in GOPATH. Remove before running godep-save.sh." 1>&2
      error=1
    fi
  done
  if [ "${error}" = "1" ]; then
    exit 1
  fi
}

# Installs the specified go package at a particular commit.
kube::util::go_install_from_commit() {
  local -r pkg=$1
  local -r commit=$2

  kube::util::ensure-temp-dir
  mkdir -p "${KUBE_TEMP}/go/src"
  GOPATH="${KUBE_TEMP}/go" go get -d -u "${pkg}" 2>/dev/null
  (
    cd "${KUBE_TEMP}/go/src/${pkg}"
    git checkout -q "${commit}"
    GOPATH="${KUBE_TEMP}/go" go install "${pkg}"
  )
  PATH="${KUBE_TEMP}/go/bin:${PATH}"
  hash -r # force bash to clear PATH cache
}

# Checks that the GOPATH is simple, i.e. consists only of one directory, not multiple.
kube::util::ensure_single_dir_gopath() {
  if [[ "${GOPATH}" == *:* ]]; then
    echo "GOPATH must consist of a single directory." 1>&2
    exit 1
  fi
}

# Checks whether there are any files matching pattern $2 changed between the
# current branch and upstream branch named by $1.
# Returns 1 (false) if there are no changes, 0 (true) if there are changes
# detected.
kube::util::has_changes_against_upstream_branch() {
  local -r git_branch=$1
  local -r pattern=$2
  local -r not_pattern=${3:-totallyimpossiblepattern}
  local full_branch

  full_branch="$(kube::util::git_upstream_remote_name)/${git_branch}"
  echo "Checking for '${pattern}' changes against '${full_branch}'"
  # make sure the branch is valid, otherwise the check will pass erroneously.
  if ! git describe "${full_branch}" >/dev/null; then
    # abort!
    exit 1
  fi
  # notice this uses ... to find the first shared ancestor
  if git diff --name-only "${full_branch}...HEAD" | grep -v -E "${not_pattern}" | grep "${pattern}" > /dev/null; then
    return 0
  fi
  # also check for pending changes
  if git status --porcelain | grep -v -E "${not_pattern}" | grep "${pattern}" > /dev/null; then
    echo "Detected '${pattern}' uncommitted changes."
    return 0
  fi
  echo "No '${pattern}' changes detected."
  return 1
}

kube::util::download_file() {
  local -r url=$1
  local -r destination_file=$2

  rm  ${destination_file} 2&> /dev/null || true

  for i in $(seq 5)
  do
    if ! curl -fsSL --retry 3 --keepalive-time 2 ${url} -o ${destination_file}; then
      echo "Downloading ${url} failed. $((5-i)) retries left."
      sleep 1
    else
      echo "Downloading ${url} succeed"
      return 0
    fi
  done
  return 1
}

# Test whether openssl is installed.
# Sets:
#  OPENSSL_BIN: The path to the openssl binary to use
function kube::util::test_openssl_installed {
    openssl version >& /dev/null
    if [ "$?" != "0" ]; then
      echo "Failed to run openssl. Please ensure openssl is installed"
      exit 1
    fi
    OPENSSL_BIN=$(command -v openssl)
}

# creates a client CA, args are sudo, dest-dir, ca-id, purpose
# purpose is dropped in after "key encipherment", you usually want
# '"client auth"'
# '"server auth"'
# '"client auth","server auth"'
function kube::util::create_signing_certkey {
    local sudo=$1
    local dest_dir=$2
    local id=$3
    local purpose=$4
    # Create client ca
    ${sudo} /bin/bash -e <<EOF
    rm -f "${dest_dir}/${id}-ca.crt" "${dest_dir}/${id}-ca.key"
    ${OPENSSL_BIN} req -x509 -sha256 -new -nodes -days 365 -newkey rsa:2048 -keyout "${dest_dir}/${id}-ca.key" -out "${dest_dir}/${id}-ca.crt" -subj "/C=xx/ST=x/L=x/O=x/OU=x/CN=ca/emailAddress=x/"
    echo '{"signing":{"default":{"expiry":"43800h","usages":["signing","key encipherment",${purpose}]}}}' > "${dest_dir}/${id}-ca-config.json"
EOF
}

# signs a client certificate: args are sudo, dest-dir, CA, filename (roughly), username, groups...
function kube::util::create_client_certkey {
    local sudo=$1
    local dest_dir=$2
    local ca=$3
    local id=$4
    local cn=${5:-$4}
    local groups=""
    local SEP=""
    shift 5
    while [ -n "${1:-}" ]; do
        groups+="${SEP}{\"O\":\"$1\"}"
        SEP=","
        shift 1
    done
    ${sudo} /bin/bash -e <<EOF
    cd ${dest_dir}
    echo '{"CN":"${cn}","names":[${groups}],"hosts":[""],"key":{"algo":"rsa","size":2048}}' | ${CFSSL_BIN} gencert -ca=${ca}.crt -ca-key=${ca}.key -config=${ca}-config.json - | ${CFSSLJSON_BIN} -bare client-${id}
    mv "client-${id}-key.pem" "client-${id}.key"
    mv "client-${id}.pem" "client-${id}.crt"
    rm -f "client-${id}.csr"
EOF
}

# signs a serving certificate: args are sudo, dest-dir, ca, filename (roughly), subject, hosts...
function kube::util::create_serving_certkey {
    local sudo=$1
    local dest_dir=$2
    local ca=$3
    local id=$4
    local cn=${5:-$4}
    local hosts=""
    local SEP=""
    shift 5
    while [ -n "${1:-}" ]; do
        hosts+="${SEP}\"$1\""
        SEP=","
        shift 1
    done
    ${sudo} /bin/bash -e <<EOF
    cd ${dest_dir}
    echo '{"CN":"${cn}","hosts":[${hosts}],"key":{"algo":"rsa","size":2048}}' | ${CFSSL_BIN} gencert -ca=${ca}.crt -ca-key=${ca}.key -config=${ca}-config.json - | ${CFSSLJSON_BIN} -bare serving-${id}
    mv "serving-${id}-key.pem" "serving-${id}.key"
    mv "serving-${id}.pem" "serving-${id}.crt"
    rm -f "serving-${id}.csr"
EOF
}

# creates a self-contained kubeconfig: args are sudo, dest-dir, ca file, host, port, client id, token(optional)
function kube::util::write_client_kubeconfig {
    local sudo=$1
    local dest_dir=$2
    local ca_file=$3
    local api_host=$4
    local api_port=$5
    local client_id=$6
    local token=${7:-}
    cat <<EOF | ${sudo} tee "${dest_dir}"/${client_id}.kubeconfig > /dev/null
apiVersion: v1
kind: Config
clusters:
  - cluster:
      certificate-authority: ${ca_file}
      server: https://${api_host}:${api_port}/
    name: local-up-cluster
users:
  - user:
      token: ${token}
      client-certificate: ${dest_dir}/client-${client_id}.crt
      client-key: ${dest_dir}/client-${client_id}.key
    name: local-up-cluster
contexts:
  - context:
      cluster: local-up-cluster
      user: local-up-cluster
    name: local-up-cluster
current-context: local-up-cluster
EOF

    # flatten the kubeconfig files to make them self contained
    username=$(whoami)
    ${sudo} /bin/bash -e <<EOF
    $(kube::util::find-binary kubectl) --kubeconfig="${dest_dir}/${client_id}.kubeconfig" config view --minify --flatten > "/tmp/${client_id}.kubeconfig"
    mv -f "/tmp/${client_id}.kubeconfig" "${dest_dir}/${client_id}.kubeconfig"
    chown ${username} "${dest_dir}/${client_id}.kubeconfig"
EOF
}

# Determines if docker can be run, failures may simply require that the user be added to the docker group.
function kube::util::ensure_docker_daemon_connectivity {
  DOCKER=(docker ${DOCKER_OPTS})
  if ! "${DOCKER[@]}" info > /dev/null 2>&1 ; then
    cat <<'EOF' >&2
Can't connect to 'docker' daemon.  please fix and retry.

Possible causes:
  - Docker Daemon not started
    - Linux: confirm via your init system
    - macOS w/ docker-machine: run `docker-machine ls` and `docker-machine start <name>`
    - macOS w/ Docker for Mac: Check the menu bar and start the Docker application
  - DOCKER_HOST hasn't been set or is set incorrectly
    - Linux: domain socket is used, DOCKER_* should be unset. In Bash run `unset ${!DOCKER_*}`
    - macOS w/ docker-machine: run `eval "$(docker-machine env <name>)"`
    - macOS w/ Docker for Mac: domain socket is used, DOCKER_* should be unset. In Bash run `unset ${!DOCKER_*}`
  - Other things to check:
    - Linux: User isn't in 'docker' group.  Add and relogin.
      - Something like 'sudo usermod -a -G docker ${USER}'
      - RHEL7 bug and workaround: https://bugzilla.redhat.com/show_bug.cgi?id=1119282#c8
EOF
    return 1
  fi
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

# kube::util::join <delim> <list...>
# Concatenates the list elements with the delimiter passed as first parameter
#
# Ex: kube::util::join , a b c
#  -> a,b,c
function kube::util::join {
  local IFS="$1"
  shift
  echo "$*"
}

# Downloads cfssl/cfssljson into $1 directory if they do not already exist in PATH
#
# Assumed vars:
#   $1 (cfssl directory) (optional)
#
# Sets:
#  CFSSL_BIN: The path of the installed cfssl binary
#  CFSSLJSON_BIN: The path of the installed cfssljson binary
#
function kube::util::ensure-cfssl {
  if command -v cfssl &>/dev/null && command -v cfssljson &>/dev/null; then
    CFSSL_BIN=$(command -v cfssl)
    CFSSLJSON_BIN=$(command -v cfssljson)
    return 0
  fi

  # Create a temp dir for cfssl if no directory was given
  local cfssldir=${1:-}
  if [[ -z "${cfssldir}" ]]; then
    kube::util::ensure-temp-dir
    cfssldir="${KUBE_TEMP}/cfssl"
  fi

  mkdir -p "${cfssldir}"
  pushd "${cfssldir}" > /dev/null

    echo "Unable to successfully run 'cfssl' from $PATH; downloading instead..."
    kernel=$(uname -s)
    case "${kernel}" in
      Linux)
        curl -s -L -o cfssl https://pkg.cfssl.org/R1.2/cfssl_linux-amd64
        curl -s -L -o cfssljson https://pkg.cfssl.org/R1.2/cfssljson_linux-amd64
        ;;
      Darwin)
        curl -s -L -o cfssl https://pkg.cfssl.org/R1.2/cfssl_darwin-amd64
        curl -s -L -o cfssljson https://pkg.cfssl.org/R1.2/cfssljson_darwin-amd64
        ;;
      *)
        echo "Unknown, unsupported platform: ${kernel}." >&2
        echo "Supported platforms: Linux, Darwin." >&2
        exit 2
    esac

    chmod +x cfssl || true
    chmod +x cfssljson || true

    CFSSL_BIN="${cfssldir}/cfssl"
    CFSSLJSON_BIN="${cfssldir}/cfssljson"
    if [[ ! -x ${CFSSL_BIN} || ! -x ${CFSSLJSON_BIN} ]]; then
      echo "Failed to download 'cfssl'. Please install cfssl and cfssljson and verify they are in \$PATH."
      echo "Hint: export PATH=\$PATH:\$GOPATH/bin; go get -u github.com/cloudflare/cfssl/cmd/..."
      exit 1
    fi
  popd > /dev/null
}

# kube::util::ensure_dockerized
# Confirms that the script is being run inside a kube-build image
#
function kube::util::ensure_dockerized {
  if [[ -f /kube-build-image ]]; then
    return 0
  else
    echo "ERROR: This script is designed to be run inside a kube-build container"
    exit 1
  fi
}

# Some useful colors.
if [[ -z "${color_start-}" ]]; then
  declare -r color_start="\033["
  declare -r color_red="${color_start}0;31m"
  declare -r color_yellow="${color_start}0;33m"
  declare -r color_green="${color_start}0;32m"
  declare -r color_norm="${color_start}0m"
fi

# ex: ts=2 sw=2 et filetype=sh
