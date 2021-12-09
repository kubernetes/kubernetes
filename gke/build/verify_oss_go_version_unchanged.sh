#!/usr/bin/env bash

# This script verifies that the OSS k/k build image (and corresponding Go version)
# hasn't changed since the last time we updated our corresponding internal build config.
#
# The high-level workflow is:
# 1. OSS changes go version (currently by modifying build/build-image/cross/VERSION)
# 2. GKE picks up the OSS change automatically
# 3. A periodic job running this verify script starts failing and opens a bug or sends
#    an email to the maintainers to update our internal go-boringcrypto compiler image
#    (potentially prodding the Go team to create a new release if necessary).

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
CONFIG_FILE="${SCRIPT_DIR}/config/common.yaml"
CONFIG_FILE_COPY="${SCRIPT_DIR}/config/common-copy.yaml"  # Used to take a diff when drift occurs
KUBE_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
OSS_KUBECROSS_VERSION_FILE="${KUBE_ROOT}/build/build-image/cross/VERSION"

# Post 1.16.10, boringcrypto is hosted at this project (formerly DockerHub)
BORINGCRYPTO_IMAGE="us-docker.pkg.dev/google.com/api-project-999119582588/go-boringcrypto/golang"
PATCH_FILE="${KUBE_ROOT}/diff.patch"

# shellcheck source=./lib_assert.sh
source "${SCRIPT_DIR}/lib_assert.sh"

# shellcheck source=./lib_log.sh
source "${SCRIPT_DIR}/lib_log.sh"

# shellcheck source=./lib_yaml.sh
source "${SCRIPT_DIR}/lib_yaml.sh"

# get_val mimics the function of the same name in lib_gke.sh, without requiring
# a product build and specifying the config file directly.
function get_val {
  _yq read --stripComments "${CONFIG_FILE}" "$@"
}

function write_val {
  log.debug "Writing: $@"
  _yq write --inplace "${CONFIG_FILE_COPY}" "$@"
}

function generate_patch {
  local new_golang_image="${1}"
  local new_boringcrypto_image="${2}"
  local new_kubecross_version="${3}"

  log.info "Generating patch..."
  cp  "${CONFIG_FILE}" "${CONFIG_FILE_COPY}"

  write_val "build-env.compiler-image.deps.golang-image"  "${new_golang_image}"
  write_val "build-env.compiler-image.deps.golang-boringcrypto-image" "${new_boring_image}"
  write_val "build-env.compiler-image.fips.go-boringcrypto.last-updated-for-oss-kubecross-version" "${new_kubecross_version}"

  log.debug "Generating diff..."
  # _yq has opinions on indentation and newlines in some list items.
  # -u for git format, -bwB to aggressively ignore whitespace.
  # `|| true` to skip errexit on expected non-zero diff.
  # Use local path for git-friendly headers
  local relative_path_config=$(realpath --relative-to="${KUBE_ROOT}" "${CONFIG_FILE}")
  local relative_path_copy=$(realpath --relative-to="${KUBE_ROOT}" "${CONFIG_FILE_COPY}")
  pushd ${KUBE_ROOT} > /dev/null
    ( diff -ubwB "${relative_path_config}" "${relative_path_copy}" > "${PATCH_FILE}" ) || true
    # Adjust the path so it looks like a git patch.
    sed -i "s/common-copy.yaml/common.yaml/" "${PATCH_FILE}"
    sed -i "s/--- gke/--- a\/gke/" "${PATCH_FILE}"
    sed -i "s/\+\+\+ gke/\+\+\+ b\/gke/" "${PATCH_FILE}"
  popd > /dev/null

  # cleanup
  rm "${CONFIG_FILE_COPY}"
}

function kubecross_version_to_go_semver()
{
  # Previous cross version scheme: v<Version>-<Build>
  #   Example: v1.16.6-1
  # Current cross naming scheme: <K8s version>-go<Go version>-<debian name>.<build>
  #   Example: v1.23.0-go1.17.2-bullseye.0
  local version="$1"
  local left_strip_v=${version#v}
  local oss_trim_left=${left_strip_v#*-go}  # drop "v1.23.0-go"
  local oss_trimmed=${oss_trim_left%-*} # drop "-bullseye.0", left with just "1.17.2"
  echo ${oss_trimmed}
}

function get_boring_tags()
{
  gcloud container images list-tags "${BORINGCRYPTO_IMAGE}" --format="value(tags)"
}

log.info "Reading OSS kube-cross image version..."
assert_path_exists "${OSS_KUBECROSS_VERSION_FILE}"
oss_kubecross_version="$(cat "${OSS_KUBECROSS_VERSION_FILE}")"
oss_kubecross_go_version=$(kubecross_version_to_go_semver "${oss_kubecross_version}")
log.info "OSS kube-cross complete version: \"${oss_kubecross_version}\""
log.info "OSS kube-cross Go version: \"${oss_kubecross_go_version}\""

__old_golang_image=$(get_val "build-env.compiler-image.deps.golang-image")
__old_golang_boringcrypto_image=$(get_val "build-env.compiler-image.deps.golang-boringcrypto-image")
__last_updated_kubecross_version=$(get_val "build-env.compiler-image.fips.go-boringcrypto.last-updated-for-oss-kubecross-version")

log.debugvar __old_golang_image
log.debugvar __old_golang_boringcrypto_image
log.debugvar __last_updated_kubecross_version
log.debugvar oss_kubecross_version
log.debugvar oss_kubecross_go_version

if [[ "${oss_kubecross_go_version}" == "${__last_updated_kubecross_version}" ]]; then
  log.info "OK!"
  exit 0
else
  log.warn "Golang version was last updated for ${__last_updated_kubecross_version}, OSS is currently at Go version ${oss_kubecross_go_version} (full version: ${oss_kubecross_version})."
  log.info "See http://goto.google.com/gke-security-compliance#periodics-gob-kubernetes-verify-golang-version-failure-playbook for step-by-step resolution instructions."
  log.info "Some of this will now be attempted on your behalf..."

  # Check if desired release is available
  # boringcrypto releases follow the pattern "<go version>b<module revision>"
  # See if there is a boringcrypto release of the corresponding version using crane.  Use the b to anchor.
  new_boring_image_tag=$(get_boring_tags | grep "^${oss_kubecross_go_version}b" || true)
  if [ -n "${new_boring_image_tag:-}" ] ; then
    new_boring_image="${BORINGCRYPTO_IMAGE}:${new_boring_image_tag}"
    log.info "boringcrypto release available: ${new_boring_image}"
    generate_patch "golang:${oss_kubecross_go_version}" "${new_boring_image}" "${oss_kubecross_go_version}"
    log.fail "Run $(realpath --relative-to=${KUBE_ROOT} $0) locally to generate necessary patch.  Then call 'git apply ${PATCH_FILE}' to apply the necessary diff to bump to the updated go / boringcrypto releases."
  else
    log.fail "No boringcrypto release tag corresponding to Go version ${oss_kubecross_go_version} found at ${BORINGCRYPTO_IMAGE}.  Confirm that release should be pending and block this test's alert on that release ticket.  See the above playbook link for additional details."
  fi
fi
