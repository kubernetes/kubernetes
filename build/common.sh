#!/usr/bin/env bash

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

# shellcheck disable=SC2034 # Variables sourced in other scripts.

# Common utilities, variables and checks for all build scripts.
set -o errexit
set -o nounset
set -o pipefail

# Unset CDPATH, having it set messes up with script import paths
unset CDPATH

USER_ID=$(id -u)
GROUP_ID=$(id -g)

DOCKER_OPTS=${DOCKER_OPTS:-""}
IFS=" " read -r -a DOCKER <<< "docker ${DOCKER_OPTS}"
DOCKER_HOST=${DOCKER_HOST:-""}
GOPROXY=${GOPROXY:-""}

# This will canonicalize the path
KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)

source "${KUBE_ROOT}/hack/lib/init.sh"

# Constants
KUBE_BUILD_IMAGE_CROSS_TAG="${KUBE_CROSS_VERSION:-"$(cat "${KUBE_ROOT}/build/build-image/cross/VERSION")"}"
readonly KUBE_BUILD_IMAGE_CROSS_TAG

readonly KUBE_DOCKER_REGISTRY="${KUBE_DOCKER_REGISTRY:-registry.k8s.io}"
KUBE_BASE_IMAGE_REGISTRY="${KUBE_BASE_IMAGE_REGISTRY:-registry.k8s.io/build-image}"
readonly KUBE_BASE_IMAGE_REGISTRY

# Make it possible to override the `kube-cross` image, and tag independent of `KUBE_BASE_IMAGE_REGISTRY`
KUBE_CROSS_IMAGE="${KUBE_CROSS_IMAGE:-"${KUBE_BASE_IMAGE_REGISTRY}/kube-cross"}"
readonly KUBE_CROSS_IMAGE
KUBE_CROSS_VERSION="${KUBE_CROSS_VERSION:-"${KUBE_BUILD_IMAGE_CROSS_TAG}"}"
readonly KUBE_CROSS_VERSION
KUBE_CROSS_CONTAINER_ROOT="/go/src/k8s.io/kubernetes"
readonly KUBE_CROSS_CONTAINER_ROOT

# Here we map the output directories across both the local and remote _output
# directories:
#
# *_OUTPUT_ROOT    - the base of all output in that environment.
# *_OUTPUT_SUBPATH - location where golang stuff is built/cached.  Also
#                    persisted across docker runs with a volume mount.
# *_OUTPUT_BINPATH - location where final binaries are placed.  If the remote
#                    is really remote, this is the stuff that has to be copied
#                    back.
# OUT_DIR can come in from the Makefile, so honor it.
readonly LOCAL_OUTPUT_ROOT="${KUBE_ROOT}/${OUT_DIR:-_output}"
readonly LOCAL_OUTPUT_SUBPATH="${LOCAL_OUTPUT_ROOT}/dockerized"
readonly LOCAL_OUTPUT_BINPATH="${LOCAL_OUTPUT_SUBPATH}/bin"
readonly LOCAL_OUTPUT_GOPATH="${LOCAL_OUTPUT_SUBPATH}/go"
readonly LOCAL_OUTPUT_IMAGE_STAGING="${LOCAL_OUTPUT_ROOT}/images"

# This is a symlink to binaries for "this platform" (e.g. build tools).
readonly THIS_PLATFORM_BIN="${LOCAL_OUTPUT_ROOT}/bin"

readonly KUBE_GO_PACKAGE=k8s.io/kubernetes
readonly REMOTE_ROOT="/go/src/${KUBE_GO_PACKAGE}"
readonly REMOTE_OUTPUT_ROOT="${REMOTE_ROOT}/_output"
readonly REMOTE_OUTPUT_SUBPATH="${REMOTE_OUTPUT_ROOT}/dockerized"
readonly REMOTE_OUTPUT_BINPATH="${REMOTE_OUTPUT_SUBPATH}/bin"
readonly REMOTE_OUTPUT_GOPATH="${REMOTE_OUTPUT_SUBPATH}/go"

# These are the default versions (image tags) for their respective base images.
readonly __default_distroless_iptables_version=v0.8.6
readonly __default_go_runner_version=v2.4.0-go1.25.5-bookworm.0
readonly __default_setcap_version=bookworm-v1.0.6

# The default image for all binaries which are dynamically linked.
# Includes everything that is required by kube-proxy, which uses it
# by default. Other commands only use this when dynamically linking
# them gets requested.
readonly __default_dynamic_base_image="$KUBE_BASE_IMAGE_REGISTRY/distroless-iptables:$__default_distroless_iptables_version"

# KUBE_GORUNNER_IMAGE is the default image for commands which are built statically.
# It can be overridden to change the image for all such commands.
# When the per-command env variable is set, that env variable is
# used without considering KUBE_GORUNNER_IMAGE.
readonly KUBE_GORUNNER_IMAGE="${KUBE_GORUNNER_IMAGE:-$KUBE_BASE_IMAGE_REGISTRY/go-runner:$__default_go_runner_version}"

# __default_base_image takes the canonical build target for a Kubernetes command (e.g. k8s.io/kubernetes/cmd/kube-scheduler)
# and prints the right default base image for it, depending on whether that command gets built dynamically or statically.
__default_base_image() {
  if kube::golang::is_statically_linked "$1"; then
    echo "$KUBE_GORUNNER_IMAGE"
  else
    echo "$__default_dynamic_base_image"
  fi
}

# These are the base images for the Docker-wrapped binaries.
# These can be overridden on a case-by-case basis.
readonly KUBE_APISERVER_BASE_IMAGE="${KUBE_APISERVER_BASE_IMAGE:-$(__default_base_image k8s.io/kubernetes/cmd/kube-apiserver)}"
readonly KUBE_CONTROLLER_MANAGER_BASE_IMAGE="${KUBE_CONTROLLER_MANAGER_BASE_IMAGE:-$(__default_base_image k8s.io/kubernetes/cmd/kube-controller-manager)}"
readonly KUBE_SCHEDULER_BASE_IMAGE="${KUBE_SCHEDULER_BASE_IMAGE:-$(__default_base_image k8s.io/kubernetes/cmd/kube-scheduler)}"
readonly KUBE_PROXY_BASE_IMAGE="${KUBE_PROXY_BASE_IMAGE:-$__default_dynamic_base_image}"
readonly KUBECTL_BASE_IMAGE="${KUBECTL_BASE_IMAGE:-$(__default_base_image k8s.io/kubernetes/cmd/kubectl)}"

# This is the image used in a multi-stage build to apply capabilities to Docker-wrapped binaries.
readonly KUBE_BUILD_SETCAP_IMAGE="${KUBE_BUILD_SETCAP_IMAGE:-$KUBE_BASE_IMAGE_REGISTRY/setcap:$__default_setcap_version}"

# Get the set of master binaries that run in Docker (on Linux)
# Entry format is "<binary-name>,<base-image>".
# Binaries are placed in /usr/local/bin inside the image.
# `make` users can override any or all of the base images using the associated
# environment variables.
#
# $1 - server architecture
kube::build::get_docker_wrapped_binaries() {
  ### If you change any of these lists, please also update DOCKERIZED_BINARIES
  ### in build/BUILD. And kube::golang::server_image_targets
  local targets=(
    "kube-apiserver,${KUBE_APISERVER_BASE_IMAGE}"
    "kube-controller-manager,${KUBE_CONTROLLER_MANAGER_BASE_IMAGE}"
    "kube-scheduler,${KUBE_SCHEDULER_BASE_IMAGE}"
    "kube-proxy,${KUBE_PROXY_BASE_IMAGE}"
    "kubectl,${KUBECTL_BASE_IMAGE}"
  )

  echo "${targets[@]}"
}

# ---------------------------------------------------------------------------
# Basic setup functions

# Set up dynamic constants for build environment.
# This function sets up variables that are needed by both verification and cleaning.
#
# Vars set:
#   KUBE_ROOT_HASH
#   KUBE_BUILD_CONTAINER_NAME_BASE
#   KUBE_BUILD_CONTAINER_NAME
function kube::build::setup_vars() {
  KUBE_GIT_BRANCH=$(git symbolic-ref --short -q HEAD 2>/dev/null || true)
  KUBE_ROOT_HASH=$(kube::build::short_hash "${HOSTNAME:-}:${KUBE_ROOT}:${KUBE_GIT_BRANCH}")
  KUBE_BUILD_CONTAINER_NAME_BASE="kube-build-${KUBE_ROOT_HASH}"
  # 6 here is out of a wild excess of caution to match previous behavior where
  # this was the kube-build image version which surfaced in the name of the container
  # the last real image version was 5
  KUBE_BUILD_CONTAINER_NAME="${KUBE_BUILD_CONTAINER_NAME_BASE}-6"
}

# Verify that the right utilities and such are installed for building Kube. Set
# up some dynamic constants.
# Args:
#   $1 - boolean of whether to require functioning docker (default true)
#
# Vars set:
#   KUBE_ROOT_HASH
#   KUBE_BUILD_CONTAINER_NAME_BASE
#   KUBE_BUILD_CONTAINER_NAME
# shellcheck disable=SC2120 # optional parameters
function kube::build::verify_prereqs() {
  local -r require_docker=${1:-true}
  kube::log::status "Verifying Prerequisites...."
  kube::build::ensure_tar || return 1
  if ${require_docker}; then
    kube::build::ensure_docker_in_path || return 1
    if kube::build::is_osx; then
        kube::build::docker_available_on_osx || return 1
    fi
    kube::util::ensure_docker_daemon_connectivity || return 1

    if (( KUBE_VERBOSE > 6 )); then
      kube::log::status "Docker Version:"
      "${DOCKER[@]}" version | kube::log::info_from_stdin
    fi
  fi

  kube::build::setup_vars

  kube::version::get_version_vars
  kube::version::save_version_vars "${KUBE_ROOT}/.dockerized-kube-version-defs"

  # Without this, the user's umask can leak through.
  umask 0022
}

# ---------------------------------------------------------------------------
# Utility functions

function kube::build::docker_available_on_osx() {
  if [[ -z "${DOCKER_HOST}" ]]; then
    if [[ -S "/var/run/docker.sock" ]] || [[ -S "$(docker context inspect --format  '{{.Endpoints.docker.Host}}' | awk -F 'unix://' '{print $2}')" ]]; then
      kube::log::status "Using docker on macOS"
      return 0
    fi

    kube::log::status "No docker host is set."
    kube::log::status "It looks like you're running Mac OS X, but Docker for Mac cannot be found."
    kube::log::status "See: https://docs.docker.com/engine/installation/mac/ for installation instructions."
    return 1
  fi
}

function kube::build::is_osx() {
  [[ "$(uname)" == "Darwin" ]]
}

function kube::build::is_gnu_sed() {
  [[ $(sed --version 2>&1) == *GNU* ]]
}

function kube::build::ensure_docker_in_path() {
  if [[ -z "$(which docker)" ]]; then
    kube::log::error "Can't find 'docker' in PATH, please fix and retry."
    kube::log::error "See https://docs.docker.com/installation/#installation for installation instructions."
    return 1
  fi
}

function kube::build::ensure_tar() {
  if [[ -n "${TAR:-}" ]]; then
    return
  fi

  # Find gnu tar if it is available, bomb out if not.
  TAR=tar
  if which gtar &>/dev/null; then
      TAR=gtar
  else
      if which gnutar &>/dev/null; then
	  TAR=gnutar
      fi
  fi
  if ! "${TAR}" --version | grep -q GNU; then
    echo "  !!! Cannot find GNU tar. Build on Linux or install GNU tar"
    echo "      on Mac OS X (brew install gnu-tar)."
    return 1
  fi
}

function kube::build::has_docker() {
  which docker &> /dev/null
}

function kube::build::has_ip() {
  which ip &> /dev/null && ip -Version | grep 'iproute2' &> /dev/null
}

# Detect if a specific image exists
#
# $1 - image repo name
# $2 - image tag
function kube::build::docker_image_exists() {
  [[ -n $1 && -n $2 ]] || {
    kube::log::error "Internal error. Image not specified in docker_image_exists."
    exit 2
  }

  [[ $("${DOCKER[@]}" images -q "${1}:${2}") ]]
}

# Stop and delete all containers that match a pattern
#
# $1: The base container prefix
# $2: The current container to keep, if provided
function kube::build::docker_delete_old_containers() {
  # In Docker 1.12 we can replace this line with
  #   docker ps -a --format="{{.Names}}"
  for container in $("${DOCKER[@]}" ps -a | tail -n +2 | awk '{print $NF}') ; do
    if [[ "${container}" != "${1}"* ]] ; then
      V=3 kube::log::status "Keeping container ${container}"
      continue
    fi
    if [[ -z "${2:-}" || "${container}" != "${2}" ]] ; then
      V=2 kube::log::status "Deleting container ${container}"
      kube::build::destroy_container "${container}"
    else
      V=3 kube::log::status "Keeping container ${container}"
    fi
  done
}

# Takes $1 and computes a short hash for it. Useful for unique tag generation
function kube::build::short_hash() {
  [[ $# -eq 1 ]] || {
    kube::log::error "Internal error.  No data based to short_hash."
    exit 2
  }

  local short_hash
  if which md5 >/dev/null 2>&1; then
    short_hash=$(md5 -q -s "$1")
  else
    short_hash=$(echo -n "$1" | md5sum)
  fi
  echo "${short_hash:0:10}"
}

# Pedantically kill, wait-on and remove a container. The -f -v options
# to rm don't actually seem to get the job done, so force kill the
# container, wait to ensure it's stopped, then try the remove. This is
# a workaround for bug https://github.com/docker/docker/issues/3968.
function kube::build::destroy_container() {
  "${DOCKER[@]}" kill "$1" >/dev/null 2>&1 || true
  if [[ $("${DOCKER[@]}" version --format '{{.Server.Version}}') = 17.06.0* ]]; then
    # Workaround https://github.com/moby/moby/issues/33948.
    # TODO: remove when 17.06.0 is not relevant anymore
    DOCKER_API_VERSION=v1.29 "${DOCKER[@]}" wait "$1" >/dev/null 2>&1 || true
  else
    "${DOCKER[@]}" wait "$1" >/dev/null 2>&1 || true
  fi
  "${DOCKER[@]}" rm -f -v "$1" >/dev/null 2>&1 || true
}

function kube::build::is_docker_rootless() {
  "${DOCKER[@]}" info --format '{{json .SecurityOptions}}' | grep -q "name=rootless"
}

# ---------------------------------------------------------------------------
# Building


function kube::build::clean() {
  if kube::build::has_docker ; then
    kube::build::docker_delete_old_containers "${KUBE_BUILD_CONTAINER_NAME_BASE}"

    V=2 kube::log::status "Cleaning all untagged docker images"
    "${DOCKER[@]}" rmi "$("${DOCKER[@]}" images -q --filter 'dangling=true')" 2> /dev/null || true
  fi

  if [[ -d "${LOCAL_OUTPUT_ROOT}" ]]; then
    kube::log::status "Removing _output directory"
    # This ensures we can clean _output/local/go/cache which is not rw by default.
    #
    # We only do this path specifically instead of the entire output root
    # because recursive chmod is slow.
    # We don't need to do this at all for dockerized builds
    if [[ -d "${LOCAL_OUTPUT_ROOT}/local/go/cache" ]]; then
      chmod -R +w "${LOCAL_OUTPUT_ROOT}/local/go/cache"
    fi
    rm -rf "${LOCAL_OUTPUT_ROOT}"
  fi
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.
function kube::build::run_build_command() {
  kube::log::status "Running build command..."
  kube::build::run_build_command_ex "${KUBE_BUILD_CONTAINER_NAME}" -- "$@"
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.
#
# Arguments are in the form of
#  <container name> <extra docker args> -- <command>
function kube::build::run_build_command_ex() {
  [[ $# != 0 ]] || { echo "Invalid input - please specify a container name." >&2; return 4; }
  local container_name="${1}"
  shift

  local -a docker_run_opts=(
    "--name=${container_name}"
    "--hostname=${HOSTNAME}"
    "-e=GOPROXY=${GOPROXY}"
  )

  kube::build::is_docker_rootless || docker_run_opts+=("--user=$(id -u):$(id -g)")

  local detach=false

  [[ $# != 0 ]] || { echo "Invalid input - please specify docker arguments followed by --." >&2; return 4; }
  # Everything before "--" is an arg to docker
  until [ -z "${1-}" ] ; do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    docker_run_opts+=("$1")
    if [[ "$1" == "-d" || "$1" == "--detach" ]] ; then
      detach=true
    fi
    shift
  done

  # Everything after "--" is the command to run
  [[ $# != 0 ]] || { echo "Invalid input - please specify a command to run." >&2; return 4; }
  local -a cmd=()
  until [ -z "${1-}" ] ; do
    cmd+=("$1")
    shift
  done

  docker_run_opts+=(
    --env "KUBE_FASTBUILD=${KUBE_FASTBUILD:-false}"
    --env "KUBE_BUILDER_OS=${OSTYPE:-notdetected}"
    --env "KUBE_VERBOSE=${KUBE_VERBOSE}"
    --env "KUBE_BUILD_WITH_COVERAGE=${KUBE_BUILD_WITH_COVERAGE:-}"
    --env "KUBE_BUILD_PLATFORMS=${KUBE_BUILD_PLATFORMS:-}"
    --env "KUBE_CGO_OVERRIDES=' ${KUBE_CGO_OVERRIDES[*]:-} '"
    --env "KUBE_STATIC_OVERRIDES=' ${KUBE_STATIC_OVERRIDES[*]:-} '"
    --env "KUBE_RACE=${KUBE_RACE:-}"
    --env "FORCE_HOST_GO=${FORCE_HOST_GO:-}"
    --env "GO_VERSION=${GO_VERSION:-}"
    --env "GOTOOLCHAIN=${GOTOOLCHAIN:-}"
    --env "GOFLAGS=${GOFLAGS:-}"
    --env "GOGCFLAGS=${GOGCFLAGS:-}"
    --env "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-}"
    # mount source code / output dir
    --volume "${KUBE_ROOT}:${KUBE_CROSS_CONTAINER_ROOT}"
    # env migrated from build-image, we could consider setting this in kube-cross
    --env 'KUBE_OUTPUT_SUBPATH=_output/dockerized'
    --workdir "${KUBE_CROSS_CONTAINER_ROOT}"
    --env 'GIT_AUTHOR_EMAIL=nobody@k8s.io'
    --env 'GIT_AUTHOR_NAME=kube-build-image'
  )

  # if host has localtime, mount it so we log in local time
  if [ -f /etc/localtime ]; then
    docker_run_opts+=(
      --mount 'type=bind,source=/etc/localtime,target=/etc/localtime,readonly'
    )
  fi

  # use GOLDFLAGS only if it is set explicitly.
  if [[ -v GOLDFLAGS ]]; then
    docker_run_opts+=(
      --env "GOLDFLAGS=${GOLDFLAGS:-}"
    )
  fi

  if [[ -n "${DOCKER_CGROUP_PARENT:-}" ]]; then
    kube::log::status "Using ${DOCKER_CGROUP_PARENT} as container cgroup parent"
    docker_run_opts+=(--cgroup-parent "${DOCKER_CGROUP_PARENT}")
  fi

  # copy KUBE_GIT_VERSION_FILE to .dockerized-kube-version-defs and set environment variable.
  if [[ -n "${KUBE_GIT_VERSION_FILE:-}" ]]; then
    cp "${KUBE_GIT_VERSION_FILE}" "${KUBE_ROOT}/.dockerized-kube-version-defs"
    docker_run_opts+=(--env "KUBE_GIT_VERSION_FILE=${KUBE_CROSS_CONTAINER_ROOT}/.dockerized-kube-version-defs")
  fi

  # If we have stdin we can run interactive.  This allows things like 'shell.sh'
  # to work.  However, if we run this way and don't have stdin, then it ends up
  # running in a daemon-ish mode.  So if we don't have a stdin, we explicitly
  # attach stderr/stdout but don't bother asking for a tty.
  if [[ -t 0 ]]; then
    docker_run_opts+=(--interactive --tty)
  elif [[ "${detach}" == false ]]; then
    docker_run_opts+=("--attach=stdout" "--attach=stderr")
  fi

  local -ra docker_cmd=(
    "${DOCKER[@]}" run "${docker_run_opts[@]}" "${KUBE_CROSS_IMAGE}:${KUBE_CROSS_VERSION}")

  # Clean up container from any previous run
  kube::build::destroy_container "${container_name}"
  "${docker_cmd[@]}" "${cmd[@]}"
  if [[ "${detach}" == false ]]; then
    kube::build::destroy_container "${container_name}"
  fi
}
