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

# Common utilities, variables and checks for all build scripts.
set -o errexit
set -o nounset
set -o pipefail

DOCKER_OPTS=${DOCKER_OPTS:-""}
DOCKER_NATIVE=${DOCKER_NATIVE:-""}
DOCKER=(docker ${DOCKER_OPTS})
DOCKER_HOST=${DOCKER_HOST:-""}
DOCKER_MACHINE_NAME=${DOCKER_MACHINE_NAME:-"kube-dev"}
readonly DOCKER_MACHINE_DRIVER=${DOCKER_MACHINE_DRIVER:-"virtualbox --virtualbox-memory 4096 --virtualbox-cpu-count -1"}

# This will canonicalize the path
KUBE_ROOT=$(cd $(dirname "${BASH_SOURCE}")/.. && pwd -P)

source "${KUBE_ROOT}/hack/lib/init.sh"

# Set KUBE_BUILD_PPC64LE to y to build for ppc64le in addition to other
# platforms.
# TODO(IBM): remove KUBE_BUILD_PPC64LE and reenable ppc64le compilation by
# default when
# https://github.com/kubernetes/kubernetes/issues/30384 and
# https://github.com/kubernetes/kubernetes/issues/25886 are fixed.
# The majority of the logic is in hack/lib/golang.sh.
readonly KUBE_BUILD_PPC64LE="${KUBE_BUILD_PPC64LE:-n}"

# Constants
readonly KUBE_BUILD_IMAGE_REPO=kube-build
readonly KUBE_BUILD_IMAGE_CROSS_TAG="$(cat ${KUBE_ROOT}/build/build-image/cross/VERSION)"
# KUBE_BUILD_DATA_CONTAINER_NAME=kube-build-data-<hash>"

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

readonly REMOTE_ROOT="/go/src/${KUBE_GO_PACKAGE}"
readonly REMOTE_OUTPUT_ROOT="${REMOTE_ROOT}/_output"
readonly REMOTE_OUTPUT_SUBPATH="${REMOTE_OUTPUT_ROOT}/dockerized"
readonly REMOTE_OUTPUT_BINPATH="${REMOTE_OUTPUT_SUBPATH}/bin"
readonly REMOTE_OUTPUT_GOPATH="${REMOTE_OUTPUT_SUBPATH}/go"

readonly DOCKER_MOUNT_ARGS_BASE=(
  # timezone
  --volume /etc/localtime:/etc/localtime:ro
)

# Get the set of master binaries that run in Docker (on Linux)
# Entry format is "<name-of-binary>,<base-image>".
# Binaries are placed in /usr/local/bin inside the image.
#
# $1 - server architecture
kube::build::get_docker_wrapped_binaries() {
  case $1 in
    "amd64")
        local targets=(
          kube-apiserver,busybox
          kube-controller-manager,busybox
          kube-scheduler,busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-amd64:v3
          federation-apiserver,busybox
          federation-controller-manager,busybox
        );;
    "arm")
        local targets=(
          kube-apiserver,armel/busybox
          kube-controller-manager,armel/busybox
          kube-scheduler,armel/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-arm:v3
          federation-apiserver,armel/busybox
          federation-controller-manager,armel/busybox
        );;
    "arm64")
        local targets=(
          kube-apiserver,aarch64/busybox
          kube-controller-manager,aarch64/busybox
          kube-scheduler,aarch64/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-arm64:v3
          federation-apiserver,aarch64/busybox
          federation-controller-manager,aarch64/busybox
        );;
    "ppc64le")
        local targets=(
          kube-apiserver,ppc64le/busybox
          kube-controller-manager,ppc64le/busybox
          kube-scheduler,ppc64le/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-ppc64le:v3
          federation-apiserver,ppc64le/busybox
          federation-controller-manager,ppc64le/busybox
        );;
  esac

  echo "${targets[@]}"
}

# ---------------------------------------------------------------------------
# Basic setup functions

# Verify that the right utilities and such are installed for building Kube. Set
# up some dynamic constants.
#
# Vars set:
#   KUBE_ROOT_HASH
#   KUBE_BUILD_IMAGE_TAG
#   KUBE_BUILD_IMAGE
#   KUBE_BUILD_CONTAINER_NAME
#   KUBE_BUILD_DATA_CONTAINER_NAME
#   DOCKER_MOUNT_ARGS
#   LOCAL_OUTPUT_BUILD_CONTEXT
function kube::build::verify_prereqs() {
  kube::log::status "Verifying Prerequisites...."
  kube::build::ensure_tar || return 1
  kube::build::ensure_docker_in_path || return 1
  if kube::build::is_osx; then
      kube::build::docker_available_on_osx || return 1
  fi
  kube::build::ensure_docker_daemon_connectivity || return 1

  KUBE_ROOT_HASH=$(kube::build::short_hash "${HOSTNAME:-}:${KUBE_ROOT}")
  KUBE_BUILD_IMAGE_TAG="build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_IMAGE="${KUBE_BUILD_IMAGE_REPO}:${KUBE_BUILD_IMAGE_TAG}"
  KUBE_BUILD_CONTAINER_NAME="kube-build-${KUBE_ROOT_HASH}"
  KUBE_RSYNC_CONTAINER_NAME="kube-rsync-${KUBE_ROOT_HASH}"
  KUBE_BUILD_DATA_CONTAINER_NAME="kube-build-data-${KUBE_ROOT_HASH}"
  DOCKER_MOUNT_ARGS=("${DOCKER_MOUNT_ARGS_BASE[@]}" --volumes-from "${KUBE_BUILD_DATA_CONTAINER_NAME}")
  LOCAL_OUTPUT_BUILD_CONTEXT="${LOCAL_OUTPUT_IMAGE_STAGING}/${KUBE_BUILD_IMAGE}"
}

# ---------------------------------------------------------------------------
# Utility functions

function kube::build::docker_available_on_osx() {
  if [[ -z "${DOCKER_HOST}" ]]; then
    if [[ -S "/var/run/docker.sock" ]]; then
      kube::log::status "Using Docker for MacOS"
      return 0
    fi

    kube::log::status "No docker host is set. Checking options for setting one..."
    if [[ -z "$(which docker-machine)" ]]; then
      kube::log::status "It looks like you're running Mac OS X, yet neither Docker for Mac or docker-machine can be found."
      kube::log::status "See: https://docs.docker.com/engine/installation/mac/ for installation instructions."
      return 1
    elif [[ -n "$(which docker-machine)" ]]; then
      kube::build::prepare_docker_machine
    fi
  fi
}

function kube::build::prepare_docker_machine() {
  kube::log::status "docker-machine was found."
  docker-machine inspect "${DOCKER_MACHINE_NAME}" &> /dev/null || {
    kube::log::status "Creating a machine to build Kubernetes"
    docker-machine create --driver ${DOCKER_MACHINE_DRIVER} \
      --engine-env HTTP_PROXY="${KUBERNETES_HTTP_PROXY:-}" \
      --engine-env HTTPS_PROXY="${KUBERNETES_HTTPS_PROXY:-}" \
      --engine-env NO_PROXY="${KUBERNETES_NO_PROXY:-127.0.0.1}" \
      "${DOCKER_MACHINE_NAME}" > /dev/null || {
      kube::log::error "Something went wrong creating a machine."
      kube::log::error "Try the following: "
      kube::log::error "docker-machine create -d ${DOCKER_MACHINE_DRIVER} ${DOCKER_MACHINE_NAME}"
      return 1
    }
  }
  docker-machine start "${DOCKER_MACHINE_NAME}" &> /dev/null
  # it takes `docker-machine env` a few seconds to work if the machine was just started
  local docker_machine_out
  while ! docker_machine_out=$(docker-machine env "${DOCKER_MACHINE_NAME}" 2>&1); do
    if [[ ${docker_machine_out} =~ "Error checking TLS connection" ]]; then
      echo ${docker_machine_out}
      docker-machine regenerate-certs ${DOCKER_MACHINE_NAME}
    else
      sleep 1
    fi
  done
  eval $(docker-machine env "${DOCKER_MACHINE_NAME}")
  kube::log::status "A Docker host using docker-machine named '${DOCKER_MACHINE_NAME}' is ready to go!"
  return 0
}

function kube::build::is_osx() {
  [[ "$(uname)" == "Darwin" ]]
}

function kube::build::is_gnu_sed() {
  [[ $(sed --version 2>&1) == *GNU* ]]
}

function kube::build::update_dockerfile() {
  if kube::build::is_gnu_sed; then
    sed_opts=(-i)
  else
    sed_opts=(-i '')
  fi
  sed "${sed_opts[@]}" "s/KUBE_BUILD_IMAGE_CROSS_TAG/${KUBE_BUILD_IMAGE_CROSS_TAG}/" "${LOCAL_OUTPUT_BUILD_CONTEXT}/Dockerfile"
}

function kube::build::ensure_docker_in_path() {
  if [[ -z "$(which docker)" ]]; then
    kube::log::error "Can't find 'docker' in PATH, please fix and retry."
    kube::log::error "See https://docs.docker.com/installation/#installation for installation instructions."
    return 1
  fi
}

function kube::build::ensure_docker_daemon_connectivity {
  if ! "${DOCKER[@]}" info > /dev/null 2>&1 ; then
    {
      echo "Can't connect to 'docker' daemon.  please fix and retry."
      echo
      echo "Possible causes:"
      echo "  - On Mac OS X, DOCKER_HOST hasn't been set. You may need to: "
      echo "    - Set up Docker for Mac (https://docs.docker.com/docker-for-mac/)"
      echo "    - Or, set up docker-machine"
      echo "      - Create and start your VM using docker-machine: "
      echo "        - docker-machine create -d ${DOCKER_MACHINE_DRIVER} ${DOCKER_MACHINE_NAME}"
      echo "      - Set your environment variables using: "
      echo "        - eval \$(docker-machine env ${DOCKER_MACHINE_NAME})"
      echo "      - Update your Docker VM"
      echo "        - Error Message: 'Error response from daemon: client is newer than server (...)' "
      echo "        - docker-machine upgrade ${DOCKER_MACHINE_NAME}"
      echo "  - On Linux, user isn't in 'docker' group.  Add and relogin."
      echo "    - Something like 'sudo usermod -a -G docker ${USER-user}'"
      echo "    - RHEL7 bug and workaround: https://bugzilla.redhat.com/show_bug.cgi?id=1119282#c8"
      echo "  - On Linux, Docker daemon hasn't been started or has crashed."
    } >&2
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

function kube::build::clean_output() {
  # Clean out the output directory if it exists.
  if kube::build::has_docker ; then
    if kube::build::build_image_built ; then
      kube::log::status "Cleaning out _output/dockerized/bin/ via docker build image"
      kube::build::run_build_command bash -c "rm -rf '${REMOTE_OUTPUT_BINPATH}'/*"
    else
      kube::log::error "Build image not built.  Cannot clean via docker build image."
    fi

    kube::log::status "Removing data container ${KUBE_BUILD_DATA_CONTAINER_NAME}"
    "${DOCKER[@]}" rm -v "${KUBE_BUILD_DATA_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  kube::log::status "Removing _output directory"
  rm -rf "${LOCAL_OUTPUT_ROOT}"
}

# Make sure the _output directory is created and mountable by docker
function kube::build::prepare_output() {
  # See auto-creation of host mounts: https://github.com/docker/docker/pull/21666
  # if selinux is enabled, docker run -v /foo:/foo:Z will not autocreate the host dir
  mkdir -p "${LOCAL_OUTPUT_SUBPATH}"
  mkdir -p "${LOCAL_OUTPUT_BINPATH}"
  # On RHEL/Fedora SELinux is enabled by default and currently breaks docker
  # volume mounts.  We can work around this by explicitly adding a security
  # context to the _output directory.
  # Details: http://www.projectatomic.io/blog/2015/06/using-volumes-with-docker-can-cause-problems-with-selinux/
  if which selinuxenabled &>/dev/null && \
      selinuxenabled && \
      which chcon >/dev/null ; then
    if [[ ! $(ls -Zd "${LOCAL_OUTPUT_ROOT}") =~ svirt_sandbox_file_t ]] ; then
      kube::log::status "Applying SELinux policy to '_output' directory."
      if ! chcon -Rt svirt_sandbox_file_t "${LOCAL_OUTPUT_ROOT}"; then
        echo "    ***Failed***.  This may be because you have root owned files under _output."
        echo "    Continuing, but this build may fail later if SELinux prevents access."
      fi
    fi
    number=${#DOCKER_MOUNT_ARGS[@]}
    for (( i=0; i<number; i++ )); do
      if [[ "${DOCKER_MOUNT_ARGS[i]}" =~ "${KUBE_ROOT}" ]]; then
        ## Ensure we don't label the argument multiple times
        if [[ ! "${DOCKER_MOUNT_ARGS[i]}" == *:Z ]]; then
          DOCKER_MOUNT_ARGS[i]="${DOCKER_MOUNT_ARGS[i]}:Z"
        fi
      fi
    done
  fi

}

function kube::build::has_docker() {
  which docker &> /dev/null
}

# Detect if a specific image exists
#
# $1 - image repo name
# #2 - image tag
function kube::build::docker_image_exists() {
  [[ -n $1 && -n $2 ]] || {
    kube::log::error "Internal error. Image not specified in docker_image_exists."
    exit 2
  }

  # We cannot just specify the IMAGE here as `docker images` doesn't behave as
  # expected.  See: https://github.com/docker/docker/issues/8048
  # Also we cannot use the `-q` option on grep as it causes pipefail to trigger.
  # See http://stackoverflow.com/questions/19120263/why-exit-code-141-with-grep-q
  "${DOCKER[@]}" images | grep -E "^(\S+/)?${1}\s+${2}\s+" > /dev/null
}

# Takes $1 and computes a short has for it. Useful for unique tag generation
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
  echo ${short_hash:0:10}
}

# Pedantically kill, wait-on and remove a container. The -f -v options
# to rm don't actually seem to get the job done, so force kill the
# container, wait to ensure it's stopped, then try the remove. This is
# a workaround for bug https://github.com/docker/docker/issues/3968.
function kube::build::destroy_container() {
  "${DOCKER[@]}" kill "$1" >/dev/null 2>&1 || true
  "${DOCKER[@]}" wait "$1" >/dev/null 2>&1 || true
  "${DOCKER[@]}" rm -f -v "$1" >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# Building

function kube::build::build_image_built() {
  kube::build::docker_image_exists "${KUBE_BUILD_IMAGE_REPO}" "${KUBE_BUILD_IMAGE_TAG}"
}

# Set up the context directory for the kube-build image and build it.
function kube::build::build_image() {
  mkdir -p "${LOCAL_OUTPUT_BUILD_CONTEXT}"

  kube::version::get_version_vars
  kube::version::save_version_vars "${LOCAL_OUTPUT_BUILD_CONTEXT}/kube-version-defs"

  cp build/build-image/Dockerfile "${LOCAL_OUTPUT_BUILD_CONTEXT}/Dockerfile"
  cp build/build-image/rsyncd.sh "${LOCAL_OUTPUT_BUILD_CONTEXT}/"
  kube::build::update_dockerfile

  kube::build::docker_build "${KUBE_BUILD_IMAGE}" "${LOCAL_OUTPUT_BUILD_CONTEXT}" 'false'

  kube::build::sync_to_container
}

# Build a docker image from a Dockerfile.
# $1 is the name of the image to build
# $2 is the location of the "context" directory, with the Dockerfile at the root.
# $3 is the value to set the --pull flag for docker build; true by default
function kube::build::docker_build() {
  local -r image=$1
  local -r context_dir=$2
  local -r pull="${3:-true}"
  local -ra build_cmd=("${DOCKER[@]}" build -t "${image}" "--pull=${pull}" "${context_dir}")

  kube::log::status "Building Docker image ${image}"
  local docker_output
  docker_output=$("${build_cmd[@]}" 2>&1) || {
    cat <<EOF >&2
+++ Docker build command failed for ${image}

${docker_output}

To retry manually, run:

${build_cmd[*]}

EOF
    return 1
  }
}

function kube::build::clean_image() {
  local -r image=$1

  kube::log::status "Deleting docker image ${image}"
  "${DOCKER[@]}" rmi ${image} 2> /dev/null || true
}

function kube::build::clean_images() {
  kube::build::has_docker || return 0

  kube::build::clean_image "${KUBE_BUILD_IMAGE}"

  kube::log::status "Cleaning all other untagged docker images"
  "${DOCKER[@]}" rmi $("${DOCKER[@]}" images -q --filter 'dangling=true') 2> /dev/null || true
}

function kube::build::ensure_data_container() {
  # If the data container exists AND exited successfully, we can use it.
  # Otherwise nuke it and start over.
  local ret=0
  local code=$(docker inspect \
      -f '{{.State.ExitCode}}' \
      "${KUBE_BUILD_DATA_CONTAINER_NAME}" 2>/dev/null || ret=$?)
  if [[ "${ret}" == 0 && "${code}" != 0 ]]; then
    kube::build::destroy_container "${KUBE_BUILD_DATA_CONTAINER_NAME}"
    ret=1
  fi
  if [[ "${ret}" != 0 ]]; then
    kube::log::status "Creating data container ${KUBE_BUILD_DATA_CONTAINER_NAME}"
    # We have to ensure the directory exists, or else the docker run will
    # create it as root.
    mkdir -p "${LOCAL_OUTPUT_GOPATH}"
    # We want this to run as root to be able to chown, so non-root users can
    # later use the result as a data container.  This run both creates the data
    # container and chowns the GOPATH.
    #
    # The data container creates volumes for all of the directories that store
    # intermediates for the Go build. This enables incremental builds across
    # Docker sessions. The *_cgo paths are re-compiled versions of the go std
    # libraries for true static building.
    local -ra docker_cmd=(
      "${DOCKER[@]}" run
      --volume "${REMOTE_ROOT}"   # white-out the whole output dir
      --volume /usr/local/go/pkg/linux_386_cgo
      --volume /usr/local/go/pkg/linux_amd64_cgo
      --volume /usr/local/go/pkg/linux_arm_cgo
      --volume /usr/local/go/pkg/linux_arm64_cgo
      --volume /usr/local/go/pkg/linux_ppc64le_cgo
      --volume /usr/local/go/pkg/darwin_amd64_cgo
      --volume /usr/local/go/pkg/darwin_386_cgo
      --volume /usr/local/go/pkg/windows_amd64_cgo
      --volume /usr/local/go/pkg/windows_386_cgo
      --name "${KUBE_BUILD_DATA_CONTAINER_NAME}"
      --hostname "${HOSTNAME}"
      "${KUBE_BUILD_IMAGE}"
      chown -R $(id -u).$(id -g)
        "${REMOTE_ROOT}"
        /usr/local/go/pkg/linux_386_cgo
        /usr/local/go/pkg/linux_amd64_cgo
        /usr/local/go/pkg/linux_arm_cgo
        /usr/local/go/pkg/linux_arm64_cgo
        /usr/local/go/pkg/linux_ppc64le_cgo
        /usr/local/go/pkg/darwin_amd64_cgo
        /usr/local/go/pkg/darwin_386_cgo
        /usr/local/go/pkg/windows_amd64_cgo
        /usr/local/go/pkg/windows_386_cgo
    )
    "${docker_cmd[@]}"
  fi
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
function kube::build::run_build_command() {
  kube::log::status "Running build command..."
  kube::build::run_build_command_ex "${KUBE_BUILD_CONTAINER_NAME}" -- "$@"
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
#
# Arguments are in the form of
#  <container name> <extra docker args> -- <command>
function kube::build::run_build_command_ex() {
  kube::build::ensure_data_container
  kube::build::prepare_output

  [[ $# != 0 ]] || { echo "Invalid input - please specify a the container name." >&2; return 4; }
  local container_name="${1}"
  shift

  local -a docker_run_opts=(
    "--name=${container_name}"
    "--user=$(id -u):$(id -g)"
    "--hostname=${HOSTNAME}"
    "${DOCKER_MOUNT_ARGS[@]}"
  )

  [[ $# != 0 ]] || { echo "Invalid input - please specify docker arguments followed by --." >&2; return 4; }
  # Everything before "--" is an arg to docker
  until [ -z "${1-}" ] ; do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    docker_run_opts+=("$1")
    shift
  done

  # Everything after "--" is the command to run
  [[ $# != 0 ]] || { echo "Invalid input - please specify a command to run." >&2; return 4; }
  local -a cmd=()
  until [ -z "${1-}" ] ; do
    cmd+=("$1")
    shift
  done

  if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
    docker_run_opts+=(-e "KUBERNETES_CONTRIB=${KUBERNETES_CONTRIB}")
  fi

  docker_run_opts+=(
    --env "KUBE_FASTBUILD=${KUBE_FASTBUILD:-false}"
    --env "KUBE_BUILDER_OS=${OSTYPE:-notdetected}"
    --env "KUBE_BUILD_PPC64LE=${KUBE_BUILD_PPC64LE}"  # TODO(IBM): remove
  )

  # If we have stdin we can run interactive.  This allows things like 'shell.sh'
  # to work.  However, if we run this way and don't have stdin, then it ends up
  # running in a daemon-ish mode.  So if we don't have a stdin, we explicitly
  # attach stderr/stdout but don't bother asking for a tty.
  if [[ -t 0 ]]; then
    docker_run_opts+=(--interactive --tty)
  else
    docker_run_opts+=(--attach=stdout --attach=stderr)
  fi

  local -ra docker_cmd=(
    "${DOCKER[@]}" run "${docker_run_opts[@]}" "${KUBE_BUILD_IMAGE}")

  # Clean up container from any previous run
  kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
  "${docker_cmd[@]}" "${cmd[@]}"
  kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
}

function kube::build::start_rsyncd_container() {
  kube::build::stop_rsyncd_container
  kube::build::run_build_command_ex \
    "${KUBE_RSYNC_CONTAINER_NAME}" -p 127.0.0.1:8730:8730 -d \
    -- /rsyncd.sh >/dev/null
}

function kube::build::stop_rsyncd_container() {
  kube::build::destroy_container "${KUBE_RSYNC_CONTAINER_NAME}"
}

# This will launch rsyncd in a container and then sync the source tree to the
# container over the local network.
function kube::build::sync_to_container() {
  kube::log::status "Syncing sources to container"

  kube::build::start_rsyncd_container

  rsync \
    --filter='- /.git/' \
    --filter='- /.make/' \
    --filter='- /_output/' \
    --filter='- /' \
    --prune-empty-dirs \
    -ap \
    "${KUBE_ROOT}/" rsync://localhost:8730/k8s/

  kube::build::stop_rsyncd_container
}

# If the Docker server is remote, copy the results back out.
function kube::build::copy_output() {
  kube::log::status "Syncing out of container"

  kube::build::start_rsyncd_container

  rsync \
    --prune-empty-dirs \
    --filter='+ /_output/dockerized/bin/**' \
    --filter='+ zz_generated.*' \
    --filter='+ */' \
    --filter='- /**' \
    -ap \
    rsync://localhost:8730/k8s/ "${KUBE_ROOT}"

  kube::build::stop_rsyncd_container
}
