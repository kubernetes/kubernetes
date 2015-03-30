#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
cd "${LMKTFY_ROOT}"

# This'll canonicalize the path
LMKTFY_ROOT=$PWD

source hack/lib/init.sh

# Incoming options
#
readonly LMKTFY_SKIP_CONFIRMATIONS="${LMKTFY_SKIP_CONFIRMATIONS:-n}"
readonly LMKTFY_GCS_UPLOAD_RELEASE="${LMKTFY_GCS_UPLOAD_RELEASE:-n}"
readonly LMKTFY_GCS_NO_CACHING="${LMKTFY_GCS_NO_CACHING:-y}"
readonly LMKTFY_GCS_MAKE_PUBLIC="${LMKTFY_GCS_MAKE_PUBLIC:-y}"
# LMKTFY_GCS_RELEASE_BUCKET default: lmktfy-releases-${project_hash}
readonly LMKTFY_GCS_RELEASE_PREFIX=${LMKTFY_GCS_RELEASE_PREFIX-devel}/
readonly LMKTFY_GCS_DOCKER_REG_PREFIX=${LMKTFY_GCS_DOCKER_REG_PREFIX-docker-reg}/
readonly LMKTFY_GCS_LATEST_FILE=${LMKTFY_GCS_LATEST_FILE:-}
readonly LMKTFY_GCS_LATEST_CONTENTS=${LMKTFY_GCS_LATEST_CONTENTS:-}

# Constants
readonly LMKTFY_BUILD_IMAGE_REPO=lmktfy-build
# These get set in verify_prereqs with a unique hash based on LMKTFY_ROOT
# LMKTFY_BUILD_IMAGE_TAG=<hash>
# LMKTFY_BUILD_IMAGE="${LMKTFY_BUILD_IMAGE_REPO}:${LMKTFY_BUILD_IMAGE_TAG}"
# LMKTFY_BUILD_CONTAINER_NAME=lmktfy-build-<hash>
readonly LMKTFY_BUILD_IMAGE_CROSS_TAG=cross
readonly LMKTFY_BUILD_IMAGE_CROSS="${LMKTFY_BUILD_IMAGE_REPO}:${LMKTFY_BUILD_IMAGE_CROSS_TAG}"
readonly LMKTFY_BUILD_GOLANG_VERSION=1.4
# LMKTFY_BUILD_DATA_CONTAINER_NAME=lmktfy-build-data-<hash>

# Here we map the output directories across both the local and remote _output
# directories:
#
# *_OUTPUT_ROOT    - the base of all output in that environment.
# *_OUTPUT_SUBPATH - location where golang stuff is built/cached.  Also
#                    persisted across docker runs with a volume mount.
# *_OUTPUT_BINPATH - location where final binaries are placed.  If the remote
#                    is really remote, this is the stuff that has to be copied
#                    back.
readonly LOCAL_OUTPUT_ROOT="${LMKTFY_ROOT}/_output"
readonly LOCAL_OUTPUT_SUBPATH="${LOCAL_OUTPUT_ROOT}/dockerized"
readonly LOCAL_OUTPUT_BINPATH="${LOCAL_OUTPUT_SUBPATH}/bin"
readonly LOCAL_OUTPUT_IMAGE_STAGING="${LOCAL_OUTPUT_ROOT}/images"

readonly OUTPUT_BINPATH="${CUSTOM_OUTPUT_BINPATH:-$LOCAL_OUTPUT_BINPATH}"

readonly REMOTE_OUTPUT_ROOT="/go/src/${LMKTFY_GO_PACKAGE}/_output"
readonly REMOTE_OUTPUT_SUBPATH="${REMOTE_OUTPUT_ROOT}/dockerized"
readonly REMOTE_OUTPUT_BINPATH="${REMOTE_OUTPUT_SUBPATH}/bin"

readonly DOCKER_MOUNT_ARGS_BASE=(--volume "${OUTPUT_BINPATH}:${REMOTE_OUTPUT_BINPATH}")
# DOCKER_MOUNT_ARGS=("${DOCKER_MOUNT_ARGS_BASE[@]}" --volumes-from "${LMKTFY_BUILD_DATA_CONTAINER_NAME}")

# We create a Docker data container to cache incremental build artifacts.  We
# need to cache both the go tree in _output and the go tree under Godeps.
readonly REMOTE_OUTPUT_GOPATH="${REMOTE_OUTPUT_SUBPATH}/go"
readonly REMOTE_GODEP_GOPATH="/go/src/${LMKTFY_GO_PACKAGE}/Godeps/_workspace/pkg"
readonly DOCKER_DATA_MOUNT_ARGS=(
  --volume "${REMOTE_OUTPUT_GOPATH}"
  --volume "${REMOTE_GODEP_GOPATH}"
)

# This is where the final release artifacts are created locally
readonly RELEASE_STAGE="${LOCAL_OUTPUT_ROOT}/release-stage"
readonly RELEASE_DIR="${LOCAL_OUTPUT_ROOT}/release-tars"

# ---------------------------------------------------------------------------
# Basic setup functions

# Verify that the right utilities and such are installed for building LMKTFY.  Set
# up some dynamic constants.
#
# Args:
#   $1 The type of operation to verify for.  Only 'clean' is supported in which
#   case we don't verify docker.
#
# Vars set:
#   LMKTFY_ROOT_HASH
#   LMKTFY_BUILD_IMAGE_TAG
#   LMKTFY_BUILD_IMAGE
#   LMKTFY_BUILD_CONTAINER_NAME
#   LMKTFY_BUILD_DATA_CONTAINER_NAME
#   DOCKER_MOUNT_ARGS
function lmktfy::build::verify_prereqs() {
  lmktfy::log::status "Verifying Prerequisites...."

  if [[ "${1-}" != "clean" ]]; then
    if [[ -z "$(which docker)" ]]; then
      echo "Can't find 'docker' in PATH, please fix and retry." >&2
      echo "See https://docs.docker.com/installation/#installation for installation instructions." >&2
      exit 1
    fi

    if lmktfy::build::is_osx; then
      if [[ -z "$DOCKER_NATIVE" ]];then
        if [[ -z "$(which boot2docker)" ]]; then
          echo "It looks like you are running on Mac OS X and boot2docker can't be found." >&2
          echo "See: https://docs.docker.com/installation/mac/" >&2
          exit 1
        fi
        if [[ $(boot2docker status) != "running" ]]; then
          echo "boot2docker VM isn't started.  Please run 'boot2docker start'" >&2
          exit 1
        else
          # Reach over and set the clock. After sleep/resume the clock will skew.
          lmktfy::log::status "Setting boot2docker clock"
          boot2docker ssh sudo date -u -D "%Y%m%d%H%M.%S" --set "$(date -u +%Y%m%d%H%M.%S)" >/dev/null
          if [[ -z "$DOCKER_HOST" ]]; then
            lmktfy::log::status "Setting boot2docker env variables"
            $(boot2docker shellinit)
          fi
        fi
      fi
    fi

    if ! "${DOCKER[@]}" info > /dev/null 2>&1 ; then
      {
        echo "Can't connect to 'docker' daemon.  please fix and retry."
        echo
        echo "Possible causes:"
        echo "  - On Mac OS X, boot2docker VM isn't installed or started"
        echo "  - On Mac OS X, docker env variable isn't set appropriately. Run:"
        echo "      \$(boot2docker shellinit)"
        echo "  - On Linux, user isn't in 'docker' group.  Add and relogin."
        echo "    - Something like 'sudo usermod -a -G docker ${USER-user}'"
        echo "    - RHEL7 bug and workaround: https://bugzilla.redhat.com/show_bug.cgi?id=1119282#c8"
        echo "  - On Linux, Docker daemon hasn't been started or has crashed"
      } >&2
      exit 1
    fi
  else

    # On OS X, set boot2docker env vars for the 'clean' target if boot2docker is running
    if lmktfy::build::is_osx && lmktfy::build::has_docker ; then
      if [[ ! -z "$(which boot2docker)" ]]; then
        if [[ $(boot2docker status) == "running" ]]; then
          if [[ -z "$DOCKER_HOST" ]]; then
            lmktfy::log::status "Setting boot2docker env variables"
            $(boot2docker shellinit)
          fi
        fi
      fi
    fi

  fi

  LMKTFY_ROOT_HASH=$(lmktfy::build::short_hash "$LMKTFY_ROOT")
  LMKTFY_BUILD_IMAGE_TAG="build-${LMKTFY_ROOT_HASH}"
  LMKTFY_BUILD_IMAGE="${LMKTFY_BUILD_IMAGE_REPO}:${LMKTFY_BUILD_IMAGE_TAG}"
  LMKTFY_BUILD_CONTAINER_NAME="lmktfy-build-${LMKTFY_ROOT_HASH}"
  LMKTFY_BUILD_DATA_CONTAINER_NAME="lmktfy-build-data-${LMKTFY_ROOT_HASH}"
  DOCKER_MOUNT_ARGS=("${DOCKER_MOUNT_ARGS_BASE[@]}" --volumes-from "${LMKTFY_BUILD_DATA_CONTAINER_NAME}")
}

# ---------------------------------------------------------------------------
# Utility functions

function lmktfy::build::is_osx() {
  [[ "$(uname)" == "Darwin" ]]
}

function lmktfy::build::clean_output() {
  # Clean out the output directory if it exists.
  if lmktfy::build::has_docker ; then
    if lmktfy::build::build_image_built ; then
      lmktfy::log::status "Cleaning out _output/dockerized/bin/ via docker build image"
      lmktfy::build::run_build_command bash -c "rm -rf '${REMOTE_OUTPUT_BINPATH}'/*"
    else
      lmktfy::log::error "Build image not built.  Cannot clean via docker build image."
    fi

    lmktfy::log::status "Removing data container"
    "${DOCKER[@]}" rm -v "${LMKTFY_BUILD_DATA_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  lmktfy::log::status "Cleaning out local _output directory"
  rm -rf "${LOCAL_OUTPUT_ROOT}"
}

# Make sure the _output directory is created and mountable by docker
function lmktfy::build::prepare_output() {
  mkdir -p "${LOCAL_OUTPUT_SUBPATH}"

  # On RHEL/Fedora SELinux is enabled by default and currently breaks docker
  # volume mounts.  We can work around this by explicitly adding a security
  # context to the _output directory.
  # Details: https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/7/html/Resource_Management_and_Linux_Containers_Guide/sec-Sharing_Data_Across_Containers.html#sec-Mounting_a_Host_Directory_to_a_Container
  if which selinuxenabled &>/dev/null && \
      selinuxenabled && \
      which chcon >/dev/null ; then
    if [[ ! $(ls -Zd "${LOCAL_OUTPUT_ROOT}") =~ svirt_sandbox_file_t ]] ; then
      lmktfy::log::status "Applying SELinux policy to '_output' directory."
      if ! chcon -Rt svirt_sandbox_file_t "${LOCAL_OUTPUT_ROOT}"; then
        echo "    ***Failed***.  This may be because you have root owned files under _output."
        echo "    Continuing, but this build may fail later if SELinux prevents access."
      fi
    fi
  fi

}

function lmktfy::build::has_docker() {
  which docker &> /dev/null
}

# Detect if a specific image exists
#
# $1 - image repo name
# #2 - image tag
function lmktfy::build::docker_image_exists() {
  [[ -n $1 && -n $2 ]] || {
    lmktfy::log::error "Internal error. Image not specified in docker_image_exists."
    exit 2
  }

  # We cannot just specify the IMAGE here as `docker images` doesn't behave as
  # expected.  See: https://github.com/docker/docker/issues/8048
  "${DOCKER[@]}" images | grep -Eq "^${1}\s+${2}\s+"
}

# Takes $1 and computes a short has for it. Useful for unique tag generation
function lmktfy::build::short_hash() {
  [[ $# -eq 1 ]] || {
    lmktfy::log::error "Internal error.  No data based to short_hash."
    exit 2
  }

  local short_hash
  if which md5 >/dev/null 2>&1; then
    short_hash=$(md5 -q -s "$1")
  else
    short_hash=$(echo -n "$1" | md5sum)
  fi
  echo ${short_hash:0:5}
}

# Pedantically kill, wait-on and remove a container. The -f -v options
# to rm don't actually seem to get the job done, so force kill the
# container, wait to ensure it's stopped, then try the remove. This is
# a workaround for bug https://github.com/docker/docker/issues/3968.
function lmktfy::build::destroy_container() {
  "${DOCKER[@]}" kill "$1" >/dev/null 2>&1 || true
  "${DOCKER[@]}" wait "$1" >/dev/null 2>&1 || true
  "${DOCKER[@]}" rm -f -v "$1" >/dev/null 2>&1 || true
}


# ---------------------------------------------------------------------------
# Building

function lmktfy::build::build_image_built() {
  lmktfy::build::docker_image_exists "${LMKTFY_BUILD_IMAGE_REPO}" "${LMKTFY_BUILD_IMAGE_TAG}"
}

function lmktfy::build::ensure_golang() {
  lmktfy::build::docker_image_exists golang "${LMKTFY_BUILD_GOLANG_VERSION}" || {
    [[ ${LMKTFY_SKIP_CONFIRMATIONS} =~ ^[yY]$ ]] || {
      echo "You don't have a local copy of the golang docker image. This image is 450MB."
      read -p "Download it now? [y/n] " -r
      echo
      [[ $REPLY =~ ^[yY]$ ]] || {
        echo "Aborting." >&2
        exit 1
      }
    }

    lmktfy::log::status "Pulling docker image: golang:${LMKTFY_BUILD_GOLANG_VERSION}"
    "${DOCKER[@]}" pull golang:${LMKTFY_BUILD_GOLANG_VERSION}
  }
}

# Set up the context directory for the lmktfy-build image and build it.
function lmktfy::build::build_image() {
  local -r build_context_dir="${LOCAL_OUTPUT_IMAGE_STAGING}/${LMKTFY_BUILD_IMAGE}"
  local -r source=(
    api
    build
    cmd
    docs/getting-started-guides
    examples
    Godeps/_workspace/src
    Godeps/Godeps.json
    hack
    LICENSE
    pkg
    plugin
    README.md
    test
    third_party
  )

  lmktfy::build::build_image_cross

  mkdir -p "${build_context_dir}"
  tar czf "${build_context_dir}/lmktfy-source.tar.gz" "${source[@]}"

  lmktfy::version::get_version_vars
  lmktfy::version::save_version_vars "${build_context_dir}/lmktfy-version-defs"

  cp build/build-image/Dockerfile ${build_context_dir}/Dockerfile
  lmktfy::build::docker_build "${LMKTFY_BUILD_IMAGE}" "${build_context_dir}"
}

# Build the lmktfy golang cross base image.
function lmktfy::build::build_image_cross() {
  lmktfy::build::ensure_golang

  local -r build_context_dir="${LOCAL_OUTPUT_ROOT}/images/${LMKTFY_BUILD_IMAGE}/cross"
  mkdir -p "${build_context_dir}"
  cp build/build-image/cross/Dockerfile ${build_context_dir}/Dockerfile
  lmktfy::build::docker_build "${LMKTFY_BUILD_IMAGE_CROSS}" "${build_context_dir}"
}

# Build a docker image from a Dockerfile.
# $1 is the name of the image to build
# $2 is the location of the "context" directory, with the Dockerfile at the root.
function lmktfy::build::docker_build() {
  local -r image=$1
  local -r context_dir=$2
  local -ra build_cmd=("${DOCKER[@]}" build -t "${image}" "${context_dir}")

  lmktfy::log::status "Building Docker image ${image}."
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

function lmktfy::build::clean_image() {
  local -r image=$1

  lmktfy::log::status "Deleting docker image ${image}"
  "${DOCKER[@]}" rmi ${image} 2> /dev/null || true
}

function lmktfy::build::clean_images() {
  lmktfy::build::has_docker || return 0

  lmktfy::build::clean_image "${LMKTFY_BUILD_IMAGE}"

  lmktfy::log::status "Cleaning all other untagged docker images"
  "${DOCKER[@]}" rmi $("${DOCKER[@]}" images -q --filter 'dangling=true') 2> /dev/null || true
}

function lmktfy::build::ensure_data_container() {
  if ! "${DOCKER[@]}" inspect "${LMKTFY_BUILD_DATA_CONTAINER_NAME}" >/dev/null 2>&1; then
    lmktfy::log::status "Creating data container"
    local -ra docker_cmd=(
      "${DOCKER[@]}" run
      "${DOCKER_DATA_MOUNT_ARGS[@]}"
      --name "${LMKTFY_BUILD_DATA_CONTAINER_NAME}"
      "${LMKTFY_BUILD_IMAGE}"
      true
    )
    "${docker_cmd[@]}"
  fi
}

# Run a command in the lmktfy-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
function lmktfy::build::run_build_command() {
  lmktfy::log::status "Running build command...."
  [[ $# != 0 ]] || { echo "Invalid input." >&2; return 4; }

  lmktfy::build::ensure_data_container
  lmktfy::build::prepare_output

  local -a docker_run_opts=(
    "--name=${LMKTFY_BUILD_CONTAINER_NAME}"
     "${DOCKER_MOUNT_ARGS[@]}"
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
    "${DOCKER[@]}" run "${docker_run_opts[@]}" "${LMKTFY_BUILD_IMAGE}")

  # Clean up container from any previous run
  lmktfy::build::destroy_container "${LMKTFY_BUILD_CONTAINER_NAME}"
  "${docker_cmd[@]}" "$@"
  lmktfy::build::destroy_container "${LMKTFY_BUILD_CONTAINER_NAME}"
}

# Test if the output directory is remote (and can only be accessed through
# docker) or if it is "local" and we can access the output without going through
# docker.
function lmktfy::build::is_output_remote() {
  rm -f "${LOCAL_OUTPUT_SUBPATH}/test_for_remote"
  lmktfy::build::run_build_command touch "${REMOTE_OUTPUT_BINPATH}/test_for_remote"

  [[ ! -e "${LOCAL_OUTPUT_BINPATH}/test_for_remote" ]]
}

# If the Docker server is remote, copy the results back out.
function lmktfy::build::copy_output() {
  if lmktfy::build::is_output_remote; then
    # At time of this code, docker cp does not work when copying from a volume.
    # As a workaround, the binaries are first copied to a local filesystem,
    # /tmp, then docker cp'd to the local binaries output directory.
    # The fix for the volume bug has been accepted and once it's widely
    # deployed the code below should be simplified to a simple docker cp
    # Bug: https://github.com/docker/docker/pull/8509
    local -a docker_run_opts=(
      "--name=${LMKTFY_BUILD_CONTAINER_NAME}"
       "${DOCKER_MOUNT_ARGS[@]}"
       -d
      )

    local -ra docker_cmd=(
      "${DOCKER[@]}" run "${docker_run_opts[@]}" "${LMKTFY_BUILD_IMAGE}"
    )

    lmktfy::log::status "Syncing back _output/dockerized/bin directory from remote Docker"
    rm -rf "${LOCAL_OUTPUT_BINPATH}"
    mkdir -p "${LOCAL_OUTPUT_BINPATH}"

    lmktfy::build::destroy_container "${LMKTFY_BUILD_CONTAINER_NAME}"
    "${docker_cmd[@]}" bash -c "cp -r ${REMOTE_OUTPUT_BINPATH} /tmp/bin;touch /tmp/finished;rm /tmp/bin/test_for_remote;/bin/sleep 600" > /dev/null 2>&1

    # Wait until binaries have finished coppying
    count=0
    while true;do
      if docker "${DOCKER_OPTS}" cp "${LMKTFY_BUILD_CONTAINER_NAME}:/tmp/finished" "${LOCAL_OUTPUT_BINPATH}" > /dev/null 2>&1;then
        docker "${DOCKER_OPTS}" cp "${LMKTFY_BUILD_CONTAINER_NAME}:/tmp/bin" "${LOCAL_OUTPUT_SUBPATH}"
        break;
      fi

      let count=count+1
      if [[ $count -eq 60 ]]; then
        # break after 5m
        lmktfy::log::error "Timed out waiting for binaries..."
        break
      fi
      sleep 5
    done

    "${DOCKER[@]}" rm -f -v "${LMKTFY_BUILD_CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    lmktfy::log::status "Output directory is local.  No need to copy results out."
  fi
}

# ---------------------------------------------------------------------------
# Build final release artifacts
function lmktfy::release::clean_cruft() {
  # Clean out cruft
  find ${RELEASE_STAGE} -name '*~' -exec rm {} \;
  find ${RELEASE_STAGE} -name '#*#' -exec rm {} \;
  find ${RELEASE_STAGE} -name '.DS*' -exec rm {} \;
}

function lmktfy::release::package_tarballs() {
  # Clean out any old releases
  rm -rf "${RELEASE_DIR}"
  mkdir -p "${RELEASE_DIR}"
  lmktfy::release::package_client_tarballs
  lmktfy::release::package_server_tarballs
  lmktfy::release::package_salt_tarball
  lmktfy::release::package_test_tarball
  lmktfy::release::package_full_tarball
}

# Package up all of the cross compiled clients.  Over time this should grow into
# a full SDK
function lmktfy::release::package_client_tarballs() {
   # Find all of the built client binaries
  local platform platforms
  platforms=($(cd "${LOCAL_OUTPUT_BINPATH}" ; echo */*))
  for platform in "${platforms[@]}" ; do
    local platform_tag=${platform/\//-} # Replace a "/" for a "-"
    lmktfy::log::status "Building tarball: client $platform_tag"

    local release_stage="${RELEASE_STAGE}/client/${platform_tag}/lmktfy"
    rm -rf "${release_stage}"
    mkdir -p "${release_stage}/client/bin"

    local client_bins=("${LMKTFY_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      client_bins=("${LMKTFY_CLIENT_BINARIES_WIN[@]}")
    fi

    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # LMKTFY_CLIENT_BINARIES array.
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/client/bin/"

    lmktfy::release::clean_cruft

    local package_name="${RELEASE_DIR}/lmktfy-client-${platform_tag}.tar.gz"
    lmktfy::release::create_tarball "${package_name}" "${release_stage}/.."
  done
}

# Package up all of the server binaries
function lmktfy::release::package_server_tarballs() {
  local platform
  for platform in "${LMKTFY_SERVER_PLATFORMS[@]}" ; do
    local platform_tag=${platform/\//-} # Replace a "/" for a "-"
    lmktfy::log::status "Building tarball: server $platform_tag"

    local release_stage="${RELEASE_STAGE}/server/${platform_tag}/lmktfy"
    rm -rf "${release_stage}"
    mkdir -p "${release_stage}/server/bin"

    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # LMKTFY_SERVER_BINARIES array.
    cp "${LMKTFY_SERVER_BINARIES[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/server/bin/"

    # Include the client binaries here too as they are useful debugging tools.
    local client_bins=("${LMKTFY_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      client_bins=("${LMKTFY_CLIENT_BINARIES_WIN[@]}")
    fi
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/server/bin/"

    lmktfy::release::clean_cruft

    local package_name="${RELEASE_DIR}/lmktfy-server-${platform_tag}.tar.gz"
    lmktfy::release::create_tarball "${package_name}" "${release_stage}/.."
  done
}

# Package up the salt configuration tree.  This is an optional helper to getting
# a cluster up and running.
function lmktfy::release::package_salt_tarball() {
  lmktfy::log::status "Building tarball: salt"

  local release_stage="${RELEASE_STAGE}/salt/lmktfy"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  cp -R "${LMKTFY_ROOT}/cluster/saltbase" "${release_stage}/"

  # TODO(#3579): This is a temporary hack. It gathers up the yaml,
  # yaml.in files in cluster/addons (minus any demos) and overlays
  # them into lmktfy-addons, where we expect them. (This pipeline is a
  # fancy copy, stripping anything but the files we don't want.)
  local objects
  objects=$(cd "${LMKTFY_ROOT}/cluster/addons" && find . -name \*.yaml -or -name \*.yaml.in | grep -v demo)
  tar c -C "${LMKTFY_ROOT}/cluster/addons" ${objects} | tar x -C "${release_stage}/saltbase/salt/lmktfy-addons"

  lmktfy::release::clean_cruft

  local package_name="${RELEASE_DIR}/lmktfy-salt.tar.gz"
  lmktfy::release::create_tarball "${package_name}" "${release_stage}/.."
}

# This is the stuff you need to run tests from the binary distribution.
function lmktfy::release::package_test_tarball() {
  lmktfy::log::status "Building tarball: test"

  local release_stage="${RELEASE_STAGE}/test/lmktfy"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  local platform
  for platform in "${LMKTFY_CLIENT_PLATFORMS[@]}"; do
    local test_bins=("${LMKTFY_TEST_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      test_bins=("${LMKTFY_TEST_BINARIES_WIN[@]}")
    fi
    mkdir -p "${release_stage}/platforms/${platform}"
    cp "${test_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/platforms/${platform}"
  done

  tar c ${LMKTFY_TEST_PORTABLE[@]} | tar x -C ${release_stage}

  lmktfy::release::clean_cruft

  local package_name="${RELEASE_DIR}/lmktfy-test.tar.gz"
  lmktfy::release::create_tarball "${package_name}" "${release_stage}/.."
}

# This is all the stuff you need to run/install lmktfy.  This includes:
#   - precompiled binaries for client
#   - Cluster spin up/down scripts and configs for various cloud providers
#   - tarballs for server binary and salt configs that are ready to be uploaded
#     to master by whatever means appropriate.
function lmktfy::release::package_full_tarball() {
  lmktfy::log::status "Building tarball: full"

  local release_stage="${RELEASE_STAGE}/full/lmktfy"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  # Copy all of the client binaries in here, but not test or server binaries.
  # The server binaries are included with the server binary tarball.
  local platform
  for platform in "${LMKTFY_CLIENT_PLATFORMS[@]}"; do
    local client_bins=("${LMKTFY_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      client_bins=("${LMKTFY_CLIENT_BINARIES_WIN[@]}")
    fi
    mkdir -p "${release_stage}/platforms/${platform}"
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/platforms/${platform}"
  done

  # We want everything in /cluster except saltbase.  That is only needed on the
  # server.
  cp -R "${LMKTFY_ROOT}/cluster" "${release_stage}/"
  rm -rf "${release_stage}/cluster/saltbase"

  mkdir -p "${release_stage}/server"
  cp "${RELEASE_DIR}/lmktfy-salt.tar.gz" "${release_stage}/server/"
  cp "${RELEASE_DIR}"/lmktfy-server-*.tar.gz "${release_stage}/server/"

  mkdir -p "${release_stage}/third_party"
  cp -R "${LMKTFY_ROOT}/third_party/htpasswd" "${release_stage}/third_party/htpasswd"

  cp -R "${LMKTFY_ROOT}/examples" "${release_stage}/"
  cp "${LMKTFY_ROOT}/README.md" "${release_stage}/"
  cp "${LMKTFY_ROOT}/LICENSE" "${release_stage}/"
  cp "${LMKTFY_ROOT}/Vagrantfile" "${release_stage}/"

  lmktfy::release::clean_cruft

  local package_name="${RELEASE_DIR}/lmktfy.tar.gz"
  lmktfy::release::create_tarball "${package_name}" "${release_stage}/.."
}

# Build a release tarball.  $1 is the output tar name.  $2 is the base directory
# of the files to be packaged.  This assumes that ${2}/lmktfy is what is
# being packaged.
function lmktfy::release::create_tarball() {
  local tarfile=$1
  local stagingdir=$2

  # Find gnu tar if it is available
  local tar=tar
  if which gtar &>/dev/null; then
      tar=gtar
  else
      if which gnutar &>/dev/null; then
	  tar=gnutar
      fi
  fi

  local tar_cmd=("$tar" "czf" "${tarfile}" "-C" "${stagingdir}" "lmktfy")
  if "$tar" --version | grep -q GNU; then
    tar_cmd=("${tar_cmd[@]}" "--owner=0" "--group=0")
  else
    echo "  !!! GNU tar not available.  User names will be embedded in output and"
    echo "      release tars are not official. Build on Linux or install GNU tar"
    echo "      on Mac OS X (brew install gnu-tar)"
  fi

  "${tar_cmd[@]}"
}

# ---------------------------------------------------------------------------
# GCS Release

function lmktfy::release::gcs::release() {
  [[ ${LMKTFY_GCS_UPLOAD_RELEASE} =~ ^[yY]$ ]] || return 0

  lmktfy::release::gcs::verify_prereqs
  lmktfy::release::gcs::ensure_release_bucket
  lmktfy::release::gcs::copy_release_artifacts
}

# Verify things are set up for uploading to GCS
function lmktfy::release::gcs::verify_prereqs() {
  if [[ -z "$(which gsutil)" || -z "$(which gcloud)" ]]; then
    echo "Releasing LMKTFY requires gsutil and gcloud.  Please download,"
    echo "install and authorize through the Google Cloud SDK: "
    echo
    echo "  https://developers.google.com/cloud/sdk/"
    return 1
  fi

  if [[ -z "${GCLOUD_ACCOUNT-}" ]]; then
    GCLOUD_ACCOUNT=$(gcloud auth list 2>/dev/null | awk '/(active)/ { print $2 }')
  fi
  if [[ -z "${GCLOUD_ACCOUNT-}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud auth login"
    return 1
  fi

  if [[ -z "${GCLOUD_PROJECT-}" ]]; then
    GCLOUD_PROJECT=$(gcloud config list project | awk '{project = $3} END {print project}')
  fi
  if [[ -z "${GCLOUD_PROJECT-}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud config set project <project id>"
    return 1
  fi
}

# Create a unique bucket name for releasing LMKTFY and make sure it exists.
function lmktfy::release::gcs::ensure_release_bucket() {
  local project_hash
  project_hash=$(lmktfy::build::short_hash "$GCLOUD_PROJECT")
  LMKTFY_GCS_RELEASE_BUCKET=${LMKTFY_GCS_RELEASE_BUCKET-lmktfy-releases-${project_hash}}

  if ! gsutil ls "gs://${LMKTFY_GCS_RELEASE_BUCKET}" >/dev/null 2>&1 ; then
    echo "Creating Google Cloud Storage bucket: $LMKTFY_GCS_RELEASE_BUCKET"
    gsutil mb -p "${GCLOUD_PROJECT}" "gs://${LMKTFY_GCS_RELEASE_BUCKET}"
  fi
}

function lmktfy::release::gcs::copy_release_artifacts() {
  # TODO: This isn't atomic.  There will be points in time where there will be
  # no active release.  Also, if something fails, the release could be half-
  # copied.  The real way to do this would perhaps to have some sort of release
  # version so that we are never overwriting a destination.
  local -r gcs_destination="gs://${LMKTFY_GCS_RELEASE_BUCKET}/${LMKTFY_GCS_RELEASE_PREFIX}"
  local gcs_options=()

  if [[ ${LMKTFY_GCS_NO_CACHING} =~ ^[yY]$ ]]; then
    gcs_options=("-h" "Cache-Control:private, max-age=0")
  fi

  lmktfy::log::status "Copying release artifacts to ${gcs_destination}"

  # First delete all objects at the destination
  if gsutil ls "${gcs_destination}" >/dev/null 2>&1; then
    lmktfy::log::error "${gcs_destination} not empty."
    read -p "Delete everything under ${gcs_destination}? [y/n] " -r || {
      echo "EOF on prompt.  Skipping upload"
      return
    }
    [[ $REPLY =~ ^[yY]$ ]] || {
      echo "Skipping upload"
      return
    }
    gsutil -m rm -f -R "${gcs_destination}"
  fi

  # Now upload everything in release directory
  gsutil -m "${gcs_options[@]+${gcs_options[@]}}" cp -r "${RELEASE_DIR}"/* "${gcs_destination}"

  # Having the "template" scripts from the GCE cluster deploy hosted with the
  # release is useful for GKE.  Copy everything from that directory up also.
  gsutil -m "${gcs_options[@]+${gcs_options[@]}}" cp \
    "${RELEASE_STAGE}/full/lmktfy/cluster/gce/configure-vm.sh" \
    "${gcs_destination}extra/gce/"

  # Upload the "naked" binaries to GCS.  This is useful for install scripts that
  # download the binaries directly and don't need tars.
  local platform platforms
  platforms=($(cd "${RELEASE_STAGE}/client" ; echo *))
  for platform in "${platforms[@]}"; do
    local src="${RELEASE_STAGE}/client/${platform}/lmktfy/client/bin/*"
    local dst="${gcs_destination}bin/${platform/-//}/"
    # We assume here the "server package" is a superset of the "client package"
    if [[ -d "${RELEASE_STAGE}/server/${platform}" ]]; then
      src="${RELEASE_STAGE}/server/${platform}/lmktfy/server/bin/*"
    fi
    gsutil -m "${gcs_options[@]+${gcs_options[@]}}" cp \
      "$src" "$dst"
  done

  # TODO(jbeda): Generate an HTML page with links for this release so it is easy
  # to see it.  For extra credit, generate a dynamic page that builds up the
  # release list using the GCS JSON API.  Use Angular and Bootstrap for extra
  # extra credit.

  if [[ ${LMKTFY_GCS_MAKE_PUBLIC} =~ ^[yY]$ ]]; then
    lmktfy::log::status "Marking all uploaded objects public"
    gsutil acl ch -R -g all:R "${gcs_destination}" >/dev/null 2>&1
  fi

  gsutil ls -lhr "${gcs_destination}"
}

function lmktfy::release::gcs::publish_latest() {
  local latest_file_dst="gs://${LMKTFY_GCS_RELEASE_BUCKET}/${LMKTFY_GCS_LATEST_FILE}"

  mkdir -p "${RELEASE_STAGE}/upload"
  echo ${LMKTFY_GCS_LATEST_CONTENTS} > "${RELEASE_STAGE}/upload/latest"

  gsutil -m "${gcs_options[@]+${gcs_options[@]}}" cp \
    "${RELEASE_STAGE}/upload/latest" "${latest_file_dst}"

  if [[ ${LMKTFY_GCS_MAKE_PUBLIC} =~ ^[yY]$ ]]; then
    gsutil acl ch -R -g all:R "${latest_file_dst}" >/dev/null 2>&1
  fi

  lmktfy::log::status "gsutil cat ${latest_file_dst}:"
  gsutil cat ${latest_file_dst}
}

# Publish a new latest.txt, but only if the release we're dealing with
# is newer than the contents in GCS.
function lmktfy::release::gcs::publish_latest_official() {
  local -r new_version=${LMKTFY_GCS_LATEST_CONTENTS}
  local -r latest_file_dst="gs://${LMKTFY_GCS_RELEASE_BUCKET}/${LMKTFY_GCS_LATEST_FILE}"

  local -r version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  [[ ${new_version} =~ ${version_regex} ]] || {
    lmktfy::log::error "publish_latest_official passed bogus value: '${new_version}'"
    return 1
  }

  local -r version_major="${BASH_REMATCH[1]}"
  local -r version_minor="${BASH_REMATCH[2]}"
  local -r version_patch="${BASH_REMATCH[3]}"

  local gcs_version
  gcs_version=$(gsutil cat "${latest_file_dst}")

  [[ ${gcs_version} =~ ${version_regex} ]] || {
    lmktfy::log::error "${latest_file_dst} contains invalid release version, can't compare: '${gcs_version}'"
    return 1
  }

  local -r gcs_version_major="${BASH_REMATCH[1]}"
  local -r gcs_version_minor="${BASH_REMATCH[2]}"
  local -r gcs_version_patch="${BASH_REMATCH[3]}"

  local greater=true
  if [[ "${gcs_version_major}" -gt "${version_major}" ]]; then
    greater=false
  elif [[ "${gcs_version_major}" -lt "${version_major}" ]]; then
    : # fall out
  elif [[ "${gcs_version_minor}" -gt "${version_minor}" ]]; then
    greater=false
  elif [[ "${gcs_version_minor}" -lt "${version_minor}" ]]; then
    : # fall out
  elif [[ "${gcs_version_patch}" -ge "${version_patch}" ]]; then
    greater=false
  fi

  if [[ "${greater}" != "true" ]]; then
    lmktfy::log::status "${gcs_version} (latest on GCS) >= ${new_version} (just uploaded), not updating ${latest_file_dst}"
    return 0
  fi

  lmktfy::log::status "${new_version} (just uploaded) > ${gcs_version} (latest on GCS), updating ${latest_file_dst}"
  lmktfy::release::gcs::publish_latest
}
