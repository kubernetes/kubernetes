#! /bin/bash

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

# Common utilties, variables and checks for all build scripts.

cd $(dirname "${BASH_SOURCE}")/..
readonly KUBE_REPO_ROOT="${PWD}"

readonly KUBE_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
KUBE_BUILD_IMAGE=kube-build
if [ -n "${KUBE_GIT_BRANCH}" ]; then
  KUBE_BUILD_IMAGE="${KUBE_BUILD_IMAGE}:${KUBE_GIT_BRANCH}"
fi
readonly KUBE_BUILD_IMAGE
readonly KUBE_GO_PACKAGE="github.com/GoogleCloudPlatform/kubernetes"

# We set up a volume so that we have the same output directory from one run of
# the container to the next.
#
# Note that here "LOCAL" is local to the docker daemon.  In the boot2docker case
# this is still inside the VM.  We use the same directory in both cases though.
readonly LOCAL_OUTPUT_DIR="${KUBE_REPO_ROOT}/output/build"
readonly REMOTE_OUTPUT_DIR="/go/src/${KUBE_GO_PACKAGE}/output/build"
readonly DOCKER_CONTAINER_NAME=kube-build
readonly DOCKER_MOUNT="-v ${LOCAL_OUTPUT_DIR}:${REMOTE_OUTPUT_DIR}"

# Verify that the right utilitites and such are installed.
if [[ -z "$(which docker)" ]]; then
  echo "Can't find 'docker' in PATH, please fix and retry." >&2
  echo "See https://docs.docker.com/installation/#installation for installation instructions." >&2
  exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ -z "$(which boot2docker)" ]]; then
    echo "It looks like you are running on Mac OS X and boot2docker can't be found." >&2
    echo "See: https://docs.docker.com/installation/mac/" >&2
    exit 1
  fi
  if [[ $(boot2docker status) != "running" ]]; then
    echo "boot2docker VM isn't started.  Please run 'boot2docker start'" >&2
    exit 1
  fi
fi

if ! docker info > /dev/null 2>&1 ; then
  echo "Can't connect to 'docker' daemon.  please fix and retry." >&2
  echo >&2
  echo "Possible causes:" >&2
  echo "  - On Mac OS X, boot2docker VM isn't started" >&2
  echo "  - On Mac OS X, DOCKER_HOST env variable isn't set approriately" >&2
  echo "  - On Linux, user isn't in 'docker' group.  Add and relogin." >&2
  echo "  - On Linux, Docker daemon hasn't been started or has crashed" >&2
  exit 1
fi

# Set up the context directory for the kube-build image and build it.
function build-image() {
  local -r BUILD_CONTEXT_DIR=${KUBE_REPO_ROOT}/output/build-image
  local -r SOURCE="
    api
    build
    cmd
    hack
    pkg
    third_party
    LICENSE
  "
  local -r DOCKER_BUILD_CMD="docker build -t ${KUBE_BUILD_IMAGE} ${BUILD_CONTEXT_DIR}"

  echo "+++ Building Docker image ${KUBE_BUILD_IMAGE}.  First run can take minutes."

  mkdir -p ${BUILD_CONTEXT_DIR}
  tar czf ${BUILD_CONTEXT_DIR}/kube-source.tar.gz ${SOURCE}
  cp build/build-image/Dockerfile ${BUILD_CONTEXT_DIR}/Dockerfile

  set +e # We are handling the error here manually
  local -r DOCKER_OUTPUT="$(${DOCKER_BUILD_CMD} 2>&1)"
  if [ $? -ne 0 ]; then
    set -e
    echo "+++ Docker build command failed." >&2
    echo >&2
    echo "${DOCKER_OUTPUT}" >&2
    echo >&2
    echo "To retry manually, run:" >&2
    echo >&2
    echo "  ${DOCKER_BUILD_CMD}" >&2
    echo >&2
    return 1
  fi
  set -e

}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
function run-build-command() {
  [[ -n "$@" ]] || { echo "Invalid input." >&2; return 4; }

  local -r DOCKER="docker run --rm --name=${DOCKER_CONTAINER_NAME} -it ${DOCKER_MOUNT} ${KUBE_BUILD_IMAGE}"

  docker rm ${DOCKER_CONTAINER_NAME} >/dev/null 2>&1 || true

  ${DOCKER} "$@"

}

# If the Docker server is remote, copy the results back out.
function copy-output() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # When we are on the Mac with boot2docker Now we need to copy the results
    # back out.  Ideally we would leave the container around and use 'docker cp'
    # to copy the results out.  However, that doesn't work for mounted volumes
    # currently (https://github.com/dotcloud/docker/issues/1992).  And it is
    # just plain broken (https://github.com/dotcloud/docker/issues/6483).
    #
    # The easiest thing I (jbeda) could figure out was to launch another
    # container pointed at the same volume, tar the output directory and ship
    # that tar over stdou.
    local DOCKER="docker run -a stdout --rm --name=${DOCKER_CONTAINER_NAME} ${DOCKER_MOUNT} ${KUBE_BUILD_IMAGE}"

    # Kill any leftover container
    docker rm ${DOCKER_CONTAINER_NAME} >/dev/null 2>&1 || true

    echo "+++ Syncing back output directory from boot2docker VM"
    mkdir -p "${LOCAL_OUTPUT_DIR}"
    rm -rf "${LOCAL_OUTPUT_DIR}/*"
    ${DOCKER} sh -c "tar c -C ${REMOTE_OUTPUT_DIR} ."  \
      | tar xv -C "${LOCAL_OUTPUT_DIR}"

    # I (jbeda) also tried getting rsync working using 'docker run' as the
    # 'remote shell'.  This mostly worked but there was a hang when
    # closing/finishing things off. Ug.
    #
    # local DOCKER="docker run -i --rm --name=${DOCKER_CONTAINER_NAME} ${DOCKER_MOUNT} ${KUBE_BUILD_IMAGE}"
    # DOCKER+=" bash -c 'shift ; exec \"\$@\"' --"
    # rsync --blocking-io -av -e "${DOCKER}" foo:${REMOTE_OUTPUT_DIR}/ ${LOCAL_OUTPUT_DIR}
  fi
}


