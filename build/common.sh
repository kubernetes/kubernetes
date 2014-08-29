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

# We set up a volume so that we have the same _output directory from one run of
# the container to the next.
#
# Note that here "LOCAL" is local to the docker daemon.  In the boot2docker case
# this is still inside the VM.  We use the same directory in both cases though.
readonly LOCAL_OUTPUT_DIR="${KUBE_REPO_ROOT}/_output/build"
readonly REMOTE_OUTPUT_DIR="/go/src/${KUBE_GO_PACKAGE}/_output/build"
readonly DOCKER_CONTAINER_NAME=kube-build
readonly DOCKER_MOUNT="-v ${LOCAL_OUTPUT_DIR}:${REMOTE_OUTPUT_DIR}"

readonly KUBE_RUN_IMAGE_BASE="kubernetes"
readonly KUBE_RUN_BINARIES="
    apiserver
    controller-manager
    proxy
  "

# This is where the final release artifacts are created locally
readonly RELEASE_DIR="${KUBE_REPO_ROOT}/_output/release"

# ---------------------------------------------------------------------------
# Basic setup functions

# Verify that the right utilities and such are installed for building Kube.
function verify-prereqs() {
  if [[ -z "$(which docker)" ]]; then
    echo "Can't find 'docker' in PATH, please fix and retry." >&2
    echo "See https://docs.docker.com/installation/#installation for installation instructions." >&2
    return 1
  fi

  if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ -z "$(which boot2docker)" ]]; then
      echo "It looks like you are running on Mac OS X and boot2docker can't be found." >&2
      echo "See: https://docs.docker.com/installation/mac/" >&2
      return 1
    fi
    if [[ $(boot2docker status) != "running" ]]; then
      echo "boot2docker VM isn't started.  Please run 'boot2docker start'" >&2
      return 1
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
    return 1
  fi
}

# Verify things are set up for uploading to GCS
function verify-gcs-prereqs() {
  if [[ -z "$(which gsutil)" || -z "$(which gcloud)" ]]; then
    echo "Releasing Kubernetes requires gsutil and gcloud.  Please download,"
    echo "install and authorize through the Google Cloud SDK: "
    echo
    echo "  https://developers.google.com/cloud/sdk/"
    return 1
  fi

  FIND_ACCOUNT="gcloud auth list 2>/dev/null | grep '(active)' | awk '{ print \$2 }'"
  GCLOUD_ACCOUNT=${GCLOUD_ACCOUNT-$(eval ${FIND_ACCOUNT})}
  if [[ -z "${GCLOUD_ACCOUNT}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud auth login"
    return 1
  fi

  FIND_PROJECT="gcloud config list project | tail -n 1 | awk '{ print \$3 }'"
  GCLOUD_PROJECT=${GCLOUD_PROJECT-$(eval ${FIND_PROJECT})}
  if [[ -z "${GCLOUD_PROJECT}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud config set project <project id>"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Building

# Set up the context directory for the kube-build image and build it.
function build-image() {
  local -r BUILD_CONTEXT_DIR="${KUBE_REPO_ROOT}/_output/images/${KUBE_BUILD_IMAGE}"
  local -r SOURCE="
    api
    build
    cmd
    examples
    Godeps
    hack
    LICENSE
    README.md
    pkg
    plugin
    third_party
  "
  mkdir -p ${BUILD_CONTEXT_DIR}
  tar czf ${BUILD_CONTEXT_DIR}/kube-source.tar.gz ${SOURCE}
  cp build/build-image/Dockerfile ${BUILD_CONTEXT_DIR}/Dockerfile
  docker-build "${KUBE_BUILD_IMAGE}" "${BUILD_CONTEXT_DIR}"
}

# Builds the runtime image.  Assumes that the appropriate binaries are already
# built and in _output/build/.
function run-image() {
  local -r BUILD_CONTEXT_BASE="${KUBE_REPO_ROOT}/_output/images/${KUBE_RUN_IMAGE_BASE}"

  # First build the base image.  This one brings in all of the binaries.
  mkdir -p "${BUILD_CONTEXT_BASE}"
  tar czf ${BUILD_CONTEXT_BASE}/kube-bins.tar.gz \
    -C "_output/build/linux/amd64" \
    ${KUBE_RUN_BINARIES}
  cp -R build/run-images/base/* "${BUILD_CONTEXT_BASE}/"
  docker-build "${KUBE_RUN_IMAGE_BASE}" "${BUILD_CONTEXT_BASE}"

  for b in $KUBE_RUN_BINARIES ; do
    local SUB_CONTEXT_DIR="${BUILD_CONTEXT_BASE}-$b"
    mkdir -p "${SUB_CONTEXT_DIR}"
    cp -R build/run-images/$b/* "${SUB_CONTEXT_DIR}/"
    docker-build "${KUBE_RUN_IMAGE_BASE}-$b" "${SUB_CONTEXT_DIR}"
  done
}

# Build a docker image from a Dockerfile.
# $1 is the name of the image to build
# $2 is the location of the "context" directory, with the Dockerfile at the root.
function docker-build() {
  local -r IMAGE=$1
  local -r CONTEXT_DIR=$2
  local -r BUILD_CMD="docker build -t ${IMAGE} ${CONTEXT_DIR}"

  echo "+++ Building Docker image ${IMAGE}. This can take a while."
  set +e # We are handling the error here manually
  local -r DOCKER_OUTPUT="$(${BUILD_CMD} 2>&1)"
  if [ $? -ne 0 ]; then
    set -e
    echo "+++ Docker build command failed for ${IMAGE}" >&2
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
    # When we are on the Mac with boot2docker we need to copy the results back
    # out.  Ideally we would leave the container around and use 'docker cp' to
    # copy the results out.  However, that doesn't work for mounted volumes
    # currently (https://github.com/dotcloud/docker/issues/1992).  And it is
    # just plain broken (https://github.com/dotcloud/docker/issues/6483).
    #
    # The easiest thing I (jbeda) could figure out was to launch another
    # container pointed at the same volume, tar the output directory and ship
    # that tar over stdou.
    local DOCKER="docker run -a stdout --rm --name=${DOCKER_CONTAINER_NAME} ${DOCKER_MOUNT} ${KUBE_BUILD_IMAGE}"

    # Kill any leftover container
    docker rm ${DOCKER_CONTAINER_NAME} >/dev/null 2>&1 || true

    echo "+++ Syncing back _output directory from boot2docker VM"
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

# ---------------------------------------------------------------------------
# Release

# Create a unique bucket name for releasing Kube and make sure it exists.
function ensure-gcs-release-bucket() {
  if which md5 > /dev/null 2>&1; then
    HASH=$(md5 -q -s "$GCLOUD_PROJECT")
  else
    HASH=$(echo -n "$GCLOUD_PROJECT" | md5sum)
  fi
  HASH=${HASH:0:5}
  KUBE_RELEASE_BUCKET=${KUBE_RELEASE_BUCKET-kubernetes-releases-$HASH}
  KUBE_RELEASE_PREFIX=${KUBE_RELEASE_PREFIX-devel/}
  KUBE_DOCKER_REG_PREFIX=${KUBE_DOCKER_REG_PREFIX-docker-reg/}

  if ! gsutil ls gs://${KUBE_RELEASE_BUCKET} >/dev/null 2>&1 ; then
    echo "Creating Google Cloud Storage bucket: $RELEASE_BUCKET"
    gsutil mb gs://${KUBE_RELEASE_BUCKET}
  fi
}

function ensure-gcs-docker-registry() {
  local -r REG_CONTAINER_NAME="gcs-registry"

  local -r RUNNING=$(docker inspect ${REG_CONTAINER_NAME} 2>/dev/null \
    | build/json-extractor.py 0.State.Running 2>/dev/null)

  [[ "$RUNNING" != "true" ]] || return 0

  # Grovel around and find the OAuth token in the gcloud config
  local -r BOTO=~/.config/gcloud/legacy_credentials/${GCLOUD_ACCOUNT}/.boto
  local -r REFRESH_TOKEN=$(grep 'gs_oauth2_refresh_token =' $BOTO | awk '{ print $3 }')

  if [[ -z $REFRESH_TOKEN ]]; then
    echo "Couldn't find OAuth 2 refresh token in ${BOTO}" >&2
    return 1
  fi

  # If we have an old one sitting around, remove it
  docker rm ${REG_CONTAINER_NAME} >/dev/null 2>&1 || true

  echo "+++ Starting GCS backed Docker registry"
  local DOCKER="docker run -d --name=${REG_CONTAINER_NAME} "
  DOCKER+="-e GCS_BUCKET=${KUBE_RELEASE_BUCKET} "
  DOCKER+="-e STORAGE_PATH=${KUBE_DOCKER_REG_PREFIX} "
  DOCKER+="-e GCP_OAUTH2_REFRESH_TOKEN=${REFRESH_TOKEN} "
  DOCKER+="-p 127.0.0.1:5000:5000 "
  DOCKER+="google/docker-registry"

  ${DOCKER}

  # Give it time to spin up before we start throwing stuff at it
  sleep 5
}

function push-images-to-gcs() {
  ensure-gcs-docker-registry

  # Tag each of our run binaries with the right registry and push
  for b in ${KUBE_RUN_BINARIES} ; do
    echo "+++ Tagging and pushing ${KUBE_RUN_IMAGE_BASE}-$b to GCS bucket ${KUBE_RELEASE_BUCKET}"
    docker tag "${KUBE_RUN_IMAGE_BASE}-$b" "localhost:5000/${KUBE_RUN_IMAGE_BASE}-$b"
    docker push "localhost:5000/${KUBE_RUN_IMAGE_BASE}-$b"
    docker rmi "localhost:5000/${KUBE_RUN_IMAGE_BASE}-$b"
  done
}

# Package up all of the cross compiled clients
function package-tarballs() {
  mkdir -p "${RELEASE_DIR}"

  # Find all of the built kubecfg binaries
  for platform in _output/build/*/* ; do
    echo $platform
    local PLATFORM_TAG=$(echo $platform | awk -F / '{ printf "%s-%s", $3, $4 }')
    echo "+++ Building client package for $PLATFORM_TAG"

    local CLIENT_RELEASE_STAGE="${KUBE_REPO_ROOT}/_output/release-stage/${PLATFORM_TAG}/kubernetes"
    mkdir -p "${CLIENT_RELEASE_STAGE}"
    mkdir -p "${CLIENT_RELEASE_STAGE}/bin"

    cp $platform/* "${CLIENT_RELEASE_STAGE}/bin"

    local CLIENT_PACKAGE_NAME="${RELEASE_DIR}/kubernetes-${PLATFORM_TAG}.tar.gz"
    tar czf ${CLIENT_PACKAGE_NAME} \
      -C "${CLIENT_RELEASE_STAGE}/.." \
      .
  done
}

function copy-release-to-gcs() {
  # TODO: This isn't atomic.  There will be points in time where there will be
  # no active release.  Also, if something fails, the release could be half-
  # copied.  The real way to do this would perhaps to have some sort of release
  # version so that we are never overwriting a destination.
  local -r GCS_DESTINATION="gs://${KUBE_RELEASE_BUCKET}/${KUBE_RELEASE_PREFIX}"

  echo "+++ Copying client tarballs to ${GCS_DESTINATION}"

  # First delete all objects at the destination
  gsutil -q rm -f -R "${GCS_DESTINATION}" >/dev/null 2>&1 || true

  # Now upload everything in release directory
  gsutil -m cp -r "${RELEASE_DIR}" "${GCS_DESTINATION}" >/dev/null 2>&1
}
