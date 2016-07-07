#!/bin/bash

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

# Run e2e tests using environment variables exported in e2e.sh.

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

: ${KUBE_GCS_RELEASE_BUCKET:="kubernetes-release"}
: ${KUBE_GCS_DEV_RELEASE_BUCKET:="kubernetes-release-dev"}

function running_in_docker() {
    grep -q docker /proc/self/cgroup
}

function fetch_output_tars() {
    echo "Using binaries from _output."
    cp _output/release-tars/kubernetes*.tar.gz .
    unpack_binaries
}

function fetch_server_version_tars() {
    local -r server_version="$(gcloud ${CMD_GROUP:-} container get-server-config --project=${PROJECT} --zone=${ZONE}  --format='value(defaultClusterVersion)')"
    # Use latest build of the server version's branch for test files.
    fetch_published_version_tars "ci/latest-${server_version:0:3}"
    # Unset cluster api version; we want to use server default for the cluster
    # version.
    unset CLUSTER_API_VERSION
}

# Use a published version like "ci/latest" (default), "release/latest",
# "release/latest-1", or "release/stable"
function fetch_published_version_tars() {
    local -r published_version="${1}"
    IFS='/' read -a varr <<< "${published_version}"
    path="${varr[0]}"
    if [[ "${path}" == "release" ]]; then
      local -r bucket="${KUBE_GCS_RELEASE_BUCKET}"
    else
      local -r bucket="${KUBE_GCS_DEV_RELEASE_BUCKET}"
    fi
    build_version=$(gsutil cat "gs://${bucket}/${published_version}.txt")
    echo "Using published version $bucket/$build_version (from ${published_version})"
    fetch_tars_from_gcs "gs://${bucket}/${path}" "${build_version}"
    unpack_binaries
    # Set CLUSTER_API_VERSION for GKE CI
    export CLUSTER_API_VERSION=$(echo ${build_version} | cut -c 2-)
}

# TODO(ihmccreery) I'm not sure if this is necesssary, with the workspace check
# below.
function clean_binaries() {
    echo "Cleaning up binaries."
    rm -rf kubernetes*
}

function fetch_tars_from_gcs() {
    local -r gspath="${1}"
    local -r build_version="${2}"
    echo "Pulling binaries from GCS; using server version ${gspath}/${build_version}."
    gsutil -mq cp "${gspath}/${build_version}/kubernetes.tar.gz" "${gspath}/${build_version}/kubernetes-test.tar.gz" .
}

function unpack_binaries() {
    md5sum kubernetes*.tar.gz
    tar -xzf kubernetes.tar.gz
    tar -xzf kubernetes-test.tar.gz
}

# Get the latest GCI image in a family.
function get_latest_gci_image() {
    local -r image_project="$1"
    local -r image_family="$2"
    echo "$(gcloud compute images describe-from-family ${image_family} --project=${image_project} --format='value(name)')"
}

function get_latest_docker_release() {
  # Typical Docker release versions are like v1.11.2-rc1, v1.11.2, and etc.
  local -r version_re='.*\"tag_name\":[[:space:]]+\"v([0-9\.r|c-]+)\",.*'
  local -r releases="$(curl -fsSL --retry 3 https://api.github.com/repos/docker/docker/releases)"
  # The GitHub API returns releases in descending order of creation time so the
  # first one is always the latest.
  # TODO: if we can install `jq` on the Jenkins nodes, we won't have to craft
  # regular expressions here.
  while read -r line; do
    if [[ "${line}" =~ ${version_re} ]]; then
      echo "${BASH_REMATCH[1]}"
      return
    fi
  done <<< "${releases}"
  echo "Failed to determine the latest Docker release."
  exit 1
}

function install_google_cloud_sdk_tarball() {
    local -r tarball=$1
    local -r install_dir=$2
    mkdir -p "${install_dir}"
    tar xzf "${tarball}" -C "${install_dir}"

    export CLOUDSDK_CORE_DISABLE_PROMPTS=1
    "${install_dir}/google-cloud-sdk/install.sh" --disable-installation-options --bash-completion=false --path-update=false --usage-reporting=false
    export PATH=${install_dir}/google-cloud-sdk/bin:${PATH}
}

# Figures out the builtin k8s version of a GCI image.
function get_gci_k8s_version() {
    local -r image_description=$(gcloud compute images describe ${KUBE_GCE_MASTER_IMAGE} --project=${KUBE_GCE_MASTER_PROJECT})
    # Staged GCI images all include versions in their image descriptions so we
    # extract builtin Kubernetes version from them.
    local -r k8s_version_re='.*Kubernetes: ([0-9a-z.-]+),.*'
    if [[ ${image_description} =~ ${k8s_version_re} ]]; then
        local -r gci_k8s_version="v${BASH_REMATCH[1]}"
    else
        echo "Failed to determine builtin k8s version for image ${image_name}: ${image_description}"
        exit 1
    fi
    echo "${gci_k8s_version}"
}

# GCI specific settings.
# Assumes: JENKINS_GCI_IMAGE_FAMILY
function setup_gci_vars() {
    local -r gci_staging_project=container-vm-image-staging
    local -r image_name="$(gcloud compute images describe-from-family ${JENKINS_GCI_IMAGE_FAMILY} --project=${gci_staging_project} --format='value(name)')"

    export KUBE_GCE_MASTER_PROJECT="${gci_staging_project}"
    export KUBE_GCE_MASTER_IMAGE="${image_name}"
    export KUBE_MASTER_OS_DISTRIBUTION="gci"
    if [[ "${JENKINS_GCI_IMAGE_FAMILY}" == "gci-canary-test" ]]; then
        # The family "gci-canary-test" is reserved for a special type of GCI images
        # that are used to continuously validate Docker releases.
        export KUBE_GCI_DOCKER_VERSION="$(get_latest_docker_release)"
    fi
}

### Pre Set Up ###
if running_in_docker; then
    curl -fsSL --retry 3 -o "${WORKSPACE}/google-cloud-sdk.tar.gz" 'https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz'
    install_google_cloud_sdk_tarball "${WORKSPACE}/google-cloud-sdk.tar.gz" /
fi

# Install gcloud from a custom path if provided. Used to test GKE with gcloud
# at HEAD, release candidate.
# TODO: figure out how to avoid installing the cloud sdk twice if run inside Docker.
if [[ -n "${CLOUDSDK_BUCKET:-}" ]]; then
    # Retry the download a few times to mitigate transient server errors and
    # race conditions where the bucket contents change under us as we download.
    for n in $(seq 3); do
        gsutil -mq cp -r "${CLOUDSDK_BUCKET}" ~ && break || sleep 1
        # Delete any temporary files from the download so that we start from
        # scratch when we retry.
        rm -rf ~/.gsutil
    done
    rm -rf ~/repo ~/cloudsdk
    mv ~/$(basename "${CLOUDSDK_BUCKET}") ~/repo
    export CLOUDSDK_COMPONENT_MANAGER_SNAPSHOT_URL=file://${HOME}/repo/components-2.json
    install_google_cloud_sdk_tarball ~/repo/google-cloud-sdk.tar.gz ~/cloudsdk
    # TODO: is this necessary? this won't work inside Docker currently.
    export CLOUDSDK_CONFIG=/var/lib/jenkins/.config/gcloud
fi

# We get the image project and name for GCI dynamically.
if [[ -n "${JENKINS_GCI_IMAGE_FAMILY:-}" ]]; then
  GCI_STAGING_PROJECT=container-vm-image-staging
  export KUBE_GCE_MASTER_PROJECT="${GCI_STAGING_PROJECT}"
  export KUBE_GCE_MASTER_IMAGE="$(get_latest_gci_image "${GCI_STAGING_PROJECT}" "${JENKINS_GCI_IMAGE_FAMILY}")"
  export KUBE_MASTER_OS_DISTRIBUTION="gci"
  if [[ "${JENKINS_GCI_IMAGE_FAMILY}" == "gci-preview-test" ]]; then
    # The family "gci-preview-test" is reserved for a special type of GCI images
    # that are used to continuously validate Docker releases.
    export KUBE_GCI_DOCKER_VERSION="$(get_latest_docker_release)"
  fi
fi

echo "--------------------------------------------------------------------------------"
echo "Test Environment:"
printenv | sort
echo "--------------------------------------------------------------------------------"

# Set this var instead of exiting-- we must do the cluster teardown step. We'll
# return this at the very end.
EXIT_CODE=0

# We get the Kubernetes tarballs unless we are going to use old ones
if [[ "${JENKINS_USE_EXISTING_BINARIES:-}" =~ ^[yY]$ ]]; then
    echo "Using existing binaries; not cleaning, fetching, or unpacking new ones."
elif [[ "${KUBE_RUN_FROM_OUTPUT:-}" =~ ^[yY]$ ]]; then
    # TODO(spxtr) This should probably be JENKINS_USE_BINARIES_FROM_OUTPUT or
    # something, rather than being prepended with KUBE, since it's sort of a
    # meta-thing.
    clean_binaries
    fetch_output_tars
elif [[ "${JENKINS_USE_SERVER_VERSION:-}" =~ ^[yY]$ ]]; then
    # This is for test, staging, and prod jobs on GKE, where we want to
    # test what's running in GKE by default rather than some CI build.
    clean_binaries
    fetch_server_version_tars
else
    # use JENKINS_PUBLISHED_VERSION, default to 'ci/latest', since that's
    # usually what we're testing.
    clean_binaries
    fetch_published_version_tars "${JENKINS_PUBLISHED_VERSION:-ci/latest}"
fi

# Copy GCE keys so we don't keep cycling them.
# To set this up, you must know the <project>, <zone>, and <instance>
# on which your jenkins jobs are running. Then do:
#
# # SSH from your computer into the instance.
# $ gcloud compute ssh --project="<prj>" ssh --zone="<zone>" <instance>
#
# # Generate a key by ssh'ing from the instance into itself, then exit.
# $ gcloud compute ssh --project="<prj>" ssh --zone="<zone>" <instance>
# $ ^D
#
# # Copy the keys to the desired location (e.g. /var/lib/jenkins/gce_keys/).
# $ sudo mkdir -p /var/lib/jenkins/gce_keys/
# $ sudo cp ~/.ssh/google_compute_engine /var/lib/jenkins/gce_keys/
# $ sudo cp ~/.ssh/google_compute_engine.pub /var/lib/jenkins/gce_keys/
#
# # Move the permissions for the keys to Jenkins.
# $ sudo chown -R jenkins /var/lib/jenkins/gce_keys/
# $ sudo chgrp -R jenkins /var/lib/jenkins/gce_keys/
case "${KUBERNETES_PROVIDER}" in
    gce|gke|kubemark)
        if ! running_in_docker; then
            mkdir -p ${WORKSPACE}/.ssh/
            cp /var/lib/jenkins/gce_keys/google_compute_engine ${WORKSPACE}/.ssh/
            cp /var/lib/jenkins/gce_keys/google_compute_engine.pub ${WORKSPACE}/.ssh/
        fi
        echo 'Checking existence of private ssh key'
        gce_key="${WORKSPACE}/.ssh/google_compute_engine"
        if [[ ! -f "${gce_key}" || ! -f "${gce_key}.pub" ]]; then
            echo 'google_compute_engine ssh key missing!'
            exit 1
        fi
        echo "Checking presence of public key in ${PROJECT}"
        if ! gcloud compute --project="${PROJECT}" project-info describe |
             grep "$(cat "${gce_key}.pub")" >/dev/null; then
            echo 'Uploading public ssh key to project metadata...'
            gcloud compute --project="${PROJECT}" config-ssh
        fi
        ;;
    default)
        echo "Not copying ssh keys for ${KUBERNETES_PROVIDER}"
        ;;
esac


# Allow download & unpack of alternate version of tests, for cross-version & upgrade testing.
#
# JENKINS_PUBLISHED_SKEW_VERSION downloads an alternate version of Kubernetes
# for testing, moving the old one to kubernetes_old.
#
# E2E_UPGRADE_TEST=true triggers a run of the e2e tests, to do something like
# upgrade the cluster, before the main test run.  It uses
# GINKGO_UPGRADE_TESTS_ARGS for the test run.
#
# JENKINS_USE_SKEW_TESTS=true will run tests from the skewed version rather
# than the original version.
if [[ -n "${JENKINS_PUBLISHED_SKEW_VERSION:-}" ]]; then
  mv kubernetes kubernetes_orig
  fetch_published_version_tars "${JENKINS_PUBLISHED_SKEW_VERSION}"
  mv kubernetes kubernetes_skew
  mv kubernetes_orig kubernetes
  if [[ "${JENKINS_USE_SKEW_TESTS:-}" != "true" ]]; then
    # Append kubectl-path of skewed kubectl to test args, since we always
    #   # want that to use the skewed kubectl version:
    #     #
    #       # - for upgrade jobs, we want kubectl to be at the same version as
    #       master.
    #         # - for client skew tests, we want to use the skewed kubectl
    #         (that's what we're testing).
    GINKGO_TEST_ARGS="${GINKGO_TEST_ARGS:-} --kubectl-path=$(pwd)/../kubernetes_skew/cluster/kubectl.sh"
  fi
fi

cd kubernetes

# Upload build start time and k8s version to GCS, but not on PR Jenkins.
# On PR Jenkins this is done before the build.
if [[ ! "${JOB_NAME}" =~ -pull- ]]; then
    JENKINS_BUILD_STARTED=true bash <(curl -fsS --retry 3 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/upload-to-gcs.sh")
fi

# Have cmd/e2e run by goe2e.sh generate JUnit report in ${WORKSPACE}/junit*.xml
ARTIFACTS=${WORKSPACE}/_artifacts
mkdir -p ${ARTIFACTS}
# When run inside Docker, we need to make sure all files are world-readable
# (since they will be owned by root on the host).
trap "chmod -R o+r '${ARTIFACTS}'" EXIT SIGINT SIGTERM
export E2E_REPORT_DIR=${ARTIFACTS}

e2e_go_args=( \
  -v \
  --dump="${ARTIFACTS}" \
)


case "${KUBERNETES_PROVIDER}" in
  gce|gke)
    e2e_go_args+=(--check_leaked_resources)
    ;;
esac

if [[ "${E2E_UP,,}" == "true" ]]; then
  e2e_go_args+=(--up --ctl="version --match-server-version=false")
fi

if [[ "${E2E_DOWN,,}" == "true" ]]; then
  e2e_go_args+=(--down)
fi

if [[ "${E2E_TEST,,}" == "true" ]]; then
  e2e_go_args+=(--test --test_args="${GINKGO_TEST_ARGS}")
fi

# Optionally run tests from the version in  kubernetes_skew
if [[ "${JENKINS_USE_SKEW_TESTS:-}" == "true" ]]; then
  e2e_go_args+=(--skew)
fi

# Optionally run upgrade tests before other tests.
if [[ "${E2E_UPGRADE_TEST:-}" == "true" ]]; then
  e2e_go_args+=(--upgrade_args="${GINKGO_UPGRADE_TEST_ARGS}")
fi

go run ./hack/e2e.go \
  ${E2E_OPT:-} \
  "${e2e_go_args[@]}"

if [[ "${E2E_PUBLISH_GREEN_VERSION:-}" == "true" ]]; then
  # Use plaintext version file packaged with kubernetes.tar.gz
  echo "Publish version to ci/latest-green.txt: $(cat version)"
  gsutil cp ./version "gs://${KUBE_GCS_DEV_RELEASE_BUCKET}/ci/latest-green.txt"
fi
