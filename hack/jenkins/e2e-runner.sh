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

function running_in_docker() {
    grep -q docker /proc/self/cgroup
}

function fetch_output_tars() {
    echo "Using binaries from _output."
    cp _output/release-tars/kubernetes*.tar.gz .
    unpack_binaries
}

function fetch_server_version_tars() {
    local -r msg=$(gcloud ${CMD_GROUP:-} container get-server-config --project=${PROJECT} --zone=${ZONE} | grep defaultClusterVersion)
    # msg will look like "defaultClusterVersion: 1.0.1". Strip
    # everything up to, including ": "
    local -r build_version="v${msg##*: }"
    fetch_tars_from_gcs "release" "${build_version}"
    unpack_binaries
}

# Use a published version like "ci/latest" (default), "release/latest",
# "release/latest-1", or "release/stable"
function fetch_published_version_tars() {
    local -r published_version="${1}"
    IFS='/' read -a varr <<< "${published_version}"
    bucket="${varr[0]}"
    build_version=$(gsutil cat gs://${KUBE_GCS_RELEASE_BUCKET}/${published_version}.txt)
    echo "Using published version $bucket/$build_version (from ${published_version})"
    fetch_tars_from_gcs "${bucket}" "${build_version}"
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
    local -r bucket="${1}"
    local -r build_version="${2}"
    echo "Pulling binaries from GCS; using server version ${bucket}/${build_version}."
    gsutil -mq cp \
        "gs://${KUBE_GCS_RELEASE_BUCKET}/${bucket}/${build_version}/kubernetes.tar.gz" \
        "gs://${KUBE_GCS_RELEASE_BUCKET}/${bucket}/${build_version}/kubernetes-test.tar.gz" \
        .
}

function unpack_binaries() {
    md5sum kubernetes*.tar.gz
    tar -xzf kubernetes.tar.gz
    tar -xzf kubernetes-test.tar.gz
}

# GCP Project to fetch Trusty images.
function get_trusty_image_project() {
  local project=""
  # Retry the gsutil command a couple times to mitigate the effect of
  # transient server errors.
  for n in $(seq 3); do
    project="$(gsutil cat "gs://trusty-images/image-project.txt")" && break || sleep 1
  done
  if [[ -z "${project}" ]]; then
    echo "Failed to find the image project for Trusty images."
    exit 1
  fi
  echo "${project}"
  # Clean up gsutil artifacts otherwise the later test stage will complain.
  rm -rf .config &> /dev/null
  rm -rf .gsutil &> /dev/null
}

# Get the latest Trusty image for a Jenkins job.
function get_latest_trusty_image() {
    local image_project="$1"
    local image_type="$2"
    local image_index=""
    if [[ "${image_type}" == head ]]; then
      image_index="trusty-head"
    elif [[ "${image_type}" == dev ]]; then
      image_index="trusty-dev"
    elif [[ "${image_type}" == beta ]]; then
      image_index="trusty-beta"
    elif [[ "${image_type}" == stable ]]; then
      image_index="trusty-stable"
    fi

    local image=""
    # Retry the gsutil command a couple times to mitigate the effect of
    # transient server errors.
    for n in $(seq 3); do
      image="$(gsutil cat "gs://${image_project}/image-indices/latest-test-image-${image_index}")" && break || sleep 1
    done
    if [[ -z "${image}" ]]; then
      echo "Failed to find Trusty image for ${image_type}"
      exit 1
    fi
    echo "${image}"
    # Clean up gsutil artifacts otherwise the later test stage will complain.
    rm -rf .config &> /dev/null
    rm -rf .gsutil &> /dev/null
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

# Only call after attempting to bring the cluster up. Don't call after
# bringing the cluster down.
function dump_cluster_logs_and_exit() {
    local -r exit_status=$?
    if [[ -x "cluster/log-dump.sh"  ]]; then
        ./cluster/log-dump.sh "${ARTIFACTS}"
    fi
    if [[ "${E2E_DOWN,,}" == "true" ]]; then
      # If we tried to bring the cluster up, make a courtesy attempt
      # to bring the cluster down so we're not leaving resources
      # around. Unlike later, don't sleep beforehand, though. (We're
      # just trying to tear down as many resources as we can as fast
      # as possible and don't even know if we brought the master up.)
      go run ./hack/e2e.go ${E2E_OPT:-} -v --down || true
    fi
    exit ${exit_status}
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

# We get the image project and name for Trusty dynamically.
if [[ -n "${JENKINS_TRUSTY_IMAGE_TYPE:-}" ]]; then
  trusty_image_project="$(get_trusty_image_project)"
  trusty_image="$(get_latest_trusty_image "${trusty_image_project}" "${JENKINS_TRUSTY_IMAGE_TYPE}")"
  export KUBE_GCE_MASTER_PROJECT="${trusty_image_project}"
  export KUBE_GCE_MASTER_IMAGE="${trusty_image}"
  export KUBE_OS_DISTRIBUTION="trusty"
fi

function e2e_test() {
    local -r ginkgo_test_args="${1}"
    # Check to make sure the cluster is up before running tests, and fail if it's not.
    go run ./hack/e2e.go ${E2E_OPT:-} -v --isup
    # Jenkins will look at the junit*.xml files for test failures, so don't exit with a nonzero
    # error code if it was only tests that failed.
    go run ./hack/e2e.go ${E2E_OPT:-} -v --test \
      ${ginkgo_test_args:+--test_args="${ginkgo_test_args}"} \
      && exitcode=0 || exitcode=$?
    if [[ "${E2E_PUBLISH_GREEN_VERSION:-}" == "true" && ${exitcode} == 0 ]]; then
        # Use plaintext version file packaged with kubernetes.tar.gz
        echo "Publish version to ci/latest-green.txt: $(cat version)"
        gsutil cp ./version gs://kubernetes-release/ci/latest-green.txt
    fi
}

echo "--------------------------------------------------------------------------------"
echo "Test Environment:"
printenv | sort
echo "--------------------------------------------------------------------------------"

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
        if [[ ! -f ${WORKSPACE}/.ssh/google_compute_engine ]]; then
            echo "google_compute_engine ssh key missing!"
            exit 1
        fi
        ;;

    default)
        echo "Not copying ssh keys for ${KUBERNETES_PROVIDER}"
        ;;
esac

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
declare -r gcp_list_resources_script="./cluster/gce/list-resources.sh"
declare -r gcp_resources_before="${ARTIFACTS}/gcp-resources-before.txt"
declare -r gcp_resources_cluster_up="${ARTIFACTS}/gcp-resources-cluster-up.txt"
declare -r gcp_resources_after="${ARTIFACTS}/gcp-resources-after.txt"
if [[ ( ${KUBERNETES_PROVIDER} == "gce" || ${KUBERNETES_PROVIDER} == "gke" ) && -x "${gcp_list_resources_script}" ]]; then
  gcp_list_resources="true"
  # Always pull the script from HEAD, overwriting the local one if it exists.
  # We do this to pick up fixes if we are running tests from a branch or tag.
  curl -fsS --retry 3 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/cluster/gce/list-resources.sh" > "${gcp_list_resources_script}"
else
  gcp_list_resources="false"
fi

### Set up ###
if [[ "${E2E_UP,,}" == "true" ]]; then
    go run ./hack/e2e.go ${E2E_OPT:-} -v --down
fi
if [[ "${gcp_list_resources}" == "true" ]]; then
  ${gcp_list_resources_script} > "${gcp_resources_before}"
fi
if [[ "${E2E_UP,,}" == "true" ]]; then
    # We want to try to gather logs even if kube-up fails, so collect the
    # result here and fail after dumping logs if it's nonzero.
    go run ./hack/e2e.go ${E2E_OPT:-} -v --up || dump_cluster_logs_and_exit
    go run ./hack/e2e.go -v --ctl="version --match-server-version=false"
    if [[ "${gcp_list_resources}" == "true" ]]; then
      ${gcp_list_resources_script} > "${gcp_resources_cluster_up}"
    fi
fi

# Allow download & unpack of alternate version of tests, for cross-version & upgrade testing.
if [[ -n "${JENKINS_PUBLISHED_TEST_VERSION:-}" ]]; then
    cd ..
    mv kubernetes kubernetes_old
    fetch_published_version_tars "${JENKINS_PUBLISHED_TEST_VERSION}"
    cd kubernetes
    # Upgrade the cluster before running other tests
    if [[ "${E2E_UPGRADE_TEST:-}" == "true" ]]; then
	# Add a report prefix for the e2e tests so that the tests don't get overwritten when we run
	# the rest of the e2es.
        E2E_REPORT_PREFIX='upgrade' e2e_test "${GINKGO_UPGRADE_TEST_ARGS:-}"
	# If JENKINS_USE_OLD_TESTS is set, back out into the old tests now that we've upgraded.
        if [[ "${JENKINS_USE_OLD_TESTS:-}" == "true" ]]; then
            cd ../kubernetes_old
        fi
    fi
fi

if [[ "${E2E_TEST,,}" == "true" ]]; then
    e2e_test "${GINKGO_TEST_ARGS:-}"
fi

### Start Kubemark ###
if [[ "${USE_KUBEMARK:-}" == "true" ]]; then
  export RUN_FROM_DISTRO=true
  NUM_NODES_BKP=${NUM_NODES}
  MASTER_SIZE_BKP=${MASTER_SIZE}
  ./test/kubemark/stop-kubemark.sh
  NUM_NODES=${KUBEMARK_NUM_NODES:-$NUM_NODES}
  MASTER_SIZE=${KUBEMARK_MASTER_SIZE:-$MASTER_SIZE}
  # If start-kubemark fails, we trigger empty set of tests that would trigger storing logs from the base cluster.
  ./test/kubemark/start-kubemark.sh || dump_cluster_logs_and_exit
  # Similarly, if tests fail, we trigger empty set of tests that would trigger storing logs from the base cluster.
  ./test/kubemark/run-e2e-tests.sh --ginkgo.focus="${KUBEMARK_TESTS:-starting\s30\spods}" "${KUBEMARK_TEST_ARGS:-}" || dump_cluster_logs_and_exit
  ./test/kubemark/stop-kubemark.sh
  NUM_NODES=${NUM_NODES_BKP}
  MASTER_SIZE=${MASTER_SIZE_BKP}
  unset RUN_FROM_DISTRO
  unset NUM_NODES_BKP
  unset MASTER_SIZE_BKP
fi

### Clean up ###
if [[ "${E2E_DOWN,,}" == "true" ]]; then
    # Sleep before deleting the cluster to give the controller manager time to
    # delete any cloudprovider resources still around from the last test.
    # This is calibrated to allow enough time for 3 attempts to delete the
    # resources. Each attempt is allocated 5 seconds for requests to the
    # cloudprovider plus the processingRetryInterval from servicecontroller.go
    # for the wait between attempts.
    sleep 30
    go run ./hack/e2e.go ${E2E_OPT:-} -v --down
fi
if [[ "${gcp_list_resources}" == "true" ]]; then
  ${gcp_list_resources_script} > "${gcp_resources_after}"
fi

# Compare resources if either the cluster was
# * started and destroyed (normal e2e)
# * neither started nor destroyed (soak test)
if [[ "${E2E_UP:-}" == "${E2E_DOWN:-}" && -f "${gcp_resources_before}" && -f "${gcp_resources_after}" ]]; then
  difference=$(diff -sw -U0 -F'^\[.*\]$' "${gcp_resources_before}" "${gcp_resources_after}") || true
  if [[ -n $(echo "${difference}" | tail -n +3 | grep -E "^\+") ]] && [[ "${FAIL_ON_GCP_RESOURCE_LEAK:-}" == "true" ]]; then
    echo "${difference}"
    echo "!!! FAIL: Google Cloud Platform resources leaked while running tests!"
    exit 1
  fi
fi
