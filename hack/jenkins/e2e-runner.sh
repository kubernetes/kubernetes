#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# include shell2junit library
source <(curl -fsS --retry 3 'https://raw.githubusercontent.com/kubernetes/kubernetes/master/third_party/forked/shell2junit/sh2ju.sh')

# Have cmd/e2e run by goe2e.sh generate JUnit report in ${WORKSPACE}/junit*.xml
ARTIFACTS=${WORKSPACE}/_artifacts
mkdir -p ${ARTIFACTS}

# E2E runner stages
STAGE_PRE="PRE-SETUP"
STAGE_SETUP="SETUP"
STAGE_CLEANUP="CLEANUP"
STAGE_KUBEMARK="KUBEMARK"

: ${KUBE_GCS_RELEASE_BUCKET:="kubernetes-release"}
: ${KUBE_GCS_DEV_RELEASE_BUCKET:="kubernetes-release-dev"}

# record_command runs the command and records its output/error messages in junit format
# it expects the first argument to be the class and the second to be the name of the command
# Example:
# record_command PRESETUP curltest curl google.com
# record_command CLEANUP check false
#
# WARNING: Variable changes in the command will NOT be effective after record_command returns.
#          This is because the command runs in subshell.
function record_command() {
    set +o xtrace
    set +o nounset
    set +o errexit

    local class=$1
    shift
    local name=$1
    shift
    echo "Recording: ${class} ${name}"
    echo "Running command: $@"
    juLog -output="${ARTIFACTS}" -class="${class}" -name="${name}" "$@"

    set -o nounset
    set -o errexit
    set -o xtrace
}

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

function fetch_gci_version_tars() {
    if ! [[ "${JENKINS_USE_GCI_VERSION:-}" =~ ^[yY]$ ]]; then
        echo "JENKINS_USE_GCI_VERSION must be set."
        exit 1
    fi
    local -r gci_k8s_version="$(get_gci_k8s_version)"
    echo "Using GCI builtin version: ${gci_k8s_version}"
    fetch_tars_from_gcs "gs://${KUBE_GCS_RELEASE_BUCKET}/release" "${gci_k8s_version}"
    unpack_binaries
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
    record_command "${STAGE_PRE}" "install_gcloud" "${install_dir}/google-cloud-sdk/install.sh" --disable-installation-options --bash-completion=false --path-update=false --usage-reporting=false
    export PATH=${install_dir}/google-cloud-sdk/bin:${PATH}
}

# Only call after attempting to bring the cluster up. Don't call after
# bringing the cluster down.
function dump_cluster_logs_and_exit() {
    local -r exit_status=$?
    dump_cluster_logs
    if [[ "${USE_KUBEMARK:-}" == "true" ]]; then
      # If we tried to bring the Kubemark cluster up, make a courtesy
      # attempt to bring it down so we're not leaving resources around.
      ./test/kubemark/stop-kubemark.sh || true
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

# Only call after attempting to bring the cluster up. Don't call after
# bringing the cluster down.
function dump_cluster_logs() {
    if [[ -x "cluster/log-dump.sh"  ]]; then
        ./cluster/log-dump.sh "${ARTIFACTS}"
    fi
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
    record_command "${STAGE_PRE}" "download_gcloud" curl -fsSL --retry 3 --keepalive-time 2 -o "${WORKSPACE}/google-cloud-sdk.tar.gz" 'https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz'
    install_google_cloud_sdk_tarball "${WORKSPACE}/google-cloud-sdk.tar.gz" /
    if [[ "${KUBERNETES_PROVIDER}" == 'aws' ]]; then
        pip install awscli
    fi
fi

# Install gcloud from a custom path if provided. Used to test GKE with gcloud
# at HEAD, release candidate.
# TODO: figure out how to avoid installing the cloud sdk twice if run inside Docker.
if [[ -n "${CLOUDSDK_BUCKET:-}" ]]; then
    # Retry the download a few times to mitigate transient server errors and
    # race conditions where the bucket contents change under us as we download.
    for n in {1..3}; do
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

# GCI specific settings.
if [[ -n "${JENKINS_GCI_IMAGE_FAMILY:-}" ]]; then
  setup_gci_vars
fi

if [[ -f "${KUBEKINS_SERVICE_ACCOUNT_FILE:-}" ]]; then
  echo 'Activating service account...'  # No harm in doing this multiple times.
  gcloud auth activate-service-account --key-file="${KUBEKINS_SERVICE_ACCOUNT_FILE}"
  # https://developers.google.com/identity/protocols/application-default-credentials
  export GOOGLE_APPLICATION_CREDENTIALS="${KUBEKINS_SERVICE_ACCOUNT_FILE}"
  unset KUBEKINS_SERVICE_ACCOUNT_FILE
elif [[ -n "${KUBEKINS_SERVICE_ACCOUNT_FILE:-}" ]]; then
  echo "ERROR: cannot access service account file at: ${KUBEKINS_SERVICE_ACCOUNT_FILE}"
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
        gsutil cp ./version "gs://${KUBE_GCS_DEV_RELEASE_BUCKET}/ci/latest-green.txt"
    fi
    return ${exitcode}
}

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
elif [[ "${JENKINS_USE_GCI_VERSION:-}" =~ ^[yY]$ ]]; then
    clean_binaries
    fetch_gci_version_tars
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

cd kubernetes

# Upload build start time and k8s version to GCS, but not on PR Jenkins.
# On PR Jenkins this is done before the build.
if [[ ! "${JOB_NAME}" =~ -pull- ]]; then
    JENKINS_BUILD_STARTED=true bash <(curl -fsS --retry 3 --keepalive-time 2 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/upload-to-gcs.sh")
fi

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
  curl -fsS --retry 3 --keepalive-time 2 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/cluster/gce/list-resources.sh" > "${gcp_list_resources_script}"
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
    cd ..
    mv kubernetes kubernetes_old
    fetch_published_version_tars "${JENKINS_PUBLISHED_SKEW_VERSION}"
    cd kubernetes
    # Upgrade the cluster before running other tests
    if [[ "${E2E_UPGRADE_TEST:-}" == "true" ]]; then
        # Add a report prefix for the e2e tests so that the tests don't get overwritten when we run
        # the rest of the e2es.
        E2E_REPORT_PREFIX='upgrade' e2e_test "${GINKGO_UPGRADE_TEST_ARGS:-}" || EXIT_CODE=1
    fi
    if [[ "${JENKINS_USE_SKEW_TESTS:-}" != "true" ]]; then
        # Back out into the old tests now that we've downloaded & maybe upgraded.
        cd ../kubernetes_old
        # Append kubectl-path of skewed kubectl to test args, since we always
        # want that to use the skewed kubectl version:
        #
        # - for upgrade jobs, we want kubectl to be at the same version as master.
        # - for client skew tests, we want to use the skewed kubectl (that's what we're testing).
        GINKGO_TEST_ARGS="${GINKGO_TEST_ARGS:-} --kubectl-path=$(pwd)/../kubernetes/cluster/kubectl.sh"
    fi
fi

if [[ "${E2E_TEST,,}" == "true" ]]; then
    e2e_test "${GINKGO_TEST_ARGS:-}" || EXIT_CODE=1
fi

### Start Kubemark ###
if [[ "${USE_KUBEMARK:-}" == "true" ]]; then
  export RUN_FROM_DISTRO=true
  NUM_NODES_BKP=${NUM_NODES}
  MASTER_SIZE_BKP=${MASTER_SIZE}
  ./test/kubemark/stop-kubemark.sh
  NUM_NODES=${KUBEMARK_NUM_NODES:-$NUM_NODES}
  MASTER_SIZE=${KUBEMARK_MASTER_SIZE:-$MASTER_SIZE}
  ./test/kubemark/start-kubemark.sh || dump_cluster_logs_and_exit
  # Similarly, if tests fail, we trigger empty set of tests that would trigger storing logs from the base cluster.
  # We intentionally overwrite the exit-code from `run-e2e-tests.sh` because we want jenkins to look at the
  # junit.xml results for test failures and not process the exit code.  This is needed by jenkins to more gracefully
  # handle blocking the merge queue as a result of test failure flakes.  Infrastructure failures should continue to
  # exit non-0.
  # TODO: The above comment is no longer accurate. Need to fix this before
  # turning xunit off for the postsubmit tests. See: #28200
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
  noleak=true
  if [[ -n $(echo "${difference}" | tail -n +3 | grep -E "^\+") ]] && [[ "${FAIL_ON_GCP_RESOURCE_LEAK:-}" == "true" ]]; then
    noleak=false
  fi
  if ! ${noleak} ; then
    echo "${difference}"
    echo "!!! FAIL: Google Cloud Platform resources leaked while running tests!"
    EXIT_CODE=1
  fi
  record_command "${STAGE_CLEANUP}" "gcp_resource_leak_check" ${noleak}
fi

exit ${EXIT_CODE}
