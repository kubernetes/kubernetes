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

function check_dirty_workspace() {
    if [[ "${JENKINS_TOLERATE_DIRTY_WORKSPACE:-}" =~ ^[yY]$ ]]; then
        echo "Tolerating dirty workspace (because JENKINS_TOLERATE_DIRTY_WORKSPACE)."
    else
        echo "Checking dirty workspace."
        # .config and its children are created by the gcloud call that we use to
        # get the GCE service account.
        #
        # console-log.txt is created by Jenkins, but is usually not flushed out
        # this early in this script.
        if [[ $(find . -not -path "./.config*" -not -name "console-log.txt" | wc -l) != 1 ]]; then
            echo "${PWD} not empty, bailing!"
            find .
            exit 1
        fi
    fi
}

function fetch_output_tars() {
    clean_binaries
    echo "Using binaries from _output."
    cp _output/release-tars/kubernetes*.tar.gz .
    unpack_binaries
}

function fetch_server_version_tars() {
    clean_binaries
    local -r msg=$(gcloud ${CMD_GROUP} container get-server-config --project=${PROJECT} --zone=${ZONE} | grep defaultClusterVersion)
    # msg will look like "defaultClusterVersion: 1.0.1". Strip
    # everything up to, including ": "
    local -r build_version="v${msg##*: }"
    fetch_tars_from_gcs "release" "${build_version}"
    unpack_binaries
}

# Use a published version like "ci/latest" (default), "release/latest",
# "release/latest-1", or "release/stable"
function fetch_published_version_tars() {
    clean_binaries
    local -r published_version="${1}"
    IFS='/' read -a varr <<< "${published_version}"
    bucket="${varr[0]}"
    build_version=$(gsutil cat gs://kubernetes-release/${published_version}.txt)
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
        "gs://kubernetes-release/${bucket}/${build_version}/kubernetes.tar.gz" \
        "gs://kubernetes-release/${bucket}/${build_version}/kubernetes-test.tar.gz" \
        .
}

function unpack_binaries() {
    md5sum kubernetes*.tar.gz
    tar -xzf kubernetes.tar.gz
    tar -xzf kubernetes-test.tar.gz
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
    fetch_output_tars
elif [[ "${JENKINS_USE_SERVER_VERSION:-}" =~ ^[yY]$ ]]; then
    # This is for test, staging, and prod jobs on GKE, where we want to
    # test what's running in GKE by default rather than some CI build.
    check_dirty_workspace
    fetch_server_version_tars
else
    # use JENKINS_PUBLISHED_VERSION, default to 'ci/latest', since that's
    # usually what we're testing.
    check_dirty_workspace
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
if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
    echo "Skipping SSH key copying for AWS"
else
    mkdir -p ${WORKSPACE}/.ssh/
    cp /var/lib/jenkins/gce_keys/google_compute_engine ${WORKSPACE}/.ssh/
    cp /var/lib/jenkins/gce_keys/google_compute_engine.pub ${WORKSPACE}/.ssh/
fi

cd kubernetes

# Upload build start time and k8s version to GCS, but not on PR Jenkins.
if [[ ! "${JOB_NAME}" =~ -pull- ]]; then
    bash <(curl -fsS --retry 3 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/upload-started.sh")
fi

# Have cmd/e2e run by goe2e.sh generate JUnit report in ${WORKSPACE}/junit*.xml
ARTIFACTS=${WORKSPACE}/_artifacts
mkdir -p ${ARTIFACTS}
export E2E_REPORT_DIR=${ARTIFACTS}
declare -r gcp_list_resources_script="./cluster/gce/list-resources.sh"
declare -r gcp_resources_before="${ARTIFACTS}/gcp-resources-before.txt"
declare -r gcp_resources_cluster_up="${ARTIFACTS}/gcp-resources-cluster-up.txt"
declare -r gcp_resources_after="${ARTIFACTS}/gcp-resources-after.txt"
# TODO(15492): figure out some way to run this script even if it doesn't exist
# in the Kubernetes tarball.
if [[ ( ${KUBERNETES_PROVIDER} == "gce" || ${KUBERNETES_PROVIDER} == "gke" ) && -x "${gcp_list_resources_script}" ]]; then
  gcp_list_resources="true"
else
  gcp_list_resources="false"
fi

### Pre Set Up ###
# Install gcloud from a custom path if provided. Used to test GKE with gcloud
# at HEAD, release candidate.
if [[ ! -z "${CLOUDSDK_BUCKET:-}" ]]; then
    gsutil -mq cp -r "${CLOUDSDK_BUCKET}" ~
    rm -rf ~/repo ~/cloudsdk
    mv ~/$(basename "${CLOUDSDK_BUCKET}") ~/repo
    mkdir ~/cloudsdk
    tar zxf ~/repo/google-cloud-sdk.tar.gz -C ~/cloudsdk
    export CLOUDSDK_CORE_DISABLE_PROMPTS=1
    export CLOUDSDK_COMPONENT_MANAGER_SNAPSHOT_URL=file://${HOME}/repo/components-2.json
    ~/cloudsdk/google-cloud-sdk/install.sh --disable-installation-options --bash-completion=false --path-update=false --usage-reporting=false
    export PATH=${HOME}/cloudsdk/google-cloud-sdk/bin:${PATH}
    export CLOUDSDK_CONFIG=/var/lib/jenkins/.config/gcloud
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
    go run ./hack/e2e.go ${E2E_OPT:-} -v --up && up_result="$?" || up_result="$?"
    if [[ "${up_result}" -ne 0 ]]; then
        if [[ -x "cluster/log-dump.sh"  ]]; then
            ./cluster/log-dump.sh "${ARTIFACTS}"
        fi
        exit "${up_result}"
    fi
    go run ./hack/e2e.go -v --ctl="version --match-server-version=false"
    if [[ "${gcp_list_resources}" == "true" ]]; then
      ${gcp_list_resources_script} > "${gcp_resources_cluster_up}"
    fi
fi

### Run tests ###
# Jenkins will look at the junit*.xml files for test failures, so don't exit
# with a nonzero error code if it was only tests that failed.
if [[ "${E2E_TEST,,}" == "true" ]]; then
    # Check to make sure the cluster is up before running tests, and fail if it's not.
    go run ./hack/e2e.go ${E2E_OPT:-} -v --isup
    go run ./hack/e2e.go ${E2E_OPT:-} -v --test \
      ${GINKGO_TEST_ARGS:+--test_args="${GINKGO_TEST_ARGS}"} \
      && exitcode=0 || exitcode=$?
    if [[ "${E2E_PUBLISH_GREEN_VERSION:-}" == "true" && ${exitcode} == 0 && -n ${build_version:-} ]]; then
        echo "Publish build_version to ci/latest-green.txt: ${build_version}"
        gsutil cp ./version gs://kubernetes-release/ci/latest-green.txt
    fi
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
  ./test/kubemark/start-kubemark.sh && kubemark_started="$?" || kubemark_started="$?"
  if [[ "${kubemark_started}" != "0" ]]; then
    go run ./hack/e2e.go -v --test --test_args="--ginkgo.focus=DO\sNOT\sMATCH\sANYTHING"
    exit 1
  fi
  # Similarly, if tests fail, we trigger empty set of tests that would trigger storing logs from the base cluster.
  ./test/kubemark/run-e2e-tests.sh --ginkgo.focus="${KUBEMARK_TESTS}" --gather-resource-usage="false" && kubemark_succeeded="$?" || kubemark_succeeded="$?"
  if [[ "${kubemark_succeeded}" != "0" ]]; then
    go run ./hack/e2e.go -v --test --test_args="--ginkgo.focus=DO\sNOT\sMATCH\sANYTHING"
    exit 1
  fi
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
  if ! diff -sw -U0 -F'^\[.*\]$' "${gcp_resources_before}" "${gcp_resources_after}" && [[ "${FAIL_ON_GCP_RESOURCE_LEAK:-}" == "true" ]]; then
    echo "!!! FAIL: Google Cloud Platform resources leaked while running tests!"
    exit 1
  fi
fi
