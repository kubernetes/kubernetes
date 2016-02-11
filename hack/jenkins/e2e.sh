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

# Sets up environment variables for an e2e test specified in JOB_NAME, then
# runs e2e-runner.sh.

set -o errexit
set -o nounset
set -o pipefail

# Properly configure globals for an upgrade step in a GKE or GCE upgrade suite
#
# These suites:
#   step1: launch a cluster at $old_version,
#   step2: runs $new_version Kubectl e2es,
#   step3: upgrades the master to $new_version,
#   step4: runs $old_version e2es,
#   step5: upgrades the rest of the cluster,
#   step6: runs $old_version e2es again, then
#   step7: runs $new_version e2es and tears down the cluster.
#
# Assumes globals:
#   $JOB_NAME
#   $KUBERNETES_PROVIDER
#
# Args:
#   $1 old_version:  the version to deploy a cluster at, and old e2e tests to run
#                    against the upgraded cluster (should be something like
#                    'release/latest', to work with JENKINS_PUBLISHED_VERSION logic)
#   $2 new_version:  the version to upgrade the cluster to, and new e2e tests to run
#                    against the upgraded cluster (should be something like
#                    'ci/latest', to work with JENKINS_PUBLISHED_VERSION logic)
#   $3 cluster_name: determines E2E_CLUSTER_NAME and E2E_NETWORK
#   $4 project:      determines PROJECT

function configure_upgrade_step() {
  local -r old_version="$1"
  local -r new_version="$2"
  local -r cluster_name="$3"
  local -r project="$4"

  [[ "${JOB_NAME}" =~ .*-(step[1-7])-.* ]] || {
    echo "JOB_NAME ${JOB_NAME} is not a valid upgrade job name, could not parse"
    exit 1
  }
  local -r step="${BASH_REMATCH[1]}"

  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
    KUBE_GCE_INSTANCE_PREFIX="$cluster_name"
    NUM_NODES=5
    KUBE_ENABLE_DEPLOYMENTS=true
    KUBE_ENABLE_DAEMONSETS=true
  fi

  E2E_CLUSTER_NAME="$cluster_name"
  E2E_NETWORK="$cluster_name"
  PROJECT="$project"

  case $step in
    step1)
      # Deploy at old version
      JENKINS_PUBLISHED_VERSION="${old_version}"

      E2E_UP="true"
      E2E_TEST="false"
      E2E_DOWN="false"

      if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
        E2E_SET_CLUSTER_API_VERSION=y
      fi
      ;;

    step2)
      # Run new e2e kubectl tests
      JENKINS_PUBLISHED_VERSION="${new_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"
      GINKGO_TEST_ARGS="--ginkgo.focus=Kubectl"
      ;;

    step3)
      # Use upgrade logic of version we're upgrading to.
      JENKINS_PUBLISHED_VERSION="${new_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"
      GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Upgrade\].*upgrade-master --upgrade-target=${new_version}"
      ;;

    step4)
      # Run old e2es
      JENKINS_PUBLISHED_VERSION="${old_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"
      ;;

    step5)
      # Use upgrade logic of version we're upgrading to.
      JENKINS_PUBLISHED_VERSION="${new_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"
      GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Upgrade\].*upgrade-cluster --upgrade-target=${new_version}"
      ;;

    step6)
      # Run old e2es
      JENKINS_PUBLISHED_VERSION="${old_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"
      ;;

    step7)
      # Run new e2es
      JENKINS_PUBLISHED_VERSION="${new_version}"
      JENKINS_FORCE_GET_TARS=y

      # TODO(15011): these really shouldn't be (very) version skewed, but
      # because we have to get ci/latest again, it could get slightly out of
      # whack.
      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="true"
      ;;
  esac
}

echo "--------------------------------------------------------------------------------"
echo "Initial Environment:"
printenv | sort
echo "--------------------------------------------------------------------------------"

if [[ "${CIRCLECI:-}" == "true" ]]; then
    JOB_NAME="circleci-${CIRCLE_PROJECT_USERNAME}-${CIRCLE_PROJECT_REPONAME}"
    BUILD_NUMBER=${CIRCLE_BUILD_NUM}
    WORKSPACE=`pwd`
else
    # Jenkins?
    export HOME=${WORKSPACE} # Nothing should want Jenkins $HOME
fi

# Additional parameters that are passed to hack/e2e.go
E2E_OPT=${E2E_OPT:-""}

# Set environment variables shared for all of the GCE Jenkins projects.
if [[ ${JOB_NAME} =~ ^kubernetes-.*-gce ]]; then
  KUBERNETES_PROVIDER="gce"
  : ${E2E_MIN_STARTUP_PODS:="1"}
  : ${E2E_ZONE:="us-central1-f"}
  : ${NUM_NODES_PARALLEL:="6"}  # Number of nodes required to run all of the tests in parallel

elif [[ ${JOB_NAME} =~ ^kubernetes-.*-gke ]]; then
  KUBERNETES_PROVIDER="gke"
  : ${E2E_ZONE:="us-central1-f"}
  # By default, GKE tests run against the GKE test endpoint using CI Cloud SDK.
  # Release jobs (e.g. prod, staging, and test) override these two variables.
  : ${CLOUDSDK_BUCKET:="gs://cloud-sdk-build/testing/staging"}
  : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
elif [[ ${JOB_NAME} =~ ^kubernetes-.*-aws ]]; then
  KUBERNETES_PROVIDER="aws"
  : ${E2E_MIN_STARTUP_PODS:="1"}
  : ${E2E_ZONE:="us-east-1a"}
  : ${NUM_NODES_PARALLEL:="6"}  # Number of nodes required to run all of the tests in parallel
fi

if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
  if [[ "${PERFORMANCE:-}" == "true" ]]; then
    : ${MASTER_SIZE:="m3.xlarge"}
    : ${NUM_NODES:="100"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Performance\]"}
  else
    : ${MASTER_SIZE:="m3.medium"}
    : ${NODE_SIZE:="m3.medium"}
    : ${NUM_NODES:="3"}
  fi
fi

# CURRENT_RELEASE_PUBLISHED_VERSION is the JENKINS_PUBLISHED_VERSION for the
# release we are currently pointing our release testing infrastructure at.
# When 1.2.0-beta.0 comes out, e.g., this will become "ci/latest-1.2"
CURRENT_RELEASE_PUBLISHED_VERSION="ci/latest-1.1"

# Define environment variables based on the Jenkins project name.
# NOTE: Not all jobs are defined here. The hack/jenkins/e2e.sh in master and
# release branches defines relevant jobs for that particular version of
# Kubernetes.
case ${JOB_NAME} in

  # Runs a subset of tests on GCE in parallel. Run against all pending PRs.
  kubernetes-pull-build-test-e2e-gce)
    : ${E2E_CLUSTER_NAME:="jnks-e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${E2E_NETWORK:="e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${GINKGO_PARALLEL:="y"}
    # This list should match the list in kubernetes-e2e-gce.
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${PROJECT:="kubernetes-jenkins-pull"}
    : ${ENABLE_DEPLOYMENTS:=true}
    # Override GCE defaults
    NUM_NODES=${NUM_NODES_PARALLEL}
    ;;

  # Runs the performance/scalability test on huge 1000-node cluster on GCE.
  # Flannel is used as network provider.
  # Allows a couple of nodes to be NotReady during startup
  kubernetes-e2e-gce-enormous-cluster)
    export E2E_CLUSTER_NAME="jenkins-gce-enormous-cluster"
    export E2E_NETWORK="e2e-enormous-cluster"
    # TODO: Currently run only density test.
    # Once this is stable, run the whole [Performance] suite.
    export GINKGO_TEST_ARGS="--ginkgo.focus=starting\s30\spods\sper\snode"
    export KUBE_GCE_INSTANCE_PREFIX="e2e-enormous-cluster"
    export PROJECT="kubernetes-scale"
    # Override GCE defaults.
    export NETWORK_PROVIDER="flannel"
    # Temporarily switch of Heapster, as this will not schedule anywhere.
    # TODO: Think of a solution to enable it.
    export ENABLE_CLUSTER_MONITORING="none"
    export E2E_ZONE="asia-east1-a"
    export MASTER_SIZE="n1-standard-32"
    export NODE_SIZE="n1-standard-1"
    export NODE_DISK_SIZE="50GB"
    export NUM_NODES="1000"
    export ALLOWED_NOTREADY_NODES="2"
    export EXIT_ON_WEAK_ERROR="false"
    # Reduce logs verbosity
    export TEST_CLUSTER_LOG_LEVEL="--v=1"
    # Increase resync period to simulate production
    export TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Starts and tears down 1000-node cluster on GCE using flannel networking
  # Requires all 1000 nodes to come up.
  kubernetes-e2e-gce-enormous-startup)
    export E2E_CLUSTER_NAME="jenkins-gce-enormous-startup"
    # TODO: increase a quota for networks in kubernetes-scale and move this test to its own network
    export E2E_NETWORK="e2e-enormous-cluster"
    export E2E_TEST="false"
    export KUBE_GCE_INSTANCE_PREFIX="e2e-enormous-startup"
    export PROJECT="kubernetes-scale"
    # Override GCE defaults.
    export NETWORK_PROVIDER="flannel"
    # Temporarily switch of Heapster, as this will not schedule anywhere.
    # TODO: Think of a solution to enable it.
    export ENABLE_CLUSTER_MONITORING="none"
    export E2E_ZONE="us-east1-b"
    export MASTER_SIZE="n1-standard-32"
    export NODE_SIZE="n1-standard-1"
    export NODE_DISK_SIZE="50GB"
    export NUM_NODES="1000"
    # Reduce logs verbosity
    export TEST_CLUSTER_LOG_LEVEL="--v=1"
    # Increase resync period to simulate production
    export TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Upgrade jobs

  # kubernetes-upgrade-gke-1.0-master
  #
  # Test upgrades from the latest release-1.0 build to the latest master build.
  #
  # Configurations for step1, step4, and step6 live in the release-1.0 branch.

  kubernetes-upgrade-gke-1.0-master-step2-kubectl-e2e-new)
    configure_upgrade_step 'configured-in-release-1.0' 'ci/latest' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.0-master-step3-upgrade-master)
    configure_upgrade_step 'configured-in-release-1.0' 'ci/latest' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.0-master-step5-upgrade-cluster)
    configure_upgrade_step 'configured-in-release-1.0' 'ci/latest' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.0-master-step7-e2e-new)
    configure_upgrade_step 'configured-in-release-1.0' 'ci/latest' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  # kubernetes-upgrade-gke-1.1-master
  #
  # Test upgrades from the latest release-1.1 build to the latest master build.
  #
  # Configurations for step1, step4, and step6 live in the release-1.1 branch.

  kubernetes-upgrade-gke-1.1-master-step2-kubectl-e2e-new)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gke-1-1-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.1-master-step3-upgrade-master)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gke-1-1-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.1-master-step5-upgrade-cluster)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gke-1-1-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.1-master-step7-e2e-new)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gke-1-1-master' 'kubernetes-jenkins-gke-upgrade'
    ;;

  # kubernetes-upgrade-gce-1.1-master
  #
  # Test upgrades from the latest release-1.1 build to the latest master build.
  #
  # Configurations for step1, step4, and step6 live in the release-1.1 branch.

  kubernetes-upgrade-gce-1.1-master-step2-kubectl-e2e-new)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gce-1-1-master' 'k8s-jkns-gce-upgrade'
    ;;

  kubernetes-upgrade-gce-1.1-master-step3-upgrade-master)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gce-1-1-master' 'k8s-jkns-gce-upgrade'
    ;;

  kubernetes-upgrade-gce-1.1-master-step5-upgrade-cluster)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gce-1-1-master' 'k8s-jkns-gce-upgrade'
    ;;

  kubernetes-upgrade-gce-1.1-master-step7-e2e-new)
    configure_upgrade_step 'configured-in-release-1.1' 'ci/latest' 'upgrade-gce-1-1-master' 'k8s-jkns-gce-upgrade'
    ;;
esac

# Skip gcloud update checking
export CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=true

# AWS variables
export KUBE_AWS_INSTANCE_PREFIX=${E2E_CLUSTER_NAME}
export KUBE_AWS_ZONE=${E2E_ZONE}
export AWS_CONFIG_FILE=${AWS_CONFIG_FILE:-}
export AWS_SSH_KEY=${AWS_SSH_KEY:-}
export KUBE_SSH_USER=${KUBE_SSH_USER:-}
export AWS_SHARED_CREDENTIALS_FILE=${AWS_SHARED_CREDENTIALS_FILE:-}

# GCE variables
export INSTANCE_PREFIX=${E2E_CLUSTER_NAME}
export KUBE_GCE_ZONE=${E2E_ZONE}
export KUBE_GCE_NETWORK=${E2E_NETWORK}
export KUBE_GCE_INSTANCE_PREFIX=${KUBE_GCE_INSTANCE_PREFIX:-}
export KUBE_GCE_NODE_PROJECT=${KUBE_GCE_NODE_PROJECT:-}
export KUBE_GCE_NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-}
export KUBE_OS_DISTRIBUTION=${KUBE_OS_DISTRIBUTION:-}
export GCE_SERVICE_ACCOUNT=$(gcloud auth list 2> /dev/null | grep active | cut -f3 -d' ')
export FAIL_ON_GCP_RESOURCE_LEAK="${FAIL_ON_GCP_RESOURCE_LEAK:-false}"
export ALLOWED_NOTREADY_NODES=${ALLOWED_NOTREADY_NODES:-}
export EXIT_ON_WEAK_ERROR=${EXIT_ON_WEAK_ERROR:-}
export HAIRPIN_MODE=${HAIRPIN_MODE:-}

# GKE variables
export CLUSTER_NAME=${E2E_CLUSTER_NAME}
export ZONE=${E2E_ZONE}
export KUBE_GKE_NETWORK=${E2E_NETWORK}
export E2E_SET_CLUSTER_API_VERSION=${E2E_SET_CLUSTER_API_VERSION:-}
export CMD_GROUP=${CMD_GROUP:-}
export MACHINE_TYPE=${NODE_SIZE:-}  # GKE scripts use MACHINE_TYPE for the node vm size
export CLOUDSDK_BUCKET="${CLOUDSDK_BUCKET:-}"

if [[ ! -z "${GKE_API_ENDPOINT:-}" ]]; then
  export CLOUDSDK_API_ENDPOINT_OVERRIDES_CONTAINER=${GKE_API_ENDPOINT}
fi

# Shared cluster variables
export E2E_MIN_STARTUP_PODS=${E2E_MIN_STARTUP_PODS:-}
export KUBE_ENABLE_CLUSTER_MONITORING=${ENABLE_CLUSTER_MONITORING:-}
export KUBE_ENABLE_CLUSTER_REGISTRY=${ENABLE_CLUSTER_REGISTRY:-}
export KUBE_ENABLE_HORIZONTAL_POD_AUTOSCALER=${ENABLE_HORIZONTAL_POD_AUTOSCALER:-}
export KUBE_ENABLE_DEPLOYMENTS=${ENABLE_DEPLOYMENTS:-}
export KUBE_ENABLE_EXPERIMENTAL_API=${ENABLE_EXPERIMENTAL_API:-}
export MASTER_SIZE=${MASTER_SIZE:-}
export NODE_SIZE=${NODE_SIZE:-}
export NODE_DISK_SIZE=${NODE_DISK_SIZE:-}
export NUM_NODES=${NUM_NODES:-}
export TEST_CLUSTER_LOG_LEVEL=${TEST_CLUSTER_LOG_LEVEL:-}
export KUBELET_TEST_LOG_LEVEL=${KUBELET_TEST_LOG_LEVEL:-}
export TEST_CLUSTER_RESYNC_PERIOD=${TEST_CLUSTER_RESYNC_PERIOD:-}
export PROJECT=${PROJECT:-}
export NETWORK_PROVIDER=${NETWORK_PROVIDER:-}
export JENKINS_PUBLISHED_VERSION=${JENKINS_PUBLISHED_VERSION:-'ci/latest'}

export KUBE_ADMISSION_CONTROL=${ADMISSION_CONTROL:-}

export KUBERNETES_PROVIDER=${KUBERNETES_PROVIDER}
export PATH=${PATH}:/usr/local/go/bin
export KUBE_SKIP_UPDATE=y
export KUBE_SKIP_CONFIRMATIONS=y

# Kubemark
export USE_KUBEMARK="${USE_KUBEMARK:-false}"
export KUBEMARK_TESTS="${KUBEMARK_TESTS:-}"
export KUBEMARK_MASTER_SIZE="${KUBEMARK_MASTER_SIZE:-$MASTER_SIZE}"
export KUBEMARK_NUM_NODES="${KUBEMARK_NUM_NODES:-$NUM_NODES}"

# E2E Control Variables
export E2E_OPT="${E2E_OPT:-}"
export E2E_UP="${E2E_UP:-true}"
export E2E_TEST="${E2E_TEST:-true}"
export E2E_DOWN="${E2E_DOWN:-true}"
export E2E_CLEAN_START="${E2E_CLEAN_START:-}"
export E2E_PUBLISH_GREEN_VERSION="${E2E_PUBLISH_GREEN_VERSION:-false}"
# Used by hack/ginkgo-e2e.sh to enable ginkgo's parallel test runner.
export GINKGO_PARALLEL=${GINKGO_PARALLEL:-}
export GINKGO_PARALLEL_NODES=${GINKGO_PARALLEL_NODES:-}
export GINKGO_TEST_ARGS="${GINKGO_TEST_ARGS:-}"

# If we are on PR Jenkins merging into master, use the local e2e.sh. Otherwise, use the latest on github.
if [[ "${ghprbTargetBranch:-}" == "master" ]]; then
    source "hack/jenkins/e2e-runner.sh"
else
    source <(curl -fsS --retry 3 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/e2e-runner.sh")
fi
