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

  # PR builder

  # Runs a subset of tests on GCE in parallel. Run against all pending PRs.
  kubernetes-pull-build-test-e2e-gce)
    : ${E2E_CLUSTER_NAME:="jnks-e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${E2E_NETWORK:="e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${GINKGO_PARALLEL:="y"}
    # This list should match the list in kubernetes-e2e-gce.
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${PROJECT:="kubernetes-jenkins-pull"}
    # Override GCE defaults
    NUM_NODES=${NUM_NODES_PARALLEL}
    ;;

  # GCE core jobs

  # Runs all non-slow, non-serial, non-flaky, tests on GCE in parallel.
  kubernetes-e2e-gce)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e"}
    : ${E2E_PUBLISH_GREEN_VERSION:="true"}
    : ${E2E_NETWORK:="e2e-gce"}
    # This list should match the list in kubernetes-pull-build-test-e2e-gce.
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${GINKGO_PARALLEL:="y"}
    : ${KUBE_GCE_INSTANCE_PREFIX="e2e-gce"}
    : ${PROJECT:="k8s-jkns-e2e-gce"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    ;;

  # Runs slow tests on GCE, sequentially.
  kubernetes-e2e-gce-slow)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-slow"}
    : ${E2E_NETWORK:="e2e-slow"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Slow\] \
                           --ginkgo.skip=\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${GINKGO_PARALLEL:="y"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-slow"}
    : ${PROJECT:="k8s-jkns-e2e-gce-slow"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    ;;

  # Run the [Serial], [Disruptive], and [Feature:Restart] tests on GCE.
  kubernetes-e2e-gce-serial)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-serial"}
    : ${E2E_NETWORK:="jenkins-gce-e2e-serial"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Serial\]|\[Disruptive\] \
                           --ginkgo.skip=\[Flaky\]|\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-serial"}
    : ${PROJECT:="kubernetes-jkns-e2e-gce-serial"}
    ;;

  # Runs only the ingress tests on GCE.
  kubernetes-e2e-gce-ingress)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-ingress"}
    : ${E2E_NETWORK:="e2e-ingress"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Ingress\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-ingress"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    # TODO: Move this into a different project. Currently, since this test
    # shares resources with various other networking tests, so it's easier
    # to zero in on the source of a leak if it's run in isolation.
    : ${PROJECT:="kubernetes-flannel"}
    ;;

  # Runs only the ingress tests on GKE.
  kubernetes-e2e-gke-ingress)
    : ${E2E_CLUSTER_NAME:="jenkins-gke-e2e-ingress"}
    : ${E2E_NETWORK:="e2e-gke-ingress"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Ingress\]"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-gke-ingress"}
    # TODO: Move this into a different project. Currently, since this test
    # shares resources with various other networking tests, it's easier to
    # zero in on the source of a leak if it's run in isolation.
    : ${PROJECT:="kubernetes-flannel"}
    ;;

  # Runs the flaky tests on GCE, sequentially.
  kubernetes-e2e-gce-flaky)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-flaky"}
    : ${E2E_NETWORK:="e2e-flaky"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Flaky\] \
                           --ginkgo.skip=\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-flaky"}
    : ${PROJECT:="k8s-jkns-e2e-gce-flaky"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${E2E_DOWN:="true"}
    ;;

  # Runs the flaky tests on GCE in parallel.
  kubernetes-e2e-gce-parallel-flaky)
    : ${E2E_CLUSTER_NAME:="parallel-flaky"}
    : ${E2E_NETWORK:="e2e-parallel-flaky"}
    : ${GINKGO_PARALLEL:="y"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Flaky\] \
                           --ginkgo.skip=\[Serial\]|\[Disruptive\]|\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="parallel-flaky"}
    : ${PROJECT:="k8s-jkns-e2e-gce-prl-flaky"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    # Override GCE defaults.
    NUM_NODES=${NUM_NODES_PARALLEL}
    ;;

  # GKE core jobs

  # Runs all non-slow, non-serial, non-flaky, tests on GKE in parallel.
  kubernetes-e2e-gke)
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-ci"}
    : ${E2E_NETWORK:="e2e-gke-ci"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-ci"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${GINKGO_PARALLEL:="y"}
    ;;

  kubernetes-e2e-gke-slow)
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-slow"}
    : ${E2E_NETWORK:="e2e-gke-slow"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-slow"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Slow\] \
                           --ginkgo.skip=\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${GINKGO_PARALLEL:="y"}
    ;;

  # Run the [Serial], [Disruptive], and [Feature:Restart] tests on GKE.
  kubernetes-e2e-gke-serial)
    : ${E2E_CLUSTER_NAME:="jenkins-gke-e2e-serial"}
    : ${E2E_NETWORK:="jenkins-gke-e2e-serial"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Serial\]|\[Disruptive\] \
                           --ginkgo.skip=\[Flaky\]|\[Feature:.+\]"}
    : ${PROJECT:="jenkins-gke-e2e-serial"}
    ;;

  kubernetes-e2e-gke-flaky)
    : ${E2E_CLUSTER_NAME:="kubernetes-gke-e2e-flaky"}
    : ${E2E_NETWORK:="gke-e2e-flaky"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-ci-flaky"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Flaky\] \
                           --ginkgo.skip=\[Feature:.+\]"}
    ;;

  # AWS core jobs

  # Runs all non-flaky, non-slow tests on AWS, sequentially.
  kubernetes-e2e-aws)
    : ${E2E_PUBLISH_GREEN_VERSION:=true}
    : ${E2E_CLUSTER_NAME:="jenkins-aws-e2e"}
    : ${E2E_ZONE:="us-west-2a"}
    : ${ZONE:="us-west-2a"}
    : ${E2E_NETWORK:="e2e-aws"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${GINKGO_PARALLEL:="y"}
    : ${KUBE_GCE_INSTANCE_PREFIX="e2e-aws"}
    : ${PROJECT:="k8s-jkns-e2e-aws"}
    : ${AWS_CONFIG_FILE:='/var/lib/jenkins/.aws/credentials'}
    : ${AWS_SSH_KEY:='/var/lib/jenkins/.ssh/kube_aws_rsa'}
    : ${KUBE_SSH_USER:='ubuntu'}
    # This is needed to be able to create PD from the e2e test
    : ${AWS_SHARED_CREDENTIALS_FILE:='/var/lib/jenkins/.aws/credentials'}
    ;;

  # Feature jobs

  # Runs only the reboot tests on GCE.
  kubernetes-e2e-gce-reboot)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-reboot"}
    : ${E2E_NETWORK:="e2e-reboot"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Reboot\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-reboot"}
    : ${PROJECT:="kubernetes-jenkins"}
  ;;

  kubernetes-e2e-gke-reboot)
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-ci-reboot"}
    : ${E2E_NETWORK:="e2e-gke-ci-reboot"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-ci-reboot"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Reboot\]"}
  ;;

  # Runs only the examples tests on GCE.
  kubernetes-e2e-gce-examples)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-examples"}
    : ${E2E_NETWORK:="e2e-examples"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Example\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-examples"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Runs only the autoscaling tests on GCE.
  kubernetes-e2e-gce-autoscaling)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-autoscaling"}
    : ${E2E_NETWORK:="e2e-autoscaling"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:ClusterSizeAutoscaling\]|\[Feature:InitialResources\] \
                           --ginkgo.skip=\[Flaky\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-autoscaling"}
    : ${PROJECT:="k8s-jnks-e2e-gce-autoscaling"}
    : ${FAIL_ON_GCP_RESOURCE_LEAK:="true"}
    # Override GCE default for cluster size autoscaling purposes.
    ENABLE_CLUSTER_MONITORING="googleinfluxdb"
    ADMISSION_CONTROL="NamespaceLifecycle,InitialResources,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota"
    ;;

  # Runs the performance/scalability tests on GCE. A larger cluster is used.
  kubernetes-e2e-gce-scalability)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-scalability"}
    : ${E2E_NETWORK:="e2e-scalability"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Performance\] \
        --gather-resource-usage=true \
        --gather-metrics-at-teardown=true \
        --gather-logs-sizes=true \
        --output-print-type=json"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-scalability"}
    : ${PROJECT:="kubernetes-jenkins"}
    # Override GCE defaults.
    MASTER_SIZE="n1-standard-4"
    NODE_SIZE="n1-standard-2"
    NODE_DISK_SIZE="50GB"
    NUM_NODES="100"
    # Reduce logs verbosity
    TEST_CLUSTER_LOG_LEVEL="--v=2"
    # TODO: Remove when we figure out the reason for ocassional failures #19048
    KUBELET_TEST_LOG_LEVEL="--v=4"
    # Increase resync period to simulate production
    TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Runs e2e on GCE with flannel and VXLAN.
  kubernetes-e2e-gce-flannel)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-flannel"}
    : ${E2E_PUBLISH_GREEN_VERSION:="true"}
    : ${E2E_NETWORK:="e2e-gce-flannel"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-flannel"}
    : ${PROJECT:="kubernetes-flannel"}
    # Override GCE defaults.
    NETWORK_PROVIDER="flannel"
    ;;

  # Runs the performance/scalability test on huge 1000-node cluster on GCE.
  # Flannel is used as network provider.
  # Allows a couple of nodes to be NotReady during startup
  kubernetes-e2e-gce-enormous-cluster)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-enormous-cluster"}
    : ${E2E_NETWORK:="e2e-enormous-cluster"}
    # TODO: Currently run only density test.
    # Once this is stable, run the whole [Performance] suite.
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=starting\s30\spods\sper\snode"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-enormous-cluster"}
    : ${PROJECT:="kubernetes-scale"}
    # Override GCE defaults.
    NETWORK_PROVIDER="flannel"
    # Temporarily switch of Heapster, as this will not schedule anywhere.
    # TODO: Think of a solution to enable it.
    ENABLE_CLUSTER_MONITORING="none"
    E2E_ZONE="asia-east1-a"
    MASTER_SIZE="n1-standard-32"
    NODE_SIZE="n1-standard-1"
    NODE_DISK_SIZE="50GB"
    NUM_NODES="1000"
    ALLOWED_NOTREADY_NODES="2"
    EXIT_ON_WEAK_ERROR="false"
    # Reduce logs verbosity
    TEST_CLUSTER_LOG_LEVEL="--v=1"
    # Increase resync period to simulate production
    TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Starts and tears down 1000-node cluster on GCE using flannel networking
  # Requires all 1000 nodes to come up.
  kubernetes-e2e-gce-enormous-startup)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-enormous-startup"}
    # TODO: increase a quota for networks in kubernetes-scale and move this test to its own network
    : ${E2E_NETWORK:="e2e-enormous-cluster"}
    : ${E2E_TEST:="false"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-enormous-startup"}
    : ${PROJECT:="kubernetes-scale"}
    # Override GCE defaults.
    NETWORK_PROVIDER="flannel"
    # Temporarily switch of Heapster, as this will not schedule anywhere.
    # TODO: Think of a solution to enable it.
    ENABLE_CLUSTER_MONITORING="none"
    E2E_ZONE="us-east1-b"
    MASTER_SIZE="n1-standard-32"
    NODE_SIZE="n1-standard-1"
    NODE_DISK_SIZE="50GB"
    NUM_NODES="1000"
    # Reduce logs verbosity
    TEST_CLUSTER_LOG_LEVEL="--v=1"
    # Increase resync period to simulate production
    TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Run Kubemark test on a fake 100 node cluster to have a comparison
  # to the real results from scalability suite
  kubernetes-kubemark-gce)
    : ${E2E_CLUSTER_NAME:="kubernetes-kubemark"}
    : ${E2E_NETWORK:="kubernetes-kubemark"}
    : ${PROJECT:="k8s-jenkins-kubemark"}
    : ${E2E_UP:="true"}
    : ${E2E_DOWN:="true"}
    : ${E2E_TEST:="false"}
    : ${USE_KUBEMARK:="true"}
    : ${KUBEMARK_TESTS:="should\sallow\sstarting\s30\spods\sper\snode"}
    # Override defaults to be indpendent from GCE defaults and set kubemark parameters
    KUBE_GCE_INSTANCE_PREFIX="kubemark100"
    NUM_NODES="10"
    MASTER_SIZE="n1-standard-2"
    NODE_SIZE="n1-standard-1"
    E2E_ZONE="asia-east1-a"
    KUBEMARK_MASTER_SIZE="n1-standard-4"
    KUBEMARK_NUM_NODES="100"
    ;;

  # Run Kubemark test on a fake 500 node cluster to test for regressions on
  # bigger clusters
  kubernetes-kubemark-500-gce)
    : ${E2E_CLUSTER_NAME:="kubernetes-kubemark-500"}
    : ${E2E_NETWORK:="kubernetes-kubemark-500"}
    : ${PROJECT:="kubernetes-scale"}
    : ${E2E_UP:="true"}
    : ${E2E_DOWN:="true"}
    : ${E2E_TEST:="false"}
    : ${USE_KUBEMARK:="true"}
    : ${KUBEMARK_TESTS:="\[Feature:Performance\]"}
    # Override defaults to be indpendent from GCE defaults and set kubemark parameters
    NUM_NODES="6"
    MASTER_SIZE="n1-standard-4"
    NODE_SIZE="n1-standard-8"
    KUBE_GCE_INSTANCE_PREFIX="kubemark500"
    E2E_ZONE="us-east1-b"
    KUBEMARK_MASTER_SIZE="n1-standard-16"
    KUBEMARK_NUM_NODES="500"
    ;;

  # Run big Kubemark test, this currently means a 1000 node cluster and 16 core master
  kubernetes-kubemark-gce-scale)
    : ${E2E_CLUSTER_NAME:="kubernetes-kubemark-scale"}
    : ${E2E_NETWORK:="kubernetes-kubemark-scale"}
    : ${PROJECT:="kubernetes-scale"}
    : ${E2E_UP:="true"}
    : ${E2E_DOWN:="true"}
    : ${E2E_TEST:="false"}
    : ${USE_KUBEMARK:="true"}
    : ${KUBEMARK_TESTS:="\[Feature:Performance\]"}
    # Override defaults to be indpendent from GCE defaults and set kubemark parameters
    # We need 11 so that we won't hit max-pods limit (set to 100). TODO: do it in a nicer way.
    NUM_NODES="11"
    MASTER_SIZE="n1-standard-4"
    NODE_SIZE="n1-standard-8"   # Note: can fit about 17 hollow nodes per core
    #                                     so NUM_NODES x cores_per_node should
    #                                     be set accordingly.
    KUBE_GCE_INSTANCE_PREFIX="kubemark1000"
    E2E_ZONE="us-east1-b"
    KUBEMARK_MASTER_SIZE="n1-standard-16"
    KUBEMARK_NUM_NODES="1000"
    ;;

  # Soak jobs

  # Sets up the GCE soak cluster weekly using the latest CI release.
  kubernetes-soak-weekly-deploy-gce)
    : ${E2E_CLUSTER_NAME:="gce-soak-weekly"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="gce-soak-weekly"}
    : ${E2E_TEST:="false"}
    : ${E2E_UP:="true"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="gce-soak-weekly"}
    : ${HAIRPIN_MODE:="false"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Runs tests on GCE soak cluster.
  kubernetes-soak-continuous-e2e-gce)
    : ${E2E_CLUSTER_NAME:="gce-soak-weekly"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="gce-soak-weekly"}
    : ${E2E_UP:="false"}
    # Clear out any orphaned namespaces in case previous run was interrupted.
    : ${E2E_CLEAN_START:="true"}
    # We should be testing the reliability of a long-running cluster. The
    # [Disruptive] tests kill/restart components or nodes in the cluster,
    # defeating the purpose of a soak cluster. (#15722)
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="gce-soak-weekly"}
    : ${HAIRPIN_MODE:="false"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Clone of kubernetes-soak-weekly-deploy-gce. Issue #20832.
  kubernetes-soak-weekly-deploy-gce-2)
    : ${E2E_CLUSTER_NAME:="gce-soak-weekly-2"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="gce-soak-weekly-2"}
    : ${E2E_TEST:="false"}
    : ${E2E_UP:="true"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="gce-soak-weekly-2"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Clone of kubernetes-soak-continuous-e2e-gce. Issue #20832.
  kubernetes-soak-continuous-e2e-gce-2)
    : ${E2E_CLUSTER_NAME:="gce-soak-weekly-2"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="gce-soak-weekly-2"}
    : ${E2E_UP:="false"}
    # Clear out any orphaned namespaces in case previous run was interrupted.
    : ${E2E_CLEAN_START:="true"}
    # We should be testing the reliability of a long-running cluster. The
    # [Disruptive] tests kill/restart components or nodes in the cluster,
    # defeating the purpose of a soak cluster. (#15722)
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="gce-soak-weekly-2"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Sets up the GKE soak cluster weekly using the latest CI release.
  kubernetes-soak-weekly-deploy-gke)
    : ${E2E_CLUSTER_NAME:="jenkins-gke-soak-weekly"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="gke-soak-weekly"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${JENKINS_PUBLISHED_VERSION:="ci/latest"}
    : ${E2E_TEST:="false"}
    : ${E2E_UP:="true"}
    : ${PROJECT:="kubernetes-jenkins"}
    # Need at least n1-standard-2 nodes to run kubelet_perf tests
    NODE_SIZE="n1-standard-2"
    ;;

  # Runs tests on GKE soak cluster.
  kubernetes-soak-continuous-e2e-gke)
    : ${E2E_CLUSTER_NAME:="jenkins-gke-soak-weekly"}
    : ${E2E_NETWORK:="gke-soak-weekly"}
    : ${E2E_DOWN:="false"}
    : ${E2E_UP:="false"}
    # Clear out any orphaned namespaces in case previous run was interrupted.
    : ${E2E_CLEAN_START:="true"}
    : ${PROJECT:="kubernetes-jenkins"}
    : ${E2E_OPT:="--check_version_skew=false"}
    # We should be testing the reliability of a long-running cluster. The
    # [Disruptive] tests kill/restart components or nodes in the cluster,
    # defeating the purpose of a soak cluster. (#15722)
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"}
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

  # kubernetes-upgrade-gce-1.0-current-release
  #
  # This suite:
  #
  # 1. launches a cluster at ci/latest-1.0,
  # 2. upgrades the master to CURRENT_RELEASE_PUBLISHED_VERSION
  # 3. runs ci/latest-1.0 e2es,
  # 4. upgrades the rest of the cluster,
  # 5. runs ci/latest-1.0 e2es again, then
  # 6. runs CURRENT_RELEASE_PUBLISHED_VERSION e2es and tears down the cluster.

  kubernetes-upgrade-1.0-current-release-gce-step1-deploy)
    : ${E2E_CLUSTER_NAME:="gce-upgrade-1-0"}
    : ${E2E_NETWORK:="gce-upgrade-1-0"}
    : ${JENKINS_PUBLISHED_VERSION:="ci/latest-1.0"}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="true"}
    : ${E2E_TEST:="false"}
    : ${E2E_DOWN:="false"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-upgrade-1-0"}
    : ${NUM_NODES:=5}
    ;;

  kubernetes-upgrade-1.0-current-release-gce-step2-upgrade-master)
    : ${E2E_CLUSTER_NAME:="gce-upgrade-1-0"}
    : ${E2E_NETWORK:="gce-upgrade-1-0"}
    : ${E2E_OPT:="--check_version_skew=false"}
    # Use upgrade logic of version we're upgrading to.
    : ${JENKINS_PUBLISHED_VERSION:="${CURRENT_RELEASE_PUBLISHED_VERSION}"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="false"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Upgrade\].*upgrade-master --upgrade-target=${CURRENT_RELEASE_PUBLISHED_VERSION}"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-upgrade-1-0"}
    : ${NUM_NODES:=5}
    : ${KUBE_ENABLE_DAEMONSETS:=true}
    ;;

  kubernetes-upgrade-1.0-current-release-gce-step3-e2e-old)
    : ${E2E_CLUSTER_NAME:="gce-upgrade-1-0"}
    : ${E2E_NETWORK:="gce-upgrade-1-0"}
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    # Run old e2es
    : ${JENKINS_PUBLISHED_VERSION:="ci/latest-1.0"}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="false"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-upgrade-1-0"}
    : ${NUM_NODES:=5}
    ;;

  kubernetes-upgrade-1.0-current-release-gce-step4-upgrade-cluster)
    : ${E2E_CLUSTER_NAME:="gce-upgrade-1-0"}
    : ${E2E_NETWORK:="gce-upgrade-1-0"}
    : ${E2E_OPT:="--check_version_skew=false"}
    # Use upgrade logic of version we're upgrading to.
    : ${JENKINS_PUBLISHED_VERSION:="${CURRENT_RELEASE_PUBLISHED_VERSION}"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="false"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Feature:Upgrade\].*upgrade-cluster --upgrade-target=${CURRENT_RELEASE_PUBLISHED_VERSION}"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-upgrade-1-0"}
    : ${NUM_NODES:=5}
    : ${KUBE_ENABLE_DAEMONSETS:=true}
    ;;

  kubernetes-upgrade-1.0-current-release-gce-step5-e2e-old)
    : ${E2E_CLUSTER_NAME:="gce-upgrade-1-0"}
    : ${E2E_NETWORK:="gce-upgrade-1-0"}
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    # Run old e2es
    : ${JENKINS_PUBLISHED_VERSION:="ci/latest-1.0"}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="false"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-upgrade-1-0"}
    : ${NUM_NODES:=5}
    ;;

  kubernetes-upgrade-1.0-current-release-gce-step6-e2e-new)
    : ${E2E_CLUSTER_NAME:="gce-upgrade-1-0"}
    : ${E2E_NETWORK:="gce-upgrade-1-0"}
    # TODO(15011): these really shouldn't be (very) version skewed, but because
    # we have to get CURRENT_RELEASE_PUBLISHED_VERSION again, it could get slightly out of whack.
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    : ${JENKINS_PUBLISHED_VERSION:="${CURRENT_RELEASE_PUBLISHED_VERSION}"}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="true"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-upgrade-1-0"}
    : ${NUM_NODES:=5}
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
