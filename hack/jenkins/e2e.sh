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

echo "--------------------------------------------------------------------------------"
echo "Initial Environment:"
printenv | sort
echo "--------------------------------------------------------------------------------"

# Nothing should want Jenkins $HOME
export HOME=${WORKSPACE}

# Set environment variables based on provider
if [[ ${JOB_NAME} =~ ^kubernetes-.*-gce ]]; then
  export KUBERNETES_PROVIDER="gce"
  export E2E_MIN_STARTUP_PODS="1"
  export KUBE_GCE_ZONE="us-central1-f"
  export FAIL_ON_GCP_RESOURCE_LEAK="true"
elif [[ ${JOB_NAME} =~ ^kubernetes-.*-gke ]]; then
  export KUBERNETES_PROVIDER="gke"
  export ZONE="us-central1-f"
  # By default, GKE tests run against the GKE test endpoint using CI Cloud SDK.
  # Release jobs (e.g. prod, staging, and test) override these two variables.
  export CLOUDSDK_BUCKET="gs://cloud-sdk-build/testing/staging"
  export CLOUDSDK_API_ENDPOINT_OVERRIDES_CONTAINER="https://test-container.sandbox.googleapis.com/"
  export FAIL_ON_GCP_RESOURCE_LEAK="true"
elif [[ ${JOB_NAME} =~ ^kubernetes-.*-aws ]]; then
  export KUBERNETES_PROVIDER="aws"
  export E2E_MIN_STARTUP_PODS="1"
  export KUBE_AWS_ZONE="us-west-2a"
  export MASTER_SIZE="m3.medium"
  export NODE_SIZE="m3.medium"
  export NUM_NODES="3"
fi

# Set environment variables based on soak jobs
if [[ ${JOB_NAME} =~ soak-weekly ]]; then
  export FAIL_ON_GCP_RESOURCE_LEAK="false"
  export E2E_TEST="false"
  export E2E_DOWN="false"
elif [[ ${JOB_NAME} =~ soak-continuous ]]; then
  export FAIL_ON_GCP_RESOURCE_LEAK="false"
  export E2E_UP="false"
  export E2E_DOWN="false"
  # Clear out any orphaned namespaces in case previous run was interrupted.
  export E2E_CLEAN_START="true"
  # We should be testing the reliability of a long-running cluster. The
  # [Disruptive] tests kill/restart components or nodes in the cluster,
  # defeating the purpose of a soak cluster. (#15722)
  export GINKGO_TEST_ARGS="--ginkgo.skip=\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
fi

# Set environment variables based on upgrade jobs
if [[ ${JOB_NAME} =~ ^kubernetes-upgrade ]]; then
  # Upgrade jobs bounce back and forth between versions; just force
  # it to always get the tars of the version it wants to test.
  export JENKINS_FORCE_GET_TARS="y"
  export FAIL_ON_GCP_RESOURCE_LEAK="false"
  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
    export NUM_NODES=5
  fi
  if [[ "${JOB_NAME}" =~ step1 ]]; then
    export E2E_TEST="false"
    export E2E_DOWN="false"
  elif [[ "${JOB_NAME}" =~ step2 ]]; then
    export E2E_OPT="--check_version_skew=false"
    export E2E_UP="false"
    export E2E_DOWN="false"
    export GINKGO_TEST_ARGS="--ginkgo.focus=Kubectl"
  elif [[ "${JOB_NAME}" =~ step3 ]]; then
    export E2E_OPT="--check_version_skew=false"
    export E2E_UP="false"
    export E2E_DOWN="false"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Upgrade\].*upgrade-master"
  elif [[ "${JOB_NAME}" =~ step4 ]]; then
    export E2E_OPT="--check_version_skew=false"
    export E2E_UP="false"
    export E2E_DOWN="false"
  elif [[ "${JOB_NAME}" =~ step5 ]]; then
    export E2E_OPT="--check_version_skew=false"
    export E2E_UP="false"
    export E2E_DOWN="false"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Upgrade\].*upgrade-cluster"
  elif [[ "${JOB_NAME}" =~ step6 ]]; then
    export E2E_OPT="--check_version_skew=false"
    export E2E_UP="false"
    export E2E_DOWN="false"
  elif [[ "${JOB_NAME}" =~ step7 ]]; then
    # TODO(15011): these really shouldn't be (very) version skewed, but
    # because we have to get ci/latest again, it could get slightly out of
    # whack.
    export E2E_OPT="--check_version_skew=false"
    export E2E_UP="false"
  fi
fi

# Define environment variables based on the Jenkins project name.
# NOTE: Not all jobs are defined here. The hack/jenkins/e2e.sh in master and
# release branches defines relevant jobs for that particular version of
# Kubernetes.
case ${JOB_NAME} in

  # PR builder

  # Runs a subset of tests on GCE in parallel. Run against all pending PRs.
  kubernetes-pull-build-test-e2e-gce)
    # XXX Not a unique project
    export E2E_NAME="e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"
    export GINKGO_PARALLEL="y"
    # This list should match the list in kubernetes-e2e-gce.
    export GINKGO_TEST_ARGS="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    export PROJECT="kubernetes-jenkins-pull"
    # Override GCE defaults
    export NUM_NODES="6"
    ;;

  # GCE core jobs

  # Runs all non-slow, non-serial, non-flaky, tests on GCE in parallel.
  kubernetes-e2e-gce)
    # This is the *only* job that should publish the last green version.
    export E2E_PUBLISH_GREEN_VERSION="true"
    # This list should match the list in kubernetes-pull-build-test-e2e-gce.
    export GINKGO_TEST_ARGS="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
    export GINKGO_PARALLEL="y"
    export PROJECT="k8s-jkns-e2e-gce"
    ;;

  # Runs slow tests on GCE, sequentially.
  kubernetes-e2e-gce-slow)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Slow\] \
                             --ginkgo.skip=\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
    export GINKGO_PARALLEL="y"
    export PROJECT="k8s-jkns-e2e-gce-slow"
    ;;
  
  # Runs all non-flaky, non-slow tests on GCE, sequentially,
  # and in a multi-zone ("Ubernetes Lite") cluster.
  kubernetes-e2e-gce-ubernetes-lite)
    export PROJECT="k8s-jkns-e2e-gce-ubelite"
    export E2E_MULTIZONE="true"
    export KUBE_GCE_ZONE=""
    export E2E_ZONES="us-central1-a us-central1-b us-central1-f"}
    ;;

  # Run the [Serial], [Disruptive], and [Feature:Restart] tests on GCE.
  kubernetes-e2e-gce-serial)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Serial\]|\[Disruptive\] \
                             --ginkgo.skip=\[Flaky\]|\[Feature:.+\]"
    export PROJECT="kubernetes-jkns-e2e-gce-serial"
    ;;

  # Runs only the ingress tests on GCE.
  kubernetes-e2e-gce-ingress)
    # XXX Not a unique project
    export E2E_NAME="e2e-ingress"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Ingress\]"
    # TODO: Move this into a different project. Currently, since this test
    # shares resources with various other networking tests, so it's easier
    # to zero in on the source of a leak if it's run in isolation.
    export PROJECT="kubernetes-flannel"
    ;;

  # Runs only the ingress tests on GKE.
  kubernetes-e2e-gke-ingress)
    # XXX Not a unique project
    export E2E_NAME="e2e-gke-ingress"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Ingress\]"
    # TODO: Move this into a different project. Currently, since this test
    # shares resources with various other networking tests, it's easier to
    # zero in on the source of a leak if it's run in isolation.
    export PROJECT="kubernetes-flannel"
    ;;

  # Runs the flaky tests on GCE, sequentially.
  kubernetes-e2e-gce-flaky)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Flaky\] \
                             --ginkgo.skip=\[Feature:.+\]"
    export PROJECT="k8s-jkns-e2e-gce-flaky"
    ;;

  # GKE core jobs

  # Runs all non-slow, non-serial, non-flaky, tests on GKE in parallel.
  kubernetes-e2e-gke)
    export PROJECT="k8s-jkns-e2e-gke-ci"
    export GINKGO_TEST_ARGS="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
    export GINKGO_PARALLEL="y"
    ;;

  kubernetes-e2e-gke-slow)
    export PROJECT="k8s-jkns-e2e-gke-slow"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Slow\] \
                             --ginkgo.skip=\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
    export GINKGO_PARALLEL="y"
    ;;

  # Run the [Serial], [Disruptive], and [Feature:Restart] tests on GKE.
  kubernetes-e2e-gke-serial)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Serial\]|\[Disruptive\] \
                             --ginkgo.skip=\[Flaky\]|\[Feature:.+\]"
    export PROJECT="jenkins-gke-e2e-serial"
    ;;

  kubernetes-e2e-gke-flaky)
    export PROJECT="k8s-jkns-e2e-gke-ci-flaky"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Flaky\] \
                             --ginkgo.skip=\[Feature:.+\]"
    ;;

  # AWS core jobs

  # Runs all non-flaky, non-slow tests on AWS, sequentially.
  kubernetes-e2e-aws)
    export GINKGO_TEST_ARGS="--ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]"
    export GINKGO_PARALLEL="y"
    export PROJECT="k8s-jkns-e2e-aws"
    export AWS_CONFIG_FILE='/var/lib/jenkins/.aws/credentials'
    export AWS_SSH_KEY='/var/lib/jenkins/.ssh/kube_aws_rsa'
    export KUBE_SSH_USER='ubuntu'
    # This is needed to be able to create PD from the e2e test
    export AWS_SHARED_CREDENTIALS_FILE='/var/lib/jenkins/.aws/credentials'
    ;;

  # Feature jobs

  # Runs only the reboot tests on GCE.
  kubernetes-e2e-gce-reboot)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Reboot\]"
    export PROJECT="k8s-jkns-e2e-gce-ci-reboot"
  ;;

  kubernetes-e2e-gke-reboot)
    export PROJECT="k8s-jkns-e2e-gke-ci-reboot"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Reboot\]"
  ;;

  # Runs only the examples tests on GCE.
  kubernetes-e2e-gce-examples)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Example\]"
    export PROJECT="k8s-jkns-e2e-examples"
    ;;

  # Runs only the autoscaling tests on GCE.
  kubernetes-e2e-gce-autoscaling)
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:ClusterSizeAutoscaling\]|\[Feature:InitialResources\] \
                             --ginkgo.skip=\[Flaky\]"
    export PROJECT="k8s-jnks-e2e-gce-autoscaling"
    # Override GCE default for cluster size autoscaling purposes.
    export KUBE_ENABLE_CLUSTER_MONITORING="googleinfluxdb"
    export KUBE_ADMISSION_CONTROL="NamespaceLifecycle,InitialResources,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota"
    ;;

  # Runs the performance/scalability tests on GCE. A larger cluster is used.
  kubernetes-e2e-gce-scalability)
    # XXX Not a unique project
    export E2E_NAME="e2e-scalability"
    export GINKGO_TEST_ARGS="--ginkgo.focus=\[Feature:Performance\] \
                             --gather-resource-usage=true \
                             --gather-metrics-at-teardown=true \
                             --gather-logs-sizes=true \
                             --output-print-type=json"
    export PROJECT="kubernetes-jenkins"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    # Override GCE defaults.
    export MASTER_SIZE="n1-standard-4"
    export NODE_SIZE="n1-standard-2"
    export NODE_DISK_SIZE="50GB"
    export NUM_NODES="100"
    # Reduce logs verbosity
    export TEST_CLUSTER_LOG_LEVEL="--v=2"
    # TODO: Remove when we figure out the reason for occasional failures #19048
    export KUBELET_TEST_LOG_LEVEL="--v=4"
    # Increase resync period to simulate production
    export TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Runs e2e on GCE with flannel and VXLAN.
  kubernetes-e2e-gce-flannel)
    # XXX Not a unique project
    export E2E_NAME="e2e-flannel"
    export PROJECT="kubernetes-flannel"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    # Override GCE defaults.
    export NETWORK_PROVIDER="flannel"
    ;;

  # Runs the performance/scalability test on huge 1000-node cluster on GCE.
  # Flannel is used as network provider.
  # Allows a couple of nodes to be NotReady during startup
  kubernetes-e2e-gce-enormous-cluster)
    # XXX Not a unique project
    export E2E_NAME="e2e-enormous-cluster"
    # TODO: Currently run only density test.
    # Once this is stable, run the whole [Performance] suite.
    export GINKGO_TEST_ARGS="--ginkgo.focus=starting\s30\spods\sper\snode"
    export PROJECT="kubernetes-scale"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    # Override GCE defaults.
    export NETWORK_PROVIDER="flannel"
    # Temporarily switch of Heapster, as this will not schedule anywhere.
    # TODO: Think of a solution to enable it.
    export KUBE_ENABLE_CLUSTER_MONITORING="none"
    export KUBE_GCE_ZONE="asia-east1-a"
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
    # XXX Not a unique project
    # TODO: increase a quota for networks in kubernetes-scale and move this test to its own network
    export E2E_NAME="e2e-enormous-cluster"
    export E2E_TEST="false"
    export PROJECT="kubernetes-scale"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    # Override GCE defaults.
    export NETWORK_PROVIDER="flannel"
    # Temporarily switch of Heapster, as this will not schedule anywhere.
    # TODO: Think of a solution to enable it.
    export KUBE_ENABLE_CLUSTER_MONITORING="none"
    export KUBE_GCE_ZONE="us-east1-b"
    export MASTER_SIZE="n1-standard-32"
    export NODE_SIZE="n1-standard-1"
    export NODE_DISK_SIZE="50GB"
    export NUM_NODES="1000"
    # Reduce logs verbosity
    export TEST_CLUSTER_LOG_LEVEL="--v=1"
    # Increase resync period to simulate production
    export TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Run Kubemark test on a fake 100 node cluster to have a comparison
  # to the real results from scalability suite
  kubernetes-kubemark-gce)
    export PROJECT="k8s-jenkins-kubemark"
    export E2E_TEST="false"
    export USE_KUBEMARK="true"
    export KUBEMARK_TESTS="\[Feature:Performance\]"
    # Override defaults to be independent from GCE defaults and set kubemark parameters
    export NUM_NODES="10"
    export MASTER_SIZE="n1-standard-2"
    export NODE_SIZE="n1-standard-1"
    export KUBE_GCE_ZONE="us-central1-b"
    export KUBEMARK_MASTER_SIZE="n1-standard-4"
    export KUBEMARK_NUM_NODES="100"
    ;;

  # Run Kubemark test on a fake 500 node cluster to test for regressions on
  # bigger clusters
  kubernetes-kubemark-500-gce)
    # XXX Not a unique project
    export E2E_NAME="kubemark-500"
    export PROJECT="kubernetes-scale"
    export E2E_TEST="false"
    export USE_KUBEMARK="true"
    export KUBEMARK_TESTS="\[Feature:Performance\]"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    # Override defaults to be independent from GCE defaults and set kubemark parameters
    export NUM_NODES="6"
    export MASTER_SIZE="n1-standard-4"
    export NODE_SIZE="n1-standard-8"
    export KUBE_GCE_ZONE="us-east1-b"
    export KUBEMARK_MASTER_SIZE="n1-standard-16"
    export KUBEMARK_NUM_NODES="500"
    ;;

  # Run big Kubemark test, this currently means a 1000 node cluster and 16 core master
  kubernetes-kubemark-gce-scale)
    # XXX Not a unique project
    export E2E_NAME="kubemark-1000"
    export PROJECT="kubernetes-scale"
    export E2E_TEST="false"
    export USE_KUBEMARK="true"
    export KUBEMARK_TESTS="\[Feature:Performance\]"
    export FAIL_ON_GCP_RESOURCE_LEAK="false"
    # Override defaults to be independent from GCE defaults and set kubemark parameters
    # We need 11 so that we won't hit max-pods limit (set to 100). TODO: do it in a nicer way.
    export NUM_NODES="11"
    export MASTER_SIZE="n1-standard-4"
    # Note: can fit about 17 hollow nodes per core so NUM_NODES x
    # cores_per_node should be set accordingly.
    export NODE_SIZE="n1-standard-8"
    export KUBEMARK_MASTER_SIZE="n1-standard-16"
    export KUBEMARK_NUM_NODES="1000"
    export KUBE_GCE_ZONE="us-east1-b"
    ;;

  # Soak jobs

  # Sets up the GCE soak cluster weekly using the latest CI release.
  kubernetes-soak-weekly-deploy-gce)
    export HAIRPIN_MODE="false"
    export PROJECT="k8s-jkns-gce-soak"
    ;;

  # Runs tests on GCE soak cluster.
  kubernetes-soak-continuous-e2e-gce)
    export HAIRPIN_MODE="false"
    export PROJECT="k8s-jkns-gce-soak"
    ;;

  # Clone of kubernetes-soak-weekly-deploy-gce. Issue #20832.
  kubernetes-soak-weekly-deploy-gce-2)
    export PROJECT="k8s-jkns-gce-soak-2"
    ;;

  # Clone of kubernetes-soak-continuous-e2e-gce. Issue #20832.
  kubernetes-soak-continuous-e2e-gce-2)
    export PROJECT="k8s-jkns-gce-soak-2"
    ;;

  # Sets up the GKE soak cluster weekly using the latest CI release.
  kubernetes-soak-weekly-deploy-gke)
    export PROJECT="k8s-jkns-gke-soak"
    # Need at least n1-standard-2 nodes to run kubelet_perf tests
    export MACHINE_TYPE="n1-standard-2"
    ;;

  # Runs tests on GKE soak cluster.
  kubernetes-soak-continuous-e2e-gke)
    export PROJECT="k8s-jkns-gke-soak"
    export E2E_OPT="--check_version_skew=false"
    ;;

  # Upgrade jobs

  # kubernetes-upgrade-gke-1.0-master
  #
  # Test upgrades from the latest release-1.0 build to the latest master build.
  #
  # Configurations for step1, step4, and step6 live in the release-1.0 branch.

  kubernetes-upgrade-gke-1.0-master-step2-kubectl-e2e-new)
    export E2E_NAME="upgrade-gke-1-0-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  kubernetes-upgrade-gke-1.0-master-step3-upgrade-master)
    export E2E_NAME="upgrade-gke-1-0-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  kubernetes-upgrade-gke-1.0-master-step5-upgrade-cluster)
    export E2E_NAME="upgrade-gke-1-0-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  kubernetes-upgrade-gke-1.0-master-step7-e2e-new)
    export E2E_NAME="upgrade-gke-1-0-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  # kubernetes-upgrade-gke-1.1-master
  #
  # Test upgrades from the latest release-1.1 build to the latest master build.
  #
  # Configurations for step1, step4, and step6 live in the release-1.1 branch.

  kubernetes-upgrade-gke-1.1-master-step2-kubectl-e2e-new)
    export E2E_NAME="upgrade-gke-1-1-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  kubernetes-upgrade-gke-1.1-master-step3-upgrade-master)
    export E2E_NAME="upgrade-gke-1-1-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  kubernetes-upgrade-gke-1.1-master-step5-upgrade-cluster)
    export E2E_NAME="upgrade-gke-1-1-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  kubernetes-upgrade-gke-1.1-master-step7-e2e-new)
    export E2E_NAME="upgrade-gke-1-1-master"
    export PROJECT="kubernetes-jenkins-gke-upgrade"
    ;;

  # kubernetes-upgrade-gce-1.1-master
  #
  # Test upgrades from the latest release-1.1 build to the latest master build.
  #
  # Configurations for step1, step4, and step6 live in the release-1.1 branch.

  kubernetes-upgrade-gce-1.1-master-step2-kubectl-e2e-new)
    export E2E_NAME="upgrade-gce-1-1-master"
    export PROJECT="k8s-jkns-gce-upgrade"
    ;;

  kubernetes-upgrade-gce-1.1-master-step3-upgrade-master)
    export E2E_NAME="upgrade-gce-1-1-master"
    export PROJECT="k8s-jkns-gce-upgrade"
    ;;

  kubernetes-upgrade-gce-1.1-master-step5-upgrade-cluster)
    export E2E_NAME="upgrade-gce-1-1-master"
    export PROJECT="k8s-jkns-gce-upgrade"
    ;;

  kubernetes-upgrade-gce-1.1-master-step7-e2e-new)
    export E2E_NAME="upgrade-gce-1-1-master"
    export PROJECT="k8s-jkns-gce-upgrade"
    ;;
esac

# Assume we're upping, testing, and downing a cluster
export E2E_UP="${E2E_UP:-true}"
export E2E_TEST="${E2E_TEST:-true}"
export E2E_DOWN="${E2E_DOWN:-true}"

# Skip gcloud update checking
export CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=true

# AWS variables
export KUBE_AWS_INSTANCE_PREFIX=${E2E_NAME:-'jenkins-e2e'}

# GCE variables
export INSTANCE_PREFIX=${E2E_NAME:-'jenkins-e2e'}
export KUBE_GCE_NETWORK=${E2E_NAME:-'jenkins-e2e'}
export KUBE_GCE_INSTANCE_PREFIX=${E2E_NAME:-'jenkins-e2e'}
export GCE_SERVICE_ACCOUNT=$(gcloud auth list 2> /dev/null | grep active | cut -f3 -d' ')

# GKE variables
export CLUSTER_NAME=${E2E_NAME:-'jenkins-e2e'}
export KUBE_GKE_NETWORK=${E2E_NAME:-'jenkins-e2e'}

# Get golang into our PATH so we can run e2e.go
export PATH=${PATH}:/usr/local/go/bin

# If we are on PR Jenkins merging into master, use the local e2e.sh. Otherwise, use the latest on github.
if [[ "${ghprbTargetBranch:-}" == "master" ]]; then
    source "hack/jenkins/e2e-runner.sh"
else
    source <(curl -fsS --retry 3 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/e2e-runner.sh")
fi
