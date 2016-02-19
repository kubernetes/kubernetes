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
