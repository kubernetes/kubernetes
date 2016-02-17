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

# Sets up environment variables for e2e-runner.sh on the master branch.

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# Join all args with |
#   Example: join_regex_allow_empty a b "c d" e  =>  a|b|c d|e
function join_regex_allow_empty() {
    local IFS="|"
    echo "$*"
}

# Join all args with |, butin case of empty result prints "EMPTY\sSET" instead.
#   Example: join_regex_no_empty a b "c d" e  =>  a|b|c d|e
#            join_regex_no_empty => EMPTY\sSET
function join_regex_no_empty() {
    local IFS="|"
    if [ -z "$*" ]; then
        echo "EMPTY\sSET"
    else
        echo "$*"
    fi
}

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
#   $GKE_DEFAULT_SKIP_TESTS
#   $GCE_DEFAULT_SKIP_TESTS
#   $GCE_FLAKY_TESTS
#   $GCE_SLOW_TESTS
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

  local -r gce_test_args="--ginkgo.skip=$(join_regex_allow_empty \
        ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
        ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
        ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
        )"
  local -r gke_test_args="--ginkgo.skip=$(join_regex_allow_empty \
        ${GKE_DEFAULT_SKIP_TESTS[@]:+${GKE_DEFAULT_SKIP_TESTS[@]}} \
        ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
        ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
        ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
        )"

  if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
    DOGFOOD_GCLOUD="true"
    GKE_API_ENDPOINT="https://test-container.sandbox.googleapis.com/"
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
      GINKGO_TEST_ARGS="--ginkgo.focus=Cluster\supgrade.*upgrade-master --upgrade-target=${new_version}"
      ;;

    step4)
      # Run old e2es
      JENKINS_PUBLISHED_VERSION="${old_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"

      if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
        GINKGO_TEST_ARGS="${gke_test_args}"
      else
        GINKGO_TEST_ARGS="${gce_test_args}"
      fi
      ;;

    step5)
      # Use upgrade logic of version we're upgrading to.
      JENKINS_PUBLISHED_VERSION="${new_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"
      GINKGO_TEST_ARGS="--ginkgo.focus=Cluster\supgrade.*upgrade-cluster --upgrade-target=${new_version}"
      ;;

    step6)
      # Run old e2es
      JENKINS_PUBLISHED_VERSION="${old_version}"
      JENKINS_FORCE_GET_TARS=y

      E2E_OPT="--check_version_skew=false"
      E2E_UP="false"
      E2E_TEST="true"
      E2E_DOWN="false"

      if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
        GINKGO_TEST_ARGS="${gke_test_args}"
      else
        GINKGO_TEST_ARGS="${gce_test_args}"
      fi
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

      if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
        GINKGO_TEST_ARGS="${gke_test_args}"
      else
        GINKGO_TEST_ARGS="${gce_test_args}"
      fi
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
  : ${NUM_MINIONS_PARALLEL:="6"}  # Number of nodes required to run all of the tests in parallel

elif [[ ${JOB_NAME} =~ ^kubernetes-.*-gke ]]; then
  KUBERNETES_PROVIDER="gke"
  : ${E2E_ZONE:="us-central1-f"}
elif [[ ${JOB_NAME} =~ ^kubernetes-.*-aws ]]; then
  KUBERNETES_PROVIDER="aws"
  : ${E2E_MIN_STARTUP_PODS:="1"}
  : ${E2E_ZONE:="us-east-1a"}
  : ${NUM_MINIONS_PARALLEL:="6"}  # Number of nodes required to run all of the tests in parallel
fi

if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
  if [[ "${PERFORMANCE:-}" == "true" ]]; then
    : ${MASTER_SIZE:="m3.xlarge"}
    : ${NUM_MINIONS:="100"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=\[Performance\ssuite\]"}
  else
    : ${MASTER_SIZE:="m3.large"}
    : ${MINION_SIZE:="m3.large"}
    : ${NUM_MINIONS:="3"}
  fi
fi

# Specialized to skip when running reboot tests.
REBOOT_SKIP_TESTS=(
    "Skipped"
    "Restart\sshould\srestart\sall\snodes"
    "Example"
    )

# Specialized tests which should be skipped by default for projects.
GCE_DEFAULT_SKIP_TESTS=(
    "${REBOOT_SKIP_TESTS[@]}"
    "Reboot"
    "ServiceLoadBalancer"
    )

# Tests which cannot be run on GKE, e.g. because they require
# master ssh access.
GKE_REQUIRED_SKIP_TESTS=(
    "Nodes"
    "Etcd\sFailure"
    "MasterCerts"
    "experimental\sresource\susage\stracking" # Expect --max-pods=100
    "ServiceLoadBalancer" # issue: #16602
    "Shell"
    # Alpha features, remove from skip when these move to beta
    "Daemon\sset"
    "Deployment"
    )

# Specialized tests which should be skipped by default for GKE.
GKE_DEFAULT_SKIP_TESTS=(
    "Autoscaling\sSuite"
    # Perf test, slow by design
    "resource\susage\stracking"
    "${GKE_REQUIRED_SKIP_TESTS[@]}"
    )

# Tests which cannot be run on AWS.
AWS_REQUIRED_SKIP_TESTS=(
    "experimental\sresource\susage\stracking" # Expect --max-pods=100
    "GCE\sL7\sLoadBalancer\sController" # GCE L7 loadbalancing
)


# Tests which kills or restarts components and/or nodes.
DISRUPTIVE_TESTS=(
    "DaemonRestart"
    "Etcd\sfailure"
    "Nodes\sResize"
    "Reboot"
    "Services.*restarting"
)

# The following tests are known to be flaky, and are thus run only in their own
# -flaky- build variants.
GCE_FLAKY_TESTS=(
    "DaemonRestart\sController\sManager"
    "Daemon\sset\sshould\srun\sand\sstop\scomplex\sdaemon"
    "Resource\susage\sof\ssystem\scontainers"
    "allows\sscheduling\sof\spods\son\sa\sminion\safter\sit\srejoins\sthe\scluster" # file: resize_nodes.go, issue: #13258
    )

# The following tests are known to be slow running (> 2 min), and are
# thus run only in their own -slow- build variants.  Note that tests
# can be slow by explicit design (e.g. some soak tests), or slow
# through poor implementation.  Please indicate which applies in the
# comments below, and for poorly implemented tests, please quote the
# issue number tracking speed improvements.
GCE_SLOW_TESTS=(
    # Before enabling this loadbalancer test in any other test list you must
    # make sure the associated project has enough quota. At the time of this
    # writing a GCE project is allowed 3 backend services by default. This
    # test requires at least 5.
    "GCE\sL7\sLoadBalancer\sController"               # 10 min,       file: ingress.go,              slow by design
    "SchedulerPredicates\svalidates\sMaxPods\slimit " # 8 min,        file: scheduler_predicates.go, PR:    #13315
    "Nodes\sResize"                                   # 3 min 30 sec, file: resize_nodes.go,         issue: #13323
    "resource\susage\stracking"                       # 1 hour,       file: kubelet_perf.go,         slow by design
    "monotonically\sincreasing\srestart\scount"       # 1.5 to 5 min, file: pods.go,                 slow by design
    "Garbage\scollector\sshould"                      # 7 min,        file: garbage_collector.go,    slow by design
    "KubeProxy\sshould\stest\skube-proxy"             # 9 min 30 sec, file: kubeproxy.go,            issue: #14204
    "cap\sback-off\sat\sMaxContainerBackOff"          # 20 mins       file: manager.go,              PR:    #12648
    )

# Tests which are not able to be run in parallel.
GCE_PARALLEL_SKIP_TESTS=(
    "GCE\sL7\sLoadBalancer\sController" # TODO: This cannot run in parallel with other L4 tests till quota has been bumped up.
    "Nodes\sNetwork"
    "MaxPods"
    "Resource\susage\sof\ssystem\scontainers"
    "SchedulerPredicates"
    "resource\susage\stracking"
    "${DISRUPTIVE_TESTS[@]}"
    )

# Tests which are known to be flaky when run in parallel.
GCE_PARALLEL_FLAKY_TESTS=(
    "DaemonRestart"
    "Elasticsearch"
    "Namespaces\sDelete\s90\spercent"
    "ServiceAccounts"
    "Services.*identically\snamed" # error waiting for reachability, issue: #16285
    )

# Tests that should not run on soak cluster.
GCE_SOAK_CONTINUOUS_SKIP_TESTS=(
    "GCE\sL7\sLoadBalancer\sController" # issue: #17119
    "Density.*30\spods"
    "Elasticsearch"
    "external\sload\sbalancer"
    "identically\snamed\sservices"
    "network\spartition"
    "Services.*Type\sgoes\sfrom"
    "${DISRUPTIVE_TESTS[@]}"       # avoid component restarts.
    )

GCE_RELEASE_SKIP_TESTS=(
    )

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
    # This list should match the list in kubernetes-e2e-gce-parallel.
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_FLAKY_TESTS[@]:+${GCE_PARALLEL_FLAKY_TESTS[@]}} \
          ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-gce-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${KUBE_GCS_STAGING_PATH_SUFFIX:="-${NODE_NAME}-${EXECUTOR_NUMBER}"}
    : ${PROJECT:="kubernetes-jenkins-pull"}
    : ${ENABLE_DEPLOYMENTS:=true}
    # Override GCE defaults
    NUM_MINIONS=${NUM_MINIONS_PARALLEL}
    ;;

  # Runs non-flaky tests on GCE on the release-1.0 branch,
  # sequentially. As a reminder, if you need to change the skip list
  # or flaky test list on the release branch, you'll need to propose a
  # pull request directly to the release branch itself.
  kubernetes-e2e-gce-release-1.0)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-release-1.0"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-gce-release-1-0"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_RELEASE_SKIP_TESTS[@]:+${GCE_RELEASE_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX="e2e-gce-1-0"}
    : ${KUBE_GCS_STAGING_PATH_SUFFIX:="release-1.0"}
    : ${PROJECT:="k8s-jkns-e2e-gce-release"}
    ;;

  # kubernetes-upgrade-gke-1.0-current-release
  #
  # Test upgrades from the latest release-1.0 build to the latest current
  # release build.
  #
  # Configurations for step2, step3, step5, and step7 live in the current release
  # branch.

  kubernetes-upgrade-gke-1.0-current-release-step1-deploy)
    configure_upgrade_step 'ci/latest-1.0' 'configured-in-current-release-branch' 'upgrade-gke-1-0-current-release' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.0-current-release-step4-e2e-old)
    configure_upgrade_step 'ci/latest-1.0' 'configured-in-current-release-branch' 'upgrade-gke-1-0-current-release' 'kubernetes-jenkins-gke-upgrade'
    ;;

  kubernetes-upgrade-gke-1.0-current-release-step6-e2e-old)
    configure_upgrade_step 'ci/latest-1.0' 'configured-in-current-release-branch' 'upgrade-gke-1-0-current-release' 'kubernetes-jenkins-gke-upgrade'
    ;;

  # kubernetes-upgrade-gke-1.0-master
  #
  # Test upgrades from the latest release-1.0 build to the latest master build.
  #
  # Configurations for step2, step3, step5, and step7 live in master.

  kubernetes-upgrade-gke-1.0-master-step1-deploy)
    configure_upgrade_step 'ci/latest-1.0' 'configured-in-master' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    # Starting in v1.2, NUM_MINIONS/NUM_NODES defaults to 3, so we have to deploy 3 here.
    NUM_MINIONS=3
    ;;

  kubernetes-upgrade-gke-1.0-master-step4-e2e-old)
    configure_upgrade_step 'ci/latest-1.0' 'configured-in-master' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    # Starting in v1.2, NUM_MINIONS/NUM_NODES defaults to 3, so we have to deploy 3 here.
    NUM_MINIONS=3
    ;;

  kubernetes-upgrade-gke-1.0-master-step6-e2e-old)
    configure_upgrade_step 'ci/latest-1.0' 'configured-in-master' 'upgrade-gke-1-0-master' 'kubernetes-jenkins-gke-upgrade'
    # Starting in v1.2, NUM_MINIONS/NUM_NODES defaults to 3, so we have to deploy 3 here.
    NUM_MINIONS=3
    ;;
esac

# AWS variables
export KUBE_AWS_INSTANCE_PREFIX=${E2E_CLUSTER_NAME}
export KUBE_AWS_ZONE=${E2E_ZONE}

# GCE variables
export INSTANCE_PREFIX=${E2E_CLUSTER_NAME}
export KUBE_GCE_ZONE=${E2E_ZONE}
export KUBE_GCE_NETWORK=${E2E_NETWORK}
export KUBE_GCE_INSTANCE_PREFIX=${KUBE_GCE_INSTANCE_PREFIX:-}
export KUBE_GCS_STAGING_PATH_SUFFIX=${KUBE_GCS_STAGING_PATH_SUFFIX:-}
export KUBE_GCE_MINION_PROJECT=${KUBE_GCE_MINION_PROJECT:-}
export KUBE_GCE_MINION_IMAGE=${KUBE_GCE_MINION_IMAGE:-}
export KUBE_OS_DISTRIBUTION=${KUBE_OS_DISTRIBUTION:-}
export GCE_SERVICE_ACCOUNT=$(gcloud auth list 2> /dev/null | grep active | cut -f3 -d' ')

# GKE variables
export CLUSTER_NAME=${E2E_CLUSTER_NAME}
export ZONE=${E2E_ZONE}
export KUBE_GKE_NETWORK=${E2E_NETWORK}
export E2E_SET_CLUSTER_API_VERSION=${E2E_SET_CLUSTER_API_VERSION:-}
export DOGFOOD_GCLOUD=${DOGFOOD_GCLOUD:-}
export CMD_GROUP=${CMD_GROUP:-}
export MACHINE_TYPE=${MINION_SIZE:-}  # GKE scripts use MACHINE_TYPE for the node vm size

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
export MINION_SIZE=${MINION_SIZE:-}
export MINION_DISK_SIZE=${MINION_DISK_SIZE:-}
export NUM_MINIONS=${NUM_MINIONS:-}
export TEST_CLUSTER_LOG_LEVEL=${TEST_CLUSTER_LOG_LEVEL:-}
export TEST_CLUSTER_RESYNC_PERIOD=${TEST_CLUSTER_RESYNC_PERIOD:-}
export PROJECT=${PROJECT:-}
export JENKINS_EXPLICIT_VERSION=${JENKINS_EXPLICIT_VERSION:-}
export JENKINS_PUBLISHED_VERSION=${JENKINS_PUBLISHED_VERSION:-'ci/latest-1.0'}

export KUBE_ADMISSION_CONTROL=${ADMISSION_CONTROL:-}

export KUBERNETES_PROVIDER=${KUBERNETES_PROVIDER}
export PATH=${PATH}:/usr/local/go/bin
export KUBE_SKIP_UPDATE=y
export KUBE_SKIP_CONFIRMATIONS=y

# E2E Control Variables
export E2E_OPT="${E2E_OPT:-}"
export E2E_UP="${E2E_UP:-true}"
export E2E_TEST="${E2E_TEST:-true}"
export E2E_DOWN="${E2E_DOWN:-true}"
export E2E_CLEAN_START="${E2E_CLEAN_START:-}"
# Used by hack/ginkgo-e2e.sh to enable ginkgo's parallel test runner.
export GINKGO_PARALLEL=${GINKGO_PARALLEL:-}
export GINKGO_TEST_ARGS="${GINKGO_TEST_ARGS:-}"

# Use e2e-runner.sh from the master branch.
source <(curl -fsS --retry 3 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/e2e-runner.sh")
