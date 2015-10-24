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

# kubernetes-e2e-{gce, gke, gke-ci} jobs: This script is triggered by
# the kubernetes-build job, or runs every half hour. We abort this job
# if it takes more than 75m. As of initial commit, it typically runs
# in about half an hour.
#
# The "Workspace Cleanup Plugin" is installed and in use for this job,
# so the ${WORKSPACE} directory (the current directory) is currently
# empty.

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
    "Autoscaling\sSuite"
    "Skipped"
    "Reboot"
    "Restart\sshould\srestart\sall\snodes"
    "Example"
    )

# Specialized tests which should be skipped by default for projects.
GCE_DEFAULT_SKIP_TESTS=(
    "${REBOOT_SKIP_TESTS[@]}"
    "Reboot"
    )

# Tests which cannot be run on GKE, e.g. because they require
# master ssh access.
GKE_REQUIRED_SKIP_TESTS=(
    "Nodes"
    "Etcd\sFailure"
    "MasterCerts"
    "Shell"
    "Daemon\sset"
    )

# The following tests are known to be flaky, and are thus run only in their own
# -flaky- build variants.
GCE_FLAKY_TESTS=(
    "DaemonRestart"
    "Daemon\sset\sshould\slaunch\sa\sdaemon\spod\son\severy\snode\sof\sthe\scluster"
    "Resource\susage\sof\ssystem\scontainers"
    "monotonically\sincreasing\srestart\scount"
    "should\sbe\sable\sto\schange\sthe\stype\sand\snodeport\ssettings\sof\sa\sservice" # file: service.go, issue: #13032
    "allows\sscheduling\sof\spods\son\sa\sminion\safter\sit\srejoins\sthe\scluster" # file: resize_nodes.go, issue: #13258
    "should\srelease\sthe\sload\sbalancer\swhen\sType\sgoes\sfrom\sLoadBalancer" # timeouts in 20 minutes in last builds. #14424
    "should\scorrectly\sserve\sidentically\snamed\sservices\sin\sdifferent\snamespaces\son\sdifferent\sexternal\sIP\saddresses" # same as above
    "should\sbe\sable\sto\screate\sa\sfunctioning\sexternal\sload\sbalancer" # same as above, also catches "...with user-provided balancer ip"
    )

# The following tests are known to be slow running (> 2 min), and are
# thus run only in their own -slow- build variants.  Note that tests
# can be slow by explicit design (e.g. some soak tests), or slow
# through poor implementation.  Please indicate which applies in the
# comments below, and for poorly implemented tests, please quote the
# issue number tracking speed improvements.
GCE_SLOW_TESTS=(
    "SchedulerPredicates\svalidates\sMaxPods\slimit " # 8 min,        file: scheduler_predicates.go, PR:    #13315
    "Nodes\sResize"                                   # 3 min 30 sec, file: resize_nodes.go,         issue: #13323
    "resource\susage\stracking"                       # 1 hour,       file: kubelet_perf.go,         slow by design
    )

# Tests which are not able to be run in parallel.
GCE_PARALLEL_SKIP_TESTS=(
    "Etcd"
    "NetworkingNew"
    "Nodes\sNetwork"
    "Nodes\sResize"
    "MaxPods"
    "Resource\susage\sof\ssystem\scontainers"
    "SchedulerPredicates"
    "Services.*restarting"
    "Shell.*services"
    "resource\susage\stracking"
    )

# Tests which are known to be flaky when run in parallel.
GCE_PARALLEL_FLAKY_TESTS=(
    "DaemonRestart"
    "Elasticsearch"
    "Namespaces.*should\sdelete\sfast"
    "PD"
    "ServiceAccounts"
    "Services.*change\sthe\stype"
    "Services.*functioning\sexternal\sload\sbalancer"
    "Services.*identically\snamed"
    "Services.*release.*load\sbalancer"
    "Services.*endpoint"
    "Services.*up\sand\sdown"
    "Networking\sshould\sfunction\sfor\sintra-pod\scommunication"  # possibly causing Ginkgo to get stuck, issue: #13485
    "Kubectl\sexpose"
    )

# Tests that should not run on soak cluster.
GCE_SOAK_CONTINUOUS_SKIP_TESTS=(
    "Density.*30\spods"
    "Elasticsearch"
    "Etcd.*SIGKILL"
    "external\sload\sbalancer"
    "identically\snamed\sservices"
    "network\spartition"
    "Services.*Type\sgoes\sfrom"
    )

GCE_RELEASE_SKIP_TESTS=(
    )

# Define environment variables based on the Jenkins project name.
case ${JOB_NAME} in
  # Runs all non-flaky, non-slow tests on GCE, sequentially.
  kubernetes-e2e-gce)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-gce"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX="e2e-gce"}
    : ${PROJECT:="k8s-jkns-e2e-gce"}
    : ${ENABLE_DEPLOYMENTS:=true}
    ;;

  # Runs only the examples tests on GCE.
  kubernetes-e2e-gce-examples)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-examples"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-examples"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=Example"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-examples"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Runs only the autoscaling tests on GCE.
  kubernetes-e2e-gce-autoscaling)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-autoscaling"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-autoscaling"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=Autoscaling\sSuite"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-autoscaling"}
    : ${PROJECT:="k8s-jnks-e2e-gce-autoscaling"}
    # Override GCE default for cluster size autoscaling purposes.
    ENABLE_CLUSTER_MONITORING="googleinfluxdb"
    ENABLE_HORIZONTAL_POD_AUTOSCALER="true"
    ;;

  # Runs the flaky tests on GCE, sequentially.
  kubernetes-e2e-gce-flaky)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-flaky"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-flaky"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ) --ginkgo.focus=$(join_regex_no_empty \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-flaky"}
    : ${PROJECT:="k8s-jkns-e2e-gce-flaky"}
    ;;

  # Runs slow tests on GCE, sequentially.
  kubernetes-e2e-gce-slow)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-slow"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-slow"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=$(join_regex_no_empty \
          ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-slow"}
    : ${PROJECT:="k8s-jkns-e2e-gce-slow"}
    ;;

  # Runs a subset of tests on GCE in parallel. Run against all pending PRs.
  kubernetes-pull-build-test-e2e-gce)
    : ${E2E_CLUSTER_NAME:="jenkins-pull-gce-e2e-${EXECUTOR_NUMBER}"}
    : ${E2E_NETWORK:="pull-e2e-parallel-${EXECUTOR_NUMBER}"}
    : ${GINKGO_PARALLEL:="y"}
    # This list should match the list in kubernetes-e2e-gce-parallel.
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_FLAKY_TESTS[@]:+${GCE_PARALLEL_FLAKY_TESTS[@]}} \
          ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="pull-e2e-${EXECUTOR_NUMBER}"}
    : ${KUBE_GCS_STAGING_PATH_SUFFIX:="-${EXECUTOR_NUMBER}"}
    : ${PROJECT:="kubernetes-jenkins-pull"}
    : ${ENABLE_DEPLOYMENTS:=true}
    # Override GCE defaults
    NUM_MINIONS=${NUM_MINIONS_PARALLEL}
    ;;

  # Runs all non-flaky tests on GCE in parallel.
  kubernetes-e2e-gce-parallel)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-parallel"}
    : ${E2E_NETWORK:="e2e-parallel"}
    : ${GINKGO_PARALLEL:="y"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_FLAKY_TESTS[@]:+${GCE_PARALLEL_FLAKY_TESTS[@]}} \
          ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-test-parallel"}
    : ${PROJECT:="kubernetes-jenkins"}
    : ${ENABLE_DEPLOYMENTS:=true}
    # Override GCE defaults
    NUM_MINIONS=${NUM_MINIONS_PARALLEL}
    ;;

  # Runs all non-flaky tests on AWS in parallel.
  kubernetes-e2e-aws-parallel)
    : ${E2E_CLUSTER_NAME:="jenkins-aws-e2e-parallel"}
    : ${E2E_NETWORK:="e2e-parallel"}
    : ${GINKGO_PARALLEL:="y"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_FLAKY_TESTS[@]:+${GCE_PARALLEL_FLAKY_TESTS[@]}} \
          )"}
    : ${ENABLE_DEPLOYMENTS:=true}
    # Override AWS defaults.
    NUM_MINIONS="6"
    ;;

  # Runs the flaky tests on GCE in parallel.
  kubernetes-e2e-gce-parallel-flaky)
    : ${E2E_CLUSTER_NAME:="parallel-flaky"}
    : ${E2E_NETWORK:="e2e-parallel-flaky"}
    : ${GINKGO_PARALLEL:="y"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          ) --ginkgo.focus=$(join_regex_no_empty \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_FLAKY_TESTS[@]:+${GCE_PARALLEL_FLAKY_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="parallel-flaky"}
    : ${PROJECT:="k8s-jkns-e2e-gce-prl-flaky"}
    # Override GCE defaults.
    NUM_MINIONS=${NUM_MINIONS_PARALLEL}
    ;;

  # Runs only the reboot tests on GCE.
  kubernetes-e2e-gce-reboot)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-reboot"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-reboot"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=Reboot"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-reboot"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Runs the performance/scalability tests on GCE. A larger cluster is used.
  kubernetes-e2e-gce-scalability)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-scalability"}
    : ${E2E_NETWORK:="e2e-scalability"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=Performance\ssuite"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="e2e-scalability"}
    : ${PROJECT:="kubernetes-jenkins"}
    # Override GCE defaults.
    MASTER_SIZE="n1-standard-4"
    MINION_SIZE="n1-standard-2"
    MINION_DISK_SIZE="50GB"
    NUM_MINIONS="100"
    # Reduce logs verbosity
    TEST_CLUSTER_LOG_LEVEL="--v=1"
    # Increase resync period to simulate production
    TEST_CLUSTER_RESYNC_PERIOD="--min-resync-period=12h"
    ;;

  # Runs tests on GCE soak cluster.
  kubernetes-soak-continuous-e2e-gce)
    : ${E2E_CLUSTER_NAME:="gce-soak-weekly"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="gce-soak-weekly"}
    : ${E2E_UP:="false"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_SOAK_CONTINUOUS_SKIP_TESTS[@]:+${GCE_SOAK_CONTINUOUS_SKIP_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX:="gce-soak-weekly"}
    : ${PROJECT:="kubernetes-jenkins"}
    ;;

  # Runs non-flaky tests on GCE on the release-latest branch,
  # sequentially. As a reminder, if you need to change the skip list
  # or flaky test list on the release branch, you'll need to propose a
  # pull request directly to the release branch itself.
  kubernetes-e2e-gce-release)
    : ${E2E_CLUSTER_NAME:="jenkins-gce-e2e-release"}
    : ${E2E_DOWN:="false"}
    : ${E2E_NETWORK:="e2e-gce-release"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_RELEASE_SKIP_TESTS[@]:+${GCE_RELEASE_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    : ${KUBE_GCE_INSTANCE_PREFIX="e2e-gce"}
    : ${PROJECT:="k8s-jkns-e2e-gce-release"}
    ;;

  kubernetes-e2e-gke-prod)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-prod"}
    : ${E2E_NETWORK:="e2e-gke-prod"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${JENKINS_USE_SERVER_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-prod"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GKE_REQUIRED_SKIP_TESTS[@]:+${GKE_REQUIRED_SKIP_TESTS[@]}} \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    ;;

  kubernetes-e2e-gke-staging)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${GKE_API_ENDPOINT:="https://staging-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-staging"}
    : ${E2E_NETWORK:="e2e-gke-staging"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${JENKINS_USE_SERVER_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-staging"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GKE_REQUIRED_SKIP_TESTS[@]:+${GKE_REQUIRED_SKIP_TESTS[@]}} \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    ;;

  kubernetes-e2e-gke-test)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${CLOUDSDK_BUCKET:="gs://cloud-sdk-build/testing/rc"}
    : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-test"}
    : ${E2E_NETWORK:="e2e-gke-test"}
    : ${JENKINS_USE_RELEASE_TARS:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-ci"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GKE_REQUIRED_SKIP_TESTS[@]:+${GKE_REQUIRED_SKIP_TESTS[@]}} \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    ;;

  kubernetes-e2e-gke-ci)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${CLOUDSDK_BUCKET:="gs://cloud-sdk-build/testing/staging"}
    : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-ci"}
    : ${E2E_NETWORK:="e2e-gke-ci"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-ci"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GKE_REQUIRED_SKIP_TESTS[@]:+${GKE_REQUIRED_SKIP_TESTS[@]}} \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    ;;

  kubernetes-e2e-gke-ci-reboot)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${CLOUDSDK_BUCKET:="gs://cloud-sdk-build/testing/staging"}
    : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="jkns-gke-e2e-ci-reboot"}
    : ${E2E_NETWORK:="e2e-gke-ci"}
    : ${E2E_SET_CLUSTER_API_VERSION:=y}
    : ${PROJECT:="k8s-jkns-e2e-gke-ci"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GKE_REQUIRED_SKIP_TESTS[@]:+${GKE_REQUIRED_SKIP_TESTS[@]}} \
          ${REBOOT_SKIP_TESTS[@]:+${REBOOT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          )"}
    ;;

  kubernetes-upgrade-gke-step1-deploy)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="gke-upgrade"}
    : ${E2E_NETWORK:="gke-upgrade"}
    : ${JENKINS_USE_RELEASE_TARS:=y}
    : ${PROJECT:="kubernetes-jenkins-gke-upgrade"}
    : ${E2E_UP:="true"}
    : ${E2E_TEST:="false"}
    : ${E2E_DOWN:="false"}
    ;;

  kubernetes-upgrade-gke-step2-upgrade)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="gke-upgrade"}
    : ${E2E_NETWORK:="gke-upgrade"}
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    : ${JENKINS_USE_RELEASE_TARS:=n}
    : ${PROJECT:="kubernetes-jenkins-gke-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="false"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=Skipped.*Cluster\supgrade.*upgrade-cluster"}
    ;;
  
  kubernetes-upgrade-gke-step3-e2e)
    : ${DOGFOOD_GCLOUD:="true"}
    : ${GKE_API_ENDPOINT:="https://test-container.sandbox.googleapis.com/"}
    : ${E2E_CLUSTER_NAME:="gke-upgrade"}
    : ${E2E_NETWORK:="gke-upgrade"}
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${PROJECT:="kubernetes-jenkins-gke-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GKE_REQUIRED_SKIP_TESTS[@]:+${GKE_REQUIRED_SKIP_TESTS[@]}} \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          )"}
    ;;

  kubernetes-upgrade-gce-step1-deploy)
    : ${E2E_CLUSTER_NAME:="gce-upgrade"}
    : ${E2E_NETWORK:="gce-upgrade"}
    : ${JENKINS_USE_RELEASE_TARS:=y}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="true"}
    : ${E2E_TEST:="false"}
    : ${E2E_DOWN:="false"}
    : ${ENABLE_DEPLOYMENTS:=true}
    ;;

  kubernetes-upgrade-gce-step2-upgrade)
    : ${E2E_CLUSTER_NAME:="gce-upgrade"}
    : ${E2E_NETWORK:="gce-upgrade"}
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${JENKINS_FORCE_GET_TARS:=y}
    : ${JENKINS_USE_RELEASE_TARS:=n}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="false"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.focus=Skipped.*Cluster\supgrade.*upgrade-cluster"}
    ;;

  kubernetes-upgrade-gce-step3-e2e)
    : ${E2E_CLUSTER_NAME:="gce-upgrade"}
    : ${E2E_NETWORK:="gce-upgrade"}
    : ${E2E_OPT:="--check_version_skew=false"}
    : ${PROJECT:="k8s-jkns-gce-upgrade"}
    : ${E2E_UP:="false"}
    : ${E2E_TEST:="true"}
    : ${E2E_DOWN:="true"}
    : ${GINKGO_TEST_ARGS:="--ginkgo.skip=$(join_regex_allow_empty \
          ${GCE_DEFAULT_SKIP_TESTS[@]:+${GCE_DEFAULT_SKIP_TESTS[@]}} \
          ${GCE_PARALLEL_SKIP_TESTS[@]:+${GCE_PARALLEL_SKIP_TESTS[@]}} \
          ${GCE_FLAKY_TESTS[@]:+${GCE_FLAKY_TESTS[@]}} \
          ${GCE_PARALLEL_FLAKY_TESTS[@]:+${GCE_PARALLEL_FLAKY_TESTS[@]}} \
          ${GCE_SLOW_TESTS[@]:+${GCE_SLOW_TESTS[@]}} \
          )"}
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

# GKE variables
export CLUSTER_NAME=${E2E_CLUSTER_NAME}
export ZONE=${E2E_ZONE}
export KUBE_GKE_NETWORK=${E2E_NETWORK}
export E2E_SET_CLUSTER_API_VERSION=${E2E_SET_CLUSTER_API_VERSION:-}
export DOGFOOD_GCLOUD=${DOGFOOD_GCLOUD:-}
export CMD_GROUP=${CMD_GROUP:-}

if [[ ! -z "${GKE_API_ENDPOINT:-}" ]]; then
  export CLOUDSDK_API_ENDPOINT_OVERRIDES_CONTAINER=${GKE_API_ENDPOINT}
fi

# Shared cluster variables
export E2E_MIN_STARTUP_PODS=${E2E_MIN_STARTUP_PODS:-}
export KUBE_ENABLE_CLUSTER_MONITORING=${ENABLE_CLUSTER_MONITORING:-}
export KUBE_ENABLE_HORIZONTAL_POD_AUTOSCALER=${ENABLE_HORIZONTAL_POD_AUTOSCALER:-}
export KUBE_ENABLE_DEPLOYMENTS=${ENABLE_DEPLOYMENTS:-}
export MASTER_SIZE=${MASTER_SIZE:-}
export MINION_SIZE=${MINION_SIZE:-}
export NUM_MINIONS=${NUM_MINIONS:-}
export PROJECT=${PROJECT:-}

export KUBERNETES_PROVIDER=${KUBERNETES_PROVIDER}
export PATH=${PATH}:/usr/local/go/bin
export KUBE_SKIP_CONFIRMATIONS=y

# E2E Control Variables
export E2E_UP="${E2E_UP:-true}"
export E2E_TEST="${E2E_TEST:-true}"
export E2E_DOWN="${E2E_DOWN:-true}"
# Used by hack/ginkgo-e2e.sh to enable ginkgo's parallel test runner.
export GINKGO_PARALLEL=${GINKGO_PARALLEL:-}

echo "--------------------------------------------------------------------------------"
echo "Test Environment:"
printenv | sort
echo "--------------------------------------------------------------------------------"

# We get the Kubernetes tarballs on either cluster creation or when we want to
# replace existing ones in a multi-step job (e.g. a cluster upgrade).
if [[ "${E2E_UP,,}" == "true" || "${JENKINS_FORCE_GET_TARS:-}" =~ ^[yY]$ ]]; then
    if [[ ${KUBE_RUN_FROM_OUTPUT:-} =~ ^[yY]$ ]]; then
        echo "Found KUBE_RUN_FROM_OUTPUT=y; will use binaries from _output"
        cp _output/release-tars/kubernetes*.tar.gz .
    else
        echo "Pulling binaries from GCS"
        # In a multi-step job, clean up just the kubernetes build files.
        # Otherwise, we want a completely empty directory.
        if [[ "${JENKINS_FORCE_GET_TARS:-}" =~ ^[yY]$ ]]; then
            rm -rf kubernetes*
        elif [[ $(find . | wc -l) != 1 ]]; then
            echo $PWD not empty, bailing!
            exit 1
        fi

        # Tell kube-up.sh to skip the update, it doesn't lock. An internal
        # gcloud bug can cause racing component updates to stomp on each
        # other.
        export KUBE_SKIP_UPDATE=y
        {
          sudo flock -x -n 9
          gcloud components update -q || true
          gcloud components update preview -q || true
          gcloud components update alpha -q || true
          gcloud components update beta -q || true
        } 9>/var/run/lock/gcloud-components.lock

        if [[ ! -z ${JENKINS_EXPLICIT_VERSION:-} ]]; then
            # Use an explicit pinned version like "ci/v0.10.0-101-g6c814c4" or
            # "release/v0.19.1"
            IFS='/' read -a varr <<< "${JENKINS_EXPLICIT_VERSION}"
            bucket="${varr[0]}"
            githash="${varr[1]}"
            echo "$bucket / $githash"
        elif [[ ${JENKINS_USE_SERVER_VERSION:-}  =~ ^[yY]$ ]]; then
            # for GKE we can use server default version.
            bucket="release"
            msg=$(gcloud ${CMD_GROUP} container get-server-config --project=${PROJECT} --zone=${ZONE} | grep defaultClusterVersion)
            # msg will look like "defaultClusterVersion: 1.0.1". Strip
            # everything up to, including ": "
            githash="v${msg##*: }"
        else
            # The "ci" bucket is for builds like "v0.15.0-468-gfa648c1"
            bucket="ci"
            # The "latest" version picks the most recent "ci" or "release" build.
            version_file="latest"
            if [[ ${JENKINS_USE_RELEASE_TARS:-} =~ ^[yY]$ ]]; then
                # The "release" bucket is for builds like "v0.15.0"
                bucket="release"
                if [[ ${JENKINS_USE_STABLE:-} =~ ^[yY]$ ]]; then
                    # The "stable" version picks the most recent "release" build.
                    version_file="stable"
                fi
            fi
            githash=$(gsutil cat gs://kubernetes-release/${bucket}/${version_file}.txt)
        fi
        # At this point, we want to have the following vars set:
        # - bucket
        # - githash
        gsutil -m cp gs://kubernetes-release/${bucket}/${githash}/kubernetes.tar.gz gs://kubernetes-release/${bucket}/${githash}/kubernetes-test.tar.gz .
    fi

    if [[ ! "${CIRCLECI:-}" == "true" ]]; then
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
    fi

    md5sum kubernetes*.tar.gz
    tar -xzf kubernetes.tar.gz
    tar -xzf kubernetes-test.tar.gz

    # Set by GKE-CI to change the CLUSTER_API_VERSION to the git version
    if [[ ! -z ${E2E_SET_CLUSTER_API_VERSION:-} ]]; then
        export CLUSTER_API_VERSION=$(echo ${githash} | cut -c 2-)
    elif [[ ${JENKINS_USE_RELEASE_TARS:-} =~ ^[yY]$ ]]; then
        release=$(gsutil cat gs://kubernetes-release/release/${version_file}.txt | cut -c 2-)
        export CLUSTER_API_VERSION=${release}
    fi
fi

cd kubernetes

# Have cmd/e2e run by goe2e.sh generate JUnit report in ${WORKSPACE}/junit*.xml
ARTIFACTS=${WORKSPACE}/_artifacts
mkdir -p ${ARTIFACTS}
export E2E_REPORT_DIR=${ARTIFACTS}

### Pre Set Up ###
# Install gcloud from a custom path if provided. Used to test GKE with gcloud
# at HEAD, release candidate.
if [[ ! -z "${CLOUDSDK_BUCKET:-}" ]]; then
    gsutil -m cp -r "${CLOUDSDK_BUCKET}" ~
    mv ~/$(basename "${CLOUDSDK_BUCKET}") ~/repo
    mkdir ~/cloudsdk
    tar zvxf ~/repo/google-cloud-sdk.tar.gz -C ~/cloudsdk
    export CLOUDSDK_CORE_DISABLE_PROMPTS=1
    export CLOUDSDK_COMPONENT_MANAGER_SNAPSHOT_URL=file://${HOME}/repo/components-2.json
    ~/cloudsdk/google-cloud-sdk/install.sh --disable-installation-options --bash-completion=false --path-update=false --usage-reporting=false
    export PATH=${HOME}/cloudsdk/google-cloud-sdk/bin:${PATH}
    export CLOUDSDK_CONFIG=/var/lib/jenkins/.config/gcloud
fi

### Set up ###
if [[ "${E2E_UP,,}" == "true" ]]; then
    go run ./hack/e2e.go ${E2E_OPT} -v --down
    go run ./hack/e2e.go ${E2E_OPT} -v --up
    go run ./hack/e2e.go -v --ctl="version --match-server-version=false"
fi

### Run tests ###
# Jenkins will look at the junit*.xml files for test failures, so don't exit
# with a nonzero error code if it was only tests that failed.
if [[ "${E2E_TEST,,}" == "true" ]]; then
    go run ./hack/e2e.go ${E2E_OPT} -v --test --test_args="${GINKGO_TEST_ARGS}" && exitcode=0 || exitcode=$?
    if [[ "${E2E_PUBLISH_GREEN_VERSION:-}" == "true" && ${exitcode} == 0 && -n ${githash:-} ]]; then
        echo "publish githash to ci/latest-green.txt: ${githash}"
        echo "${githash}" > ${WORKSPACE}/githash.txt
        gsutil cp ${WORKSPACE}/githash.txt gs://kubernetes-release/ci/latest-green.txt
    fi
fi

# TODO(zml): We have a bunch of legacy Jenkins configs that are
# expecting junit*.xml to be in ${WORKSPACE} root and it's Friday
# afternoon, so just put the junit report where it's expected.
# If link already exists, non-zero return code should not cause build to fail.
for junit in ${ARTIFACTS}/junit*.xml; do
  ln -s -f ${junit} ${WORKSPACE} || true
done

### Clean up ###
if [[ "${E2E_DOWN,,}" == "true" ]]; then
    # Sleep before deleting the cluster to give the controller manager time to
    # delete any cloudprovider resources still around from the last test.
    # This is calibrated to allow enough time for 3 attempts to delete the
    # resources. Each attempt is allocated 5 seconds for requests to the
    # cloudprovider plus the processingRetryInterval from servicecontroller.go
    # for the wait between attempts.
    sleep 30
    go run ./hack/e2e.go ${E2E_OPT} -v --down
fi
