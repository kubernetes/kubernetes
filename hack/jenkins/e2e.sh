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

if [[ "${CIRCLECI:-}" == "true" ]]; then
    JOB_NAME="circleci-${CIRCLE_PROJECT_USERNAME}-${CIRCLE_PROJECT_REPONAME}"
    BUILD_NUMBER=${CIRCLE_BUILD_NUM}
    WORKSPACE=`pwd`
else
    # Jenkins?
    export HOME=${WORKSPACE} # Nothing should want Jenkins $HOME
fi

# Additional parameters that are passed to ginkgo runner.
GINKGO_TEST_ARGS=""

if [[ "${PERFORMANCE:-}" == "true" ]]; then
    if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
      export MASTER_SIZE="m3.xlarge"
    else
      export MASTER_SIZE="n1-standard-4"
    fi
    export NUM_MINIONS="100"
    GINKGO_TEST_ARGS="--ginkgo.focus=\[Performance suite\] "
else
    if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
      export MASTER_SIZE="t2.small"
    else
      export MASTER_SIZE="g1-small"
    fi
    export NUM_MINIONS="2"
fi


# Unlike the kubernetes-build script, we expect some environment
# variables to be set. We echo these immediately and presume "set -o
# nounset" will force the caller to set them: (The first several are
# Jenkins variables.)

echo "JOB_NAME: ${JOB_NAME}"
echo "BUILD_NUMBER: ${BUILD_NUMBER}"
echo "WORKSPACE: ${WORKSPACE}"
echo "KUBERNETES_PROVIDER: ${KUBERNETES_PROVIDER}" # Cloud provider
echo "E2E_CLUSTER_NAME: ${E2E_CLUSTER_NAME}"       # Name of the cluster (e.g. "e2e-test-jenkins")
echo "E2E_NETWORK: ${E2E_NETWORK}"                 # Name of the network (e.g. "e2e")
echo "E2E_ZONE: ${E2E_ZONE}"                       # Name of the GCE zone (e.g. "us-central1-f")
echo "E2E_OPT: ${E2E_OPT}"                         # hack/e2e.go options
echo "E2E_SET_CLUSTER_API_VERSION: ${E2E_SET_CLUSTER_API_VERSION:-<not set>}" # optional, for GKE, set CLUSTER_API_VERSION to git hash
echo "--------------------------------------------------------------------------------"


# AWS variables
export KUBE_AWS_INSTANCE_PREFIX=${E2E_CLUSTER_NAME}
export KUBE_AWS_ZONE=${E2E_ZONE}

# GCE variables
export INSTANCE_PREFIX=${E2E_CLUSTER_NAME}
export KUBE_GCE_ZONE=${E2E_ZONE}
export KUBE_GCE_NETWORK=${E2E_NETWORK}

# GKE variables
export CLUSTER_NAME=${E2E_CLUSTER_NAME}
export ZONE=${E2E_ZONE}
export KUBE_GKE_NETWORK=${E2E_NETWORK}

export PATH=${PATH}:/usr/local/go/bin
export KUBE_SKIP_CONFIRMATIONS=y

if [[ ${KUBE_RUN_FROM_OUTPUT:-} =~ ^[yY]$ ]]; then
    echo "Found KUBE_RUN_FROM_OUTPUT=y; will use binaries from _output"
    cp _output/release-tars/kubernetes*.tar.gz .
else
    echo "Pulling binaries from GCS"
    if [[ $(find . | wc -l) != 1 ]]; then
        echo $PWD not empty, bailing!
        exit 1
    fi

    # Tell kube-up.sh to skip the update, it doesn't lock. An internal
    # gcloud bug can cause racing component updates to stomp on each
    # other.
    export KUBE_SKIP_UPDATE=y
    sudo flock -x -n /var/run/lock/gcloud-components.lock -c "gcloud components update -q" || true

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
    gsutil -m cp gs://kubernetes-release/${bucket}/${githash}/kubernetes.tar.gz gs://kubernetes-release/${bucket}/${githash}/kubernetes-test.tar.gz .
fi

md5sum kubernetes*.tar.gz
tar -xzf kubernetes.tar.gz
tar -xzf kubernetes-test.tar.gz
cd kubernetes

# Set by GKE-CI to change the CLUSTER_API_VERSION to the git version
if [[ ! -z ${E2E_SET_CLUSTER_API_VERSION:-} ]]; then
    export CLUSTER_API_VERSION=$(echo ${githash} | cut -c 2-)
elif [[ ${JENKINS_USE_RELEASE_TARS:-} =~ ^[yY]$ ]]; then
    release=$(gsutil cat gs://kubernetes-release/release/${version_file}.txt | cut -c 2-)
    export CLUSTER_API_VERSION=${release}
fi

# Have cmd/e2e run by goe2e.sh generate JUnit report in ${WORKSPACE}/junit*.xml
export E2E_REPORT_DIR=${WORKSPACE}

### Set up ###
go run ./hack/e2e.go ${E2E_OPT} -v --down
go run ./hack/e2e.go ${E2E_OPT} -v --up
go run ./hack/e2e.go -v --ctl="version --match-server-version=false"

### Run tests ###
# Jenkins will look at the junit*.xml files for test failures, so don't exit
# with a nonzero error code if it was only tests that failed.
go run ./hack/e2e.go ${E2E_OPT} -v --test --test_args="${GINKGO_TEST_ARGS}--ginkgo.noColor" || true

### Clean up ###
go run ./hack/e2e.go ${E2E_OPT} -v --down
