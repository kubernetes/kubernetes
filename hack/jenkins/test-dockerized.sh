#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# Runs the unit and integration tests, producing JUnit-style XML test
# reports in ${WORKSPACE}/artifacts. This script is intended to be run from
# kubekins-test container with a kubernetes repo mapped in. See
# k8s.io/test-infra/scenarios/kubernetes_verify.py

export PATH=${GOPATH}/bin:${PWD}/third_party/etcd:/usr/local/go/bin:${PATH}

# Until all GOPATH references are removed from all build scripts as well,
# explicitly disable module mode to avoid picking up user-set GO111MODULE preferences.
# As individual scripts make use of go modules, they can explicitly set GO111MODULE=on
export GO111MODULE=off

# Install tools we need
pushd "./hack/tools" >/dev/null
  GO111MODULE=on go install gotest.tools/gotestsum
popd >/dev/null

# Disable coverage report
export KUBE_COVER="n"
# Set artifacts directory
export ARTIFACTS=${ARTIFACTS:-"${WORKSPACE}/artifacts"}
# Save the verbose stdout as well.
export KUBE_KEEP_VERBOSE_TEST_OUTPUT=y
export KUBE_INTEGRATION_TEST_MAX_CONCURRENCY=1
export LOG_LEVEL=4

cd "${GOPATH}/src/k8s.io/kubernetes"

make generated_files
go install ./cmd/...
./hack/install-etcd.sh

# TODO(spiffxp): narrowing down
# make test-cmd
integration_tests=(
  k8s.io/kubernetes/test/integration/apimachinery
  k8s.io/kubernetes/test/integration/apiserver
  k8s.io/kubernetes/test/integration/apiserver/admissionwebhook
  k8s.io/kubernetes/test/integration/apiserver/apply
  k8s.io/kubernetes/test/integration/apiserver/certreload
  k8s.io/kubernetes/test/integration/apiserver/flowcontrol
  k8s.io/kubernetes/test/integration/apiserver/podlogs
  k8s.io/kubernetes/test/integration/apiserver/tracing
  k8s.io/kubernetes/test/integration/auth
  k8s.io/kubernetes/test/integration/certificates
  k8s.io/kubernetes/test/integration/client
  k8s.io/kubernetes/test/integration/configmap
  k8s.io/kubernetes/test/integration/controlplane
  k8s.io/kubernetes/test/integration/cronjob
  k8s.io/kubernetes/test/integration/daemonset
  k8s.io/kubernetes/test/integration/defaulttolerationseconds
  k8s.io/kubernetes/test/integration/deployment
  k8s.io/kubernetes/test/integration/disruption
  k8s.io/kubernetes/test/integration/dryrun
  k8s.io/kubernetes/test/integration/dualstack
  k8s.io/kubernetes/test/integration/endpoints
  k8s.io/kubernetes/test/integration/endpointslice
  k8s.io/kubernetes/test/integration/etcd
  k8s.io/kubernetes/test/integration/events
  k8s.io/kubernetes/test/integration/evictions
  k8s.io/kubernetes/test/integration/examples
  k8s.io/kubernetes/test/integration/garbagecollector
  k8s.io/kubernetes/test/integration/ipamperf
  k8s.io/kubernetes/test/integration/job
  k8s.io/kubernetes/test/integration/kubelet
  k8s.io/kubernetes/test/integration/metrics
  k8s.io/kubernetes/test/integration/namespace
  k8s.io/kubernetes/test/integration/node
  k8s.io/kubernetes/test/integration/objectmeta
  k8s.io/kubernetes/test/integration/openshift
  k8s.io/kubernetes/test/integration/pods
  k8s.io/kubernetes/test/integration/quota
  k8s.io/kubernetes/test/integration/replicaset
  k8s.io/kubernetes/test/integration/replicationcontroller
  k8s.io/kubernetes/test/integration/scale
  k8s.io/kubernetes/test/integration/scheduler
  k8s.io/kubernetes/test/integration/scheduler_perf
  k8s.io/kubernetes/test/integration/secrets
  k8s.io/kubernetes/test/integration/service
  k8s.io/kubernetes/test/integration/serviceaccount
  k8s.io/kubernetes/test/integration/serving
  k8s.io/kubernetes/test/integration/statefulset
  k8s.io/kubernetes/test/integration/storageclasses
  k8s.io/kubernetes/test/integration/storageversion
  k8s.io/kubernetes/test/integration/tls
  k8s.io/kubernetes/test/integration/ttlcontroller
  k8s.io/kubernetes/test/integration/volume
  k8s.io/kubernetes/test/integration/volumescheduling
  k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/test/integration
  k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/test/integration/conversion
)  
make test-integration WHAT="${integration_tests[*]}"
