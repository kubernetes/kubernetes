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

# The conformance test checks whether a kubernetes cluster supports
# a minimum set of features to be called "Kubernetes".  It is similar
# to `hack/e2e-test.sh` but it differs in that:
#  - hack/e2e-test.sh is intended to test a cluster with binaries built at HEAD,
#    while this conformance test does not care what version the binaries are.
#    - this means the user needs to setup a cluster first.
#    - this means the user does not need to write any cluster/... scripts.  Custom
#      clusters can be tested.
#  - hack/e2e-test.sh is intended to run e2e tests built at HEAD, while
#    this conformance test is intended to be run e2e tests built at a particular
#    version.  This ensures that all conformance testees run the same set of tests,
#    regardless of when they test for conformance.
#  - it excludes certain e2e tests:
#    - tests that are specific to certain cloud providers
#    - tests of optional features, such as volume types.
#    - tests of performance, scale, or reliability
#    - known flaky tests.

# The conformance test should be run from a github repository at
# commit TBDCOMMITNUMBER.  Otherwise, it may not include the right
# set of tests.
# e.g.:
#   cd /new/directory
#   git clone git://github.com/GoogleCloudPlatform/kubernetes.git
#   cd kubernetes
#   git checkout TBDCOMMITNUMBER.
# The working tree will be in a "detached HEAD" state.
#
# When run as described above, the conformance test tests whether a cluster is
# supports key features for Kubernetes version 1.0.
#
# TODO: when preparing to release a new major or minor version of Kubernetes,
# then update above commit number, reevaluate the set of e2e tests,
# update documentation at docs/getting-started-guides/README.md to have
# a new column for conformance at that new version, and notify
# community.

# Instructions:
#  - Setup a Kubernetes cluster with $NUM_MINIONS nodes (defined below).
#  - Provide a Kubeconfig file whose current context is set to the
#    cluster to be tested, and with suitable auth setting.
#  - Specify the location of that kubeconfig with, e.g.:
#    declare -x KUBECONFIG="$HOME/.kube/config"
#  - Specify the location of the master with, e.g.:
#    declare -x KUBE_MASTER_IP="1.2.3.4"
#  - Make sure only essential pods are running and there are no failed/pending pods.
#  - Make binaries needed by e2e, e.g.:
#      make clean
#      make quick-release
#  - Run the test and capture output:
#      hack/conformance-test.sh 2>&1 | tee conformance.$(date +%FT%T%z).log

: ${KUBECONFIG:?"Must set KUBECONFIG before running conformance test."}
: ${KUBE_MASTER_IP:?"Must set KUBE_MASTER_IP before running conformance test."}
echo "Conformance test using ${KUBECONFIG} against master at ${KUBE_MASTER_IP}"
echo -n "Conformance test run date:"
date
echo -n "Conformance test SHA:"
HEAD_SHA=$(git rev-parse HEAD)
echo $HEAD_SHA
echo "Conformance test version tag(s):"
git show-ref | grep $HEAD_SHA | grep refs/tags
echo
echo "Conformance test checking conformance with Kubernetes version 1.0"

# It runs a whitelist of tests.  This whitelist was assembled at commit
# b70b7084c93d4ce80b7463f48c23d5ac04edb2b1 starting from this list of tests:
#   grep -h 'It(\|Describe(' -R test
#
# List of test name patterns not included and why not included:
#  Cadvisor: impl detail how stats gotten from containers.
#  MasterCerts: GKE/GCE specific
#  Density: performance
#  Cluster level logging...: optional feature
#  Etcd failure: reliability
#  Load Capacity: performance
#  Monitoring: optional feature.
#  Namespaces.*seconds: performance.
#  Pod disks: uses GCE specific feature.
#  Reboot: node management
#  Nodes: node management.
#  Restart: node management.
#  Scale: performance
#  Services.*load balancer: not all cloud providers have a load balancer.
#  Services.*NodePort: flaky
#  Shell: replies on optional ssh access to nodes.
#  SSH: optional feature.
#  Volumes: contained only skipped tests.
export CONFORMANCE_TEST_SKIP_REGEX="Cadvisor|MasterCerts|Density|Cluster\slevel\slogging.*|Etcd\sfailure.*|Load\sCapacity|Monitoring|Namespaces.*seconds|Pod\sdisks|Reboot|Restart|Nodes|Scale|Services.*load\sbalancer|Services.*NodePort|Shell|SSH|Volumes"

declare -x KUBERNETES_CONFORMANCE_TEST="1"
declare -x NUM_MINIONS=4
hack/ginkgo-e2e.sh
exit $?
