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

# When run as described below, the conformance test tests whether a cluster
# supports key features for Kubernetes version 1.0.

# Instructions:
#  - Setup a Kubernetes cluster with $NUM_MINIONS nodes (defined below).
#  - Provide a Kubeconfig file whose current context is set to the
#    cluster to be tested, and with suitable auth setting.
#  - Specify the location of that kubeconfig with, e.g.:
#    declare -x KUBECONFIG="$HOME/.kube/config"
#  - Make sure only essential pods are running and there are no failed/pending pods.
#  - Go to a git tree that contains the kubernetes source.
#    - git clone git://github.com/kubernetes/kubernetes.git
#  - Checkout the upstream/conformance-test-v1 branch
#    - git checkout upstream/conformance-test-v1
#    - The working tree will be in a "detached HEAD" state.
#  - Make binaries needed by e2e
#      make clean
#      make quick-release
#  - Run the test and capture output:
#      hack/conformance-test.sh 2>&1 | tee conformance.$(date +%FT%T%z).log
#

# About the conformance test:
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
# TODO: when preparing to release a new major or minor version of Kubernetes,
# create a new conformance-test-vX.Y branch, update mentions of that branch in this file,
# reevaluate the set of e2e tests,
# update documentation at docs/getting-started-guides/README.md to have
# a new column for conformance at that new version, and notify
# community.


: ${KUBECONFIG:?"Must set KUBECONFIG before running conformance test."}
echo "Conformance test using current-context of ${KUBECONFIG}"
echo -n "Conformance test run date:"
date
echo -n "Conformance test SHA:"
HEAD_SHA=$(git rev-parse HEAD)
echo $HEAD_SHA
echo "Conformance test version tag(s):"
git show-ref | grep $HEAD_SHA | grep refs/tags
echo
echo "Conformance test checking conformance with Kubernetes version 1.0"

# It runs a whitelist of tests: all tests which are flagged with [Conformance]
# somewhere in the description (i.e. either in the Describe part or the It part).
# The list of tagged conformance tests can be retrieved by:
#
# NUM_MINIONS=4 KUBERNETES_CONFORMANCE_TEST="y" \
# hack/ginkgo-e2e.sh -ginkgo.focus='\[Conformance\]' -ginkgo.dryRun=true

declare -x KUBERNETES_CONFORMANCE_TEST="y"
declare -x NUM_MINIONS=4
hack/ginkgo-e2e.sh -ginkgo.focus='\[Conformance\]'
exit $?
