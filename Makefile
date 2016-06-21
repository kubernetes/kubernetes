# Copyright 2016 The Kubernetes Authors All rights reserved.
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

include build/build.mk

BUILD_IMAGE = gcr.io/google_containers/kube-cross:v1.6.2-experimental

KUBE_FASTBUILD ?= false

KUBE_GOFLAGS ?= $(GOFLAGS)
export KUBE_GOFLAGS

KUBE_GOLDFLAGS ?= $(GOLDFLAGS)
export KUBE_GOLDFLAGS

# For all kubernetes binaries, we define a short rule shortcut for go install.
# The rules are defined in build/build.mk. To use these you will need your
# GOPATH set up properly, see https://golang.org/doc/code.html . Note that
# kubernetes goes under k8s.io, not github.com. If you don't want to set up
# your gopath, then use either the quick-release or the release rule to do a
# containerized build.
#
# Example:
#   make kubectl
#   make kube-apiserver
#   make cmd/kubelet
#   make e2e.test
#   make kubectl-windows-amd64
#   make server-binaries
#   make test-binaries
#
# To install a library package, use go install directly.
#
# Example:
#   go install ./pkg/kubelet

# Build a full hermetic release inside a container. The build container needs
# access to docker in order to build and save kube-system images. We pass in
# the kubernetes repo (pwd) as a volume so output in _output is owned by root.
#
# Example:
#   make release
all release:
	@docker run --rm -it \
	    -e "KUBE_FASTBUILD=$(KUBE_FASTBUILD)" \
	    -v $(shell pwd):/go/src/k8s.io/kubernetes \
	    -v $(shell which docker):/bin/docker:ro \
	    -v /var/run/docker.sock:/var/run/docker.sock \
	    -w /go/src/k8s.io/kubernetes \
	    $(BUILD_IMAGE) bash -c 'make release-local -j4'
.PHONY: all release

# Build a hermetic release for a limited set of platforms.
#
# Example:
#   make quick-release
quick-release:
	@KUBE_FASTBUILD=true $(MAKE) release --no-print-directory
.PHONY: quick-release

# Build an incremental release without a container. You will need
# cross-compilers installed. See build/build-image.
#
# Example:
#   make release-local
release-local:
	@+./build/build.sh
.PHONY: release-local

# Build an incremental release for a limited set of platforms without a
# container. This is the fastest for local development and testing.
#
# Example:
#   make quick-release-local
quick-release-local:
	@+KUBE_FASTBUILD=true ./build/build.sh
.PHONY: quick-release-local

# Runs all the presubmission verifications.
#
# Args:
#   BRANCH: Branch to be passed to hack/verify-godeps.sh script.
#
# Example:
#   make verify
#   make verify BRANCH=branch_x
verify:
	KUBE_VERIFY_GIT_BRANCH=$(BRANCH) hack/verify-all.sh -v
.PHONY: verify

# Build and run tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#   TESTS: Same as WHAT.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make check
#   make test
#   make check WHAT=pkg/kubelet GOFLAGS=-v
check test:
	hack/test-go.sh $(WHAT) $(TESTS)
.PHONY: check test

# Build and run integration tests.
#
# Example:
#   make test_integration
test_integration:
	hack/test-integration.sh
.PHONY: test_integration test_integ

# Build and run end-to-end tests.
#
# Example:
#   make test_e2e
test_e2e:
	go run hack/e2e.go -v --build --up --test --down
.PHONY: test_e2e

# Build and run node end-to-end tests.
#
# Args:
#  FOCUS: regexp that matches the tests to be run.  Defaults to "".
#  SKIP: regexp that matches the tests that needs to be skipped.  Defaults to "".
#  RUN_UNTIL_FAILURE: Ff true, pass --untilItFails to ginkgo so tests are run repeatedly until they fail.  Defaults to false.
#  REMOTE: If true, run the tests on a remote host instance on GCE.  Defaults to false.
#  IMAGES: for REMOTE=true only.  Comma delimited list of images for creating remote hosts to run tests against.  Defaults to "e2e-node-containervm-v20160321-image".
#  LIST_IMAGES: If true, don't run tests.  Just output the list of available images for testing.  Defaults to false.
#  HOSTS: for REMOTE=true only.  Comma delimited list of running gce hosts to run tests against.  Defaults to "".
#  DELETE_INSTANCES: for REMOTE=true only.  Delete any instances created as part of this test run.  Defaults to false.
#  ARTIFACTS: for REMOTE=true only.  Local directory to scp test artifacts into from the remote hosts.  Defaults to ""/tmp/_artifacts".
#  REPORT: for REMOTE=false only.  Local directory to write juntil xml results to.  Defaults to "/tmp/".
#  CLEANUP: for REMOTE=true only.  If false, do not stop processes or delete test files on remote hosts.  Defaults to true.
#  IMAGE_PROJECT: for REMOTE=true only.  Project containing images provided to IMAGES.  Defaults to "kubernetes-node-e2e-images".
#  INSTANCE_PREFIX: for REMOTE=true only.  Instances created from images will have the name "${INSTANCE_PREFIX}-${IMAGE_NAME}".  Defaults to "test"/
#
# Example:
#   make test_e2e_node FOCUS=kubelet SKIP=container
#   make test_e2e_node REMOTE=true DELETE_INSTANCES=true
# Build and run tests.
test_e2e_node:
	hack/e2e-node-test.sh
.PHONY: test_e2e_node

# Remove all build artifacts.
#
# Example:
#   make clean
clean:
	build/make-clean.sh
	rm -rf $(OUT_DIR)
	rm -rf Godeps/_workspace # Just until we are sure it is gone
.PHONY: clean

# Run 'go vet'.
#
# Args:
#   WHAT: Directory names to vet.  All *.go files under these
#     directories will be vetted.  If not specified, "everything" will be
#     vetted.
#   TESTS: Same as WHAT.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make vet
#   make vet WHAT=pkg/kubelet
vet:
	hack/verify-govet.sh $(WHAT) $(TESTS)
.PHONY: vet
