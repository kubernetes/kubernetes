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

# Build rules for Kubernetes binaries. Include this file from the top-
# level kubernetes folder, it won't work otherwise. Note that everything in
# here expects GOPATH set properly.

# We will make use of some nice bash-isms such as brace expansion.
SHELL=/bin/bash

GOHOSTOS = $(shell go env GOHOSTOS)
GOHOSTARCH = $(shell go env GOHOSTARCH)
GOPATH = $(shell go env GOPATH)

# Override these to cross-compile. Otherwise we use the system default.
GOOS ?= $(shell go env GOOS)
GOARCH ?= $(shell go env GOARCH)
CC ?= $(shell go env CC)

# This contains such useful things as the build date and the git version.
GOLDFLAGS = -ldflags "$(shell source hack/lib/version.sh && KUBE_ROOT=$(shell pwd) KUBE_GO_PACKAGE=k8s.io/kubernetes kube::version::ldflags)"

ifeq "$(GOOS)" "windows"
  extension = .exe
else
  extension = 
endif

# Environment for every go tool call.
go_env = GOOS=$(GOOS) GOARCH=$(GOARCH) CC=$(CC)

# The go tool installs to GOPATH/bin for the host OS/arch and
# GOPATH/bin/OS_ARCH for others.
ifeq "$(GOHOSTOS)/$(GOHOSTARCH)" "$(GOOS)/$(GOARCH)"
  bin_dir = $(GOPATH)/bin
else
  bin_dir = $(GOPATH)/bin/$(GOOS)_$(GOARCH)
endif

# List of packages to install relative to k8s.io/kubernetes. If you want to add
# another client binary you'll have to do more than just add the entry here.
client_binary_paths = cmd/kubectl
dynamic_server_binary_paths = cmd/kubelet cmd/hyperkube cmd/kubemark
static_server_binary_paths = cmd/kube-dns cmd/kube-proxy cmd/kube-apiserver cmd/kube-controller-manager plugin/cmd/kube-scheduler federation/cmd/federation-controller-manager federation/cmd/federation-apiserver
test_binary_paths = cmd/integration cmd/gendocs cmd/genkubedocs cmd/genman cmd/genyaml cmd/mungedocs cmd/genswaggertypedocs cmd/linkcheck examples/k8petstore/web-server/src federation/cmd/genfeddocs vendor/github.com/onsi/ginkgo/ginkgo
test_package_paths = test/e2e test/e2e_node

# Just the names of the binaries (eg kubelet hyperkube ...).
dynamic_server_binaries = $(notdir $(dynamic_server_binary_paths))
static_server_binaries = $(notdir $(static_server_binary_paths))
server_binaries = $(static_server_binaries) $(dynamic_server_binaries)
test_binaries = $(addsuffix $(extension),$(notdir $(test_binary_paths)))
test_packages = $(addsuffix .test$(extension),$(notdir $(test_package_paths)))

# The paths separated by commas instead of spaces (eg cmd/kubelet,cmd/hyperkube,...).
dynamic_server_paths_commas = $(shell echo $(dynamic_server_binary_paths) | sed -e "s/ /,/g")
static_server_paths_commas = $(shell echo $(static_server_binary_paths) | sed -e "s/ /,/g")
test_binary_paths_commas = $(shell echo $(test_binary_paths) | sed -e "s/ /,/g")
test_package_paths_commas = $(shell echo $(test_package_paths) | sed -e "s/ /,/g")

# Binary names separated by commas instead of spaces (eg kubelet,hyperkube,...).
server_binaries_commas = $(shell echo "$(server_binaries)" | sed -e "s/ /,/g")
test_binaries_commas = $(shell echo $(test_binaries) | sed -e "s/ /,/g")
test_packages_commas = $(shell echo $(test_packages) | sed -e "s/ /,/g")

# This magic defines a rule for each binary, so that "make kube-apiserver" and
# "make cmd/kube-apiserver" and "make e2e" all work.
define static_go_install
.PHONY: $(notdir $(1)) $(1)
$(1) $(notdir $(1)):
	$(go_env) CGO_ENABLED=0 go install -installsuffix cgo $$(GOLDFLAGS) k8s.io/kubernetes/$(1)
endef
define dynamic_go_install
.PHONY: $(notdir $(1)) $(1)
$(1) $(notdir $(1)):
	$(go_env) go install $$(GOLDFLAGS) k8s.io/kubernetes/$(1)
endef

# For e2e test packages, manually check if they're stale before trying to build.
# The teststale binary is only build for the host OS. When installing the test
# binary we need to do a go install as well as a go test -c, because the go
# tool checks if go install will do anything.
# This makes "make e2e.test" and "make test/e2e" both work incrementally.
$(GOPATH)/bin/teststale$(extension):
	go install k8s.io/kubernetes/hack/cmd/teststale
.PHONY: $(GOPATH)/bin/teststale$(extension)
define go_test
.PHONY: $(notdir $(1)) $(1) $(1)-force
$(1) $(notdir $(1)).test$(extension): $(GOPATH)/bin/teststale$(extension)
	@if $(GOPATH)/bin/teststale$(teststale) -binary $(bin_dir)/$(notdir $(1)).test$(extension) -package k8s.io/kubernetes/$(1); then \
	    $$(MAKE) $(1)-force --no-print-directory; \
	fi
$(1)-force:
	  $(go_env) go install $$(GOLDFLAGS) k8s.io/kubernetes/$(1)
	  $(go_env) go test -c $$(GOLDFLAGS) -o $(bin_dir)/$(notdir $(1)).test$(extension) k8s.io/kubernetes/$(1)
endef

$(foreach path,$(client_binary_paths),$(eval $(call dynamic_go_install,$(path))))
$(foreach path,$(dynamic_server_binary_paths),$(eval $(call dynamic_go_install,$(path))))
$(foreach path,$(static_server_binary_paths),$(eval $(call static_go_install,$(path))))
$(foreach path,$(test_binary_paths),$(eval $(call dynamic_go_install,$(path))))
$(foreach path,$(test_package_paths),$(eval $(call go_test,$(path))))

# The go tool is much faster at building multiple binaries if they are provided
# in one install command.
.PHONY: dynamic-server-binaries static-server-binaries server-binaries test-binaries
dynamic-server-binaries:
	$(go_env) go install $(GOLDFLAGS) k8s.io/kubernetes/{$(dynamic_server_paths_commas)}
static-server-binaries:
	$(go_env) CGO_ENABLED=0 go install $(GOLDFLAGS) -installsuffix cgo k8s.io/kubernetes/{$(static_server_paths_commas)}
server-binaries: dynamic-server-binaries static-server-binaries
test-binaries:
	$(go_env) go install $(GOLDFLAGS) k8s.io/kubernetes/{$(test_binary_paths_commas)}

# Copy binaries to a location under the kubernetes repo so that dockerized
# builds will stick around after the container disappears.
output_dir = _output/dockerized/bin/$(GOOS)/$(GOARCH)
$(output_dir):
	@mkdir -p $(output_dir)
copy-server-binaries: $(output_dir) server-binaries
	@cp $(bin_dir)/{$(server_binaries_commas)} $(output_dir)
copy-kubectl: $(output_dir) kubectl
	@cp $(bin_dir)/kubectl$(extension) $(output_dir)
copy-test-binaries: $(output_dir) test-binaries $(test_packages)
	@cp $(bin_dir)/{$(test_binaries_commas)} $(output_dir)
	@cp $(bin_dir)/{$(test_packages_commas)} $(output_dir)

# The cross-platform stuff can probably be factored a bit, but I think it's
# complicated enough already. To add a new platform, you'll need to add an
# eval/call as well as an entry to the -cross rule.
.PHONY: server-cross kubectl-cross test-cross
define build_server_binaries_for_platform
.PHONY: server-binaries-$(1)-$(2)
server-binaries-$(1)-$(2):
	@GOOS=$(1) GOARCH=$(2) CC=$(3) $$(MAKE) copy-server-binaries --no-print-directory
endef
$(eval $(call build_server_binaries_for_platform,linux,amd64,gcc))
$(eval $(call build_server_binaries_for_platform,linux,arm,arm-linux-gnueabi-gcc))
$(eval $(call build_server_binaries_for_platform,linux,arm64,aarch64-linux-gnu-gcc))
server-cross: $(addprefix server-binaries-,linux-amd64 linux-arm linux-arm64)

define build_kubectl_for_platform
.PHONY: kubectl-$(1)-$(2)
kubectl-$(1)-$(2):
	@GOOS=$(1) GOARCH=$(2) $$(MAKE) copy-kubectl --no-print-directory
endef
$(eval $(call build_kubectl_for_platform,linux,amd64))
$(eval $(call build_kubectl_for_platform,linux,386))
$(eval $(call build_kubectl_for_platform,linux,arm))
$(eval $(call build_kubectl_for_platform,linux,arm64))
$(eval $(call build_kubectl_for_platform,linux,ppc64le))
$(eval $(call build_kubectl_for_platform,darwin,amd64))
$(eval $(call build_kubectl_for_platform,darwin,386))
$(eval $(call build_kubectl_for_platform,windows,amd64))
$(eval $(call build_kubectl_for_platform,windows,386))
kubectl-cross: $(addprefix kubectl-,linux-amd64 linux-386 linux-arm linux-arm64 linux-ppc64le darwin-amd64 darwin-386 windows-amd64 windows-386)

define build_test_for_platform
.PHONY: test-$(1)-$(2)
test-$(1)-$(2): $(bin_dir)/teststale
	@GOOS=$(1) GOARCH=$(2) $$(MAKE) copy-test-binaries --no-print-directory
endef
$(eval $(call build_test_for_platform,linux,amd64))
$(eval $(call build_test_for_platform,linux,arm))
$(eval $(call build_test_for_platform,darwin,amd64))
$(eval $(call build_test_for_platform,windows,amd64))
test-cross: $(addprefix test-,linux-amd64 linux-arm darwin-amd64 windows-amd64)

# For a quick-release, build server binaries for linux/amd64. Build kubectl and
# tests for linux/amd64 and the host OS if it is different.
binaries-quick: server-binaries-linux-amd64 kubectl-linux-amd64 kubectl-$(GOOS)-$(GOARCH) test-linux-amd64 test-$(GOOS)-$(GOARCH)
binaries-cross: kubectl-cross server-cross test-cross
