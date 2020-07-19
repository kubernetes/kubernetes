# Copyright 2020 The Kubernetes Authors.
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

export GO111MODULE=on

GOBIN ?= $(shell go env GOPATH)/bin
DEP_PROGS=\
	$(GOBIN)/buildozer\
	$(GOBIN)/gazelle\
	$(GOBIN)/golint\
	$(GOBIN)/gotestsum\
	$(GOBIN)/kazel\
	$(GOBIN)/misspell\
	$(GOBIN)/prettybench\
	$(GOBIN)/staticcheck\

.PHONY: all
all: clean build

.PHONY: clean
clean:
	rm -f $(DEP_PROGS)
	rm -rf vendor/

.PHONY: build
build: $(DEP_PROGS)

.PHONY: deps
deps:
	go mod vendor

$(GOBIN)/buildozer: deps
	go install github.com/bazelbuild/buildtools/buildozer

$(GOBIN)/gazelle: deps
	go install github.com/bazelbuild/bazel-gazelle/cmd/gazelle

$(GOBIN)/golint: deps
	go install golang.org/x/lint/golint

$(GOBIN)/gotestsum: deps
	go install gotest.tools/gotestsum

$(GOBIN)/kazel: deps
	go install k8s.io/repo-infra/cmd/kazel

$(GOBIN)/misspell: deps
	go install github.com/client9/misspell/cmd/misspell

$(GOBIN)/prettybench: deps
	go install github.com/cespare/prettybench

$(GOBIN)/staticcheck: deps
	go install honnef.co/go/tools/cmd/staticcheck
