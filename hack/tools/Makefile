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
	$(GOBIN)/gotestsum\
	$(GOBIN)/misspell\
	$(GOBIN)/prettybench\
	$(GOBIN)/staticcheck\
	$(GOBIN)/logcheck\

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

$(GOBIN)/gotestsum: deps
	go install gotest.tools/gotestsum

$(GOBIN)/misspell: deps
	go install github.com/client9/misspell/cmd/misspell

$(GOBIN)/prettybench: deps
	go install github.com/cespare/prettybench

$(GOBIN)/staticcheck: deps
	go install honnef.co/go/tools/cmd/staticcheck

$(GOBIN)/logcheck: deps
	go install k8s.io/klog/hack/tools/logcheck
