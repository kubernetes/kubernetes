# Copyright 2021 The Kubernetes Authors.
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

# If you update this file, please follow
# https://suva.sh/posts/well-documented-makefiles

.DEFAULT_GOAL:=help
SHELL:=/usr/bin/env bash

COLOR:=\\033[36m
NOCOLOR:=\\033[0m

build: ## Build zeitgeist
	go build

lint:
	test -z $(shell go fmt .) || (echo "Linting failed !" && exit 8)
	go vet ./...
	go get -u golang.org/x/lint/golint
	golint ./...

test: ## Runs unit testing
	go test ./... -covermode=count -coverprofile=coverage.out

test-results: test
	go tool cover -html=coverage.out

verify: verify-boilerplate verify-golangci-lint verify-go-mod  ## Runs verification scripts to ensure correct execution

verify-boilerplate: ## Runs the file header check
	./hack/verify-boilerplate.sh

verify-go-mod: ## Runs the go module linter
	./hack/verify-go-mod.sh

verify-golangci-lint: ## Runs all golang linters
	./hack/verify-golangci-lint.sh

##@ Helpers

.PHONY: help

help:  ## Display this help
	@awk \
		-v "col=${COLOR}" -v "nocol=${NOCOLOR}" \
		' \
			BEGIN { \
				FS = ":.*##" ; \
				printf "\nUsage:\n  make %s<target>%s\n", col, nocol \
			} \
			/^[a-zA-Z_-]+:.*?##/ { \
				printf "  %s%-15s%s %s\n", col, $$1, nocol, $$2 \
			} \
			/^##@/ { \
				printf "\n%s%s%s\n", col, substr($$0, 5), nocol \
			} \
		' $(MAKEFILE_LIST)
