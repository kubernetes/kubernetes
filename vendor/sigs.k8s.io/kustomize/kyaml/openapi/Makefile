# Copyright 2020 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0

.PHONY: all
all: generate

.PHONY: generate
generate:
	go generate .

.PHONY: verify
verify:
	GOWORK=off go test ./...
