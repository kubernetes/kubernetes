#!/bin/bash
# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd "$(git rev-parse --show-toplevel)"
set -e
go run ./internal/cmd/generate-alias -execute
go test ./protoc-gen-go -regenerate
