#!/bin/bash

# Copyright 2019 Google LLC.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Fail on error, and display commands being run.
set -ex

# Only run the linter on go1.13, since it needs type aliases (and we only care
# about its output once).
if [[ `go version` != *"go1.13"* ]]; then
    exit 0
fi

go install \
  golang.org/x/lint/golint \
  golang.org/x/tools/cmd/goimports \
  honnef.co/go/tools/cmd/staticcheck

# Fail if a dependency was added without the necessary go.mod/go.sum change
# being part of the commit.
go mod tidy
git diff go.mod | tee /dev/stderr | (! read)
git diff go.sum | tee /dev/stderr | (! read)

# Easier to debug CI.
pwd

gofmt -s -d -l . 2>&1 | tee /dev/stderr | (! read)
goimports -l . 2>&1 | tee /dev/stderr | (! read)

# Runs the linter. Regrettably the linter is very simple and does not provide the ability to exclude rules or files,
# so we rely on inverse grepping to do this for us.
golint ./... 2>&1 | ( \
  grep -v "gen.go" | \
  grep -v "disco.go" | \
  grep -v "exported const DefaultDelayThreshold should have comment" | \
  grep -v "exported const DefaultBundleCountThreshold should have comment" | \
  grep -v "exported const DefaultBundleByteThreshold should have comment" | \
  grep -v "exported const DefaultBufferedByteLimit should have comment" | \
  grep -v "error var Done should have name of the form ErrFoo" | \
  grep -v "exported method APIKey.RoundTrip should have comment or be unexported" | \
  grep -v "exported method MarshalStyle.JSONReader should have comment or be unexported" | \
  grep -v "UnmarshalJSON should have comment or be unexported" | \
  grep -v "MarshalJSON should have comment or be unexported" | \
  grep -vE "\.pb\.go:" || true) | tee /dev/stderr | (! read)

staticcheck -go 1.9 ./... 2>&1 | ( \
  grep -v "SA1019" | \
  grep -v "S1007" | \
  grep -v "error var Done should have name of the form ErrFoo" | \
  grep -v "examples" | \
  grep -v "gen.go" || true) | tee /dev/stderr | (! read)
