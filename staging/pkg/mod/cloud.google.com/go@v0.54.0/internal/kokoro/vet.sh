#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error
set -e

# Display commands being run
set -x

# Only run the linter on go1.13, because:
# - It needs type aliases (so we can't use anything less than 1.9).
# - It only has to run once per CI (so we just have to pick 1 version).
# - It runs out of memory in go 1.12 https://github.com/dominikh/go-tools/issues/419.
if [[ `go version` != *"go1.13"* ]]; then
    exit 0
fi

go install \
  github.com/golang/protobuf/protoc-gen-go \
  golang.org/x/lint/golint \
  golang.org/x/tools/cmd/goimports \
  honnef.co/go/tools/cmd/staticcheck

# Fail if a dependency was added without the necessary go.mod/go.sum change
# being part of the commit.
go mod tidy
for i in `find . -name go.mod`; do
  pushd `dirname $i`;
    go mod tidy;
  popd;
done
git diff go.mod | tee /dev/stderr | (! read)
git diff go.sum | tee /dev/stderr | (! read)

gofmt -s -d -l . 2>&1 | tee /dev/stderr | (! read)
goimports -l . 2>&1 | tee /dev/stderr | (! read)

# Runs the linter. Regrettably the linter is very simple and does not provide the ability to exclude rules or files,
# so we rely on inverse grepping to do this for us.
#
# Piping a bunch of greps may be slower than `grep -vE (thing|otherthing|anotherthing|etc)`, but since we have a good
# amount of things we're excluding, it seems better to optimize for readability.
#
# Note: since we added the linter after-the-fact, some of the ignored errors here are because we can't change an
# existing interface. (as opposed to us not caring about the error)
golint ./... 2>&1 | ( \
  grep -vE "gen\.go" | \
  grep -vE "receiver name [a-zA-Z]+[0-9]* should be consistent with previous receiver name" | \
  grep -vE "exported const AllUsers|AllAuthenticatedUsers|RoleOwner|SSD|HDD|PRODUCTION|DEVELOPMENT should have comment" | \
  grep -v "exported func Value returns unexported type pretty.val, which can be annoying to use" | \
  grep -v "exported func Increment returns unexported type firestore.increment, which can be annoying to use" | \
  grep -v "ExecuteStreamingSql" | \
  grep -v "MethodExecuteSql should be MethodExecuteSQL" | \
  grep -vE "pubsub\/pstest\/fake\.go.+should have comment or be unexported" | \
  grep -v "ClusterId" | \
  grep -v "InstanceId" | \
  grep -v "firestore.arrayUnion" | \
  grep -v "firestore.arrayRemove" | \
  grep -v "maxAttempts" | \
  grep -v "UptimeCheckIpIterator" | \
  grep -vE "apiv[0-9]+" | \
  grep -v "ALL_CAPS" | \
  grep -v "go-cloud-debug-agent" | \
  grep -v "mock_test" | \
  grep -v "internal/testutil/funcmock.go" | \
  grep -v "internal/backoff" | \
  grep -v "internal/trace" | \
  grep -v "internal/gapicgen/generator" | \
  grep -v "a blank import should be only in a main or test package" | \
  grep -v "method ExecuteSql should be ExecuteSQL" | \
  grep -vE "spanner/spansql/(sql|types).go:.*should have comment" | \
  grep -vE "\.pb\.go:") | \
  tee /dev/stderr | (! read)

staticcheck -go 1.11 ./... 2>&1 | ( \
  grep -v S1007 | \
  grep -v SA1019 | \
  grep -v firestore/internal/doc-snippets.go | \
  grep -v functions/metadata/metadata_test.go | \
  grep -v spanner/value.go | \
  grep -v go-cloud-debug-agent | \
  grep -v pubsub/integration_test.go | \
  grep -v internal/fields/fold.go | \
  grep -v httpreplay/internal/proxy/debug.go | \
  grep -v bigtable/internal/cbtconfig/cbtconfig.go | \
  grep -v bigtable/cmd/cbt/cbt.go | \
  grep -v asset/v1beta1/doc.go | \
  grep -v asset/v1beta1/mock_test.go | \
  grep -v spanner/value_test.go | \
  grep -v bigtable/reader.go | \
  grep -v internal/btree/btree.go | \
  grep -v container/apiv1/mock_test.go) | \
  tee /dev/stderr | (! read)

echo "Done vetting!"
