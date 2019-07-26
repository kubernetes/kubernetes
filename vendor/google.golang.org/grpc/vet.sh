#!/bin/bash

if [[ `uname -a` = *"Darwin"* ]]; then
  echo "It seems you are running on Mac. This script does not work on Mac. See https://github.com/grpc/grpc-go/issues/2047"
  exit 1
fi

set -ex  # Exit on error; debugging enabled.
set -o pipefail  # Fail a pipe if any sub-command fails.

die() {
  echo "$@" >&2
  exit 1
}

fail_on_output() {
  tee /dev/stderr | (! read)
}

# Check to make sure it's safe to modify the user's git repo.
git status --porcelain | fail_on_output

# Undo any edits made by this script.
cleanup() {
  git reset --hard HEAD
}
trap cleanup EXIT

PATH="${GOPATH}/bin:${GOROOT}/bin:${PATH}"

if [[ "$1" = "-install" ]]; then
  # Check for module support
  if go help mod >& /dev/null; then
    go install \
      golang.org/x/lint/golint \
      golang.org/x/tools/cmd/goimports \
      honnef.co/go/tools/cmd/staticcheck \
      github.com/client9/misspell/cmd/misspell \
      github.com/golang/protobuf/protoc-gen-go
  else
    # Ye olde `go get` incantation.
    # Note: this gets the latest version of all tools (vs. the pinned versions
    # with Go modules).
    go get -u \
      golang.org/x/lint/golint \
      golang.org/x/tools/cmd/goimports \
      honnef.co/go/tools/cmd/staticcheck \
      github.com/client9/misspell/cmd/misspell \
      github.com/golang/protobuf/protoc-gen-go
  fi
  if [[ -z "${VET_SKIP_PROTO}" ]]; then
    if [[ "${TRAVIS}" = "true" ]]; then
      PROTOBUF_VERSION=3.3.0
      PROTOC_FILENAME=protoc-${PROTOBUF_VERSION}-linux-x86_64.zip
      pushd /home/travis
      wget https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/${PROTOC_FILENAME}
      unzip ${PROTOC_FILENAME}
      bin/protoc --version
      popd
    elif ! which protoc > /dev/null; then
      die "Please install protoc into your path"
    fi
  fi
  exit 0
elif [[ "$#" -ne 0 ]]; then
  die "Unknown argument(s): $*"
fi

# - Ensure all source files contain a copyright message.
git ls-files "*.go" | xargs grep -L "\(Copyright [0-9]\{4,\} gRPC authors\)\|DO NOT EDIT" 2>&1 | fail_on_output

# - Make sure all tests in grpc and grpc/test use leakcheck via Teardown.
(! grep 'func Test[^(]' *_test.go)
(! grep 'func Test[^(]' test/*.go)

# - Do not import math/rand for real library code.  Use internal/grpcrand for
#   thread safety.
git ls-files "*.go" | xargs grep -l '"math/rand"' 2>&1 | (! grep -v '^examples\|^stress\|grpcrand\|wrr_test')

# - Ensure all ptypes proto packages are renamed when importing.
git ls-files "*.go" | (! xargs grep "\(import \|^\s*\)\"github.com/golang/protobuf/ptypes/")

# - Check imports that are illegal in appengine (until Go 1.11).
# TODO: Remove when we drop Go 1.10 support
go list -f {{.Dir}} ./... | xargs go run test/go_vet/vet.go

# - gofmt, goimports, golint (with exceptions for generated code), go vet.
gofmt -s -d -l . 2>&1 | fail_on_output
goimports -l . 2>&1 | (! grep -vE "(_mock|\.pb)\.go:") | fail_on_output
golint ./... 2>&1 | (! grep -vE "(_mock|\.pb)\.go:")
go vet -all .

# - Check that generated proto files are up to date.
if [[ -z "${VET_SKIP_PROTO}" ]]; then
  PATH="/home/travis/bin:${PATH}" make proto && \
    git status --porcelain 2>&1 | fail_on_output || \
    (git status; git --no-pager diff; exit 1)
fi

# - Check that our module is tidy.
if go help mod >& /dev/null; then
  go mod tidy && \
    git status --porcelain 2>&1 | fail_on_output || \
    (git status; git --no-pager diff; exit 1)
fi

# - Collection of static analysis checks
# TODO(menghanl): fix errors in transport_test.
staticcheck -go 1.9 -checks 'inherit,-ST1015' -ignore '
google.golang.org/grpc/balancer.go:SA1019
google.golang.org/grpc/balancer/roundrobin/roundrobin_test.go:SA1019
google.golang.org/grpc/balancer/xds/edsbalancer/balancergroup.go:SA1019
google.golang.org/grpc/balancer/xds/xds.go:SA1019
google.golang.org/grpc/balancer_conn_wrappers.go:SA1019
google.golang.org/grpc/balancer_test.go:SA1019
google.golang.org/grpc/benchmark/benchmain/main.go:SA1019
google.golang.org/grpc/benchmark/worker/benchmark_client.go:SA1019
google.golang.org/grpc/clientconn.go:S1024
google.golang.org/grpc/clientconn_state_transition_test.go:SA1019
google.golang.org/grpc/clientconn_test.go:SA1019
google.golang.org/grpc/internal/transport/handler_server.go:SA1019
google.golang.org/grpc/internal/transport/handler_server_test.go:SA1019
google.golang.org/grpc/resolver/dns/dns_resolver.go:SA1019
google.golang.org/grpc/stats/stats_test.go:SA1019
google.golang.org/grpc/test/channelz_test.go:SA1019
google.golang.org/grpc/test/end2end_test.go:SA1019
google.golang.org/grpc/test/healthcheck_test.go:SA1019
' ./...
misspell -error .
