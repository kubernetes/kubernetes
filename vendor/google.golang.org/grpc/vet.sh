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
    # Install the pinned versions as defined in module tools.
    pushd ./test/tools
    go install \
      golang.org/x/lint/golint \
      golang.org/x/tools/cmd/goimports \
      honnef.co/go/tools/cmd/staticcheck \
      github.com/client9/misspell/cmd/misspell \
      github.com/golang/protobuf/protoc-gen-go
    popd
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
(! git grep -L "\(Copyright [0-9]\{4,\} gRPC authors\)\|DO NOT EDIT" -- '*.go')

# - Make sure all tests in grpc and grpc/test use leakcheck via Teardown.
(! grep 'func Test[^(]' *_test.go)
(! grep 'func Test[^(]' test/*.go)

# - Do not import x/net/context.
(! git grep -l 'x/net/context' -- "*.go")

# - Do not import math/rand for real library code.  Use internal/grpcrand for
#   thread safety.
git grep -l '"math/rand"' -- "*.go" 2>&1 | (! grep -v '^examples\|^stress\|grpcrand\|^benchmark\|wrr_test')

# - Ensure all ptypes proto packages are renamed when importing.
(! git grep "\(import \|^\s*\)\"github.com/golang/protobuf/ptypes/" -- "*.go")

# - Check imports that are illegal in appengine (until Go 1.11).
# TODO: Remove when we drop Go 1.10 support
go list -f {{.Dir}} ./... | xargs go run test/go_vet/vet.go

# - gofmt, goimports, golint (with exceptions for generated code), go vet.
gofmt -s -d -l . 2>&1 | fail_on_output
goimports -l . 2>&1 | (! grep -vE "(_mock|\.pb)\.go")
golint ./... 2>&1 | (! grep -vE "(_mock|\.pb)\.go:")
go vet -all .

misspell -error .

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
#
# TODO(dfawley): don't use deprecated functions in examples or first-party
# plugins.
SC_OUT="$(mktemp)"
staticcheck -go 1.9 -checks 'inherit,-ST1015' ./... > "${SC_OUT}" || true
# Error if anything other than deprecation warnings are printed.
(! grep -v "is deprecated:.*SA1019" "${SC_OUT}")
# Only ignore the following deprecated types/fields/functions.
(! grep -Fv '.HandleResolvedAddrs
.HandleSubConnStateChange
.HeaderMap
.NewAddress
.NewServiceConfig
.Metadata is deprecated: use Attributes
.Type is deprecated: use Attributes
.UpdateBalancerState
balancer.Picker
grpc.CallCustomCodec
grpc.Code
grpc.Compressor
grpc.Decompressor
grpc.MaxMsgSize
grpc.MethodConfig
grpc.NewGZIPCompressor
grpc.NewGZIPDecompressor
grpc.RPCCompressor
grpc.RPCDecompressor
grpc.RoundRobin
grpc.ServiceConfig
grpc.WithBalancer
grpc.WithBalancerName
grpc.WithCompressor
grpc.WithDecompressor
grpc.WithDialer
grpc.WithMaxMsgSize
grpc.WithServiceConfig
grpc.WithTimeout
http.CloseNotifier
naming.Resolver
naming.Update
naming.Watcher
resolver.Backend
resolver.GRPCLB' "${SC_OUT}"
)
