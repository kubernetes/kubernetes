#!/bin/bash

set -ex  # Exit on error; debugging enabled.
set -o pipefail  # Fail a pipe if any sub-command fails.

# not makes sure the command passed to it does not exit with a return code of 0.
not() {
  # This is required instead of the earlier (! $COMMAND) because subshells and
  # pipefail don't work the same on Darwin as in Linux.
  ! "$@"
}

die() {
  echo "$@" >&2
  exit 1
}

fail_on_output() {
  tee /dev/stderr | not read
}

# Check to make sure it's safe to modify the user's git repo.
git status --porcelain | fail_on_output

# Undo any edits made by this script.
cleanup() {
  git reset --hard HEAD
}
trap cleanup EXIT

PATH="${HOME}/go/bin:${GOROOT}/bin:${PATH}"
go version

if [[ "$1" = "-install" ]]; then
  # Install the pinned versions as defined in module tools.
  pushd ./test/tools
  go install \
    golang.org/x/tools/cmd/goimports \
    honnef.co/go/tools/cmd/staticcheck \
    github.com/client9/misspell/cmd/misspell
  popd
  if [[ -z "${VET_SKIP_PROTO}" ]]; then
    if [[ "${GITHUB_ACTIONS}" = "true" ]]; then
      PROTOBUF_VERSION=22.0 # a.k.a v4.22.0 in pb.go files.
      PROTOC_FILENAME=protoc-${PROTOBUF_VERSION}-linux-x86_64.zip
      pushd /home/runner/go
      wget https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/${PROTOC_FILENAME}
      unzip ${PROTOC_FILENAME}
      bin/protoc --version
      popd
    elif not which protoc > /dev/null; then
      die "Please install protoc into your path"
    fi
  fi
  exit 0
elif [[ "$#" -ne 0 ]]; then
  die "Unknown argument(s): $*"
fi

# - Check that generated proto files are up to date.
if [[ -z "${VET_SKIP_PROTO}" ]]; then
  make proto && git status --porcelain 2>&1 | fail_on_output || \
    (git status; git --no-pager diff; exit 1)
fi

if [[ -n "${VET_ONLY_PROTO}" ]]; then
  exit 0
fi

# - Ensure all source files contain a copyright message.
# (Done in two parts because Darwin "git grep" has broken support for compound
# exclusion matches.)
(grep -L "DO NOT EDIT" $(git grep -L "\(Copyright [0-9]\{4,\} gRPC authors\)" -- '*.go') || true) | fail_on_output

# - Make sure all tests in grpc and grpc/test use leakcheck via Teardown.
not grep 'func Test[^(]' *_test.go
not grep 'func Test[^(]' test/*.go

# - Check for typos in test function names
git grep 'func (s) ' -- "*_test.go" | not grep -v 'func (s) Test'
git grep 'func [A-Z]' -- "*_test.go" | not grep -v 'func Test\|Benchmark\|Example'

# - Do not import x/net/context.
not git grep -l 'x/net/context' -- "*.go"

# - Do not import math/rand for real library code.  Use internal/grpcrand for
#   thread safety.
git grep -l '"math/rand"' -- "*.go" 2>&1 | not grep -v '^examples\|^interop/stress\|grpcrand\|^benchmark\|wrr_test'

# - Do not use "interface{}"; use "any" instead.
git grep -l 'interface{}' -- "*.go" 2>&1 | not grep -v '\.pb\.go\|protoc-gen-go-grpc\|grpc_testing_not_regenerate'

# - Do not call grpclog directly. Use grpclog.Component instead.
git grep -l -e 'grpclog.I' --or -e 'grpclog.W' --or -e 'grpclog.E' --or -e 'grpclog.F' --or -e 'grpclog.V' -- "*.go" | not grep -v '^grpclog/component.go\|^internal/grpctest/tlogger_test.go'

# - Ensure all ptypes proto packages are renamed when importing.
not git grep "\(import \|^\s*\)\"github.com/golang/protobuf/ptypes/" -- "*.go"

# - Ensure all usages of grpc_testing package are renamed when importing.
not git grep "\(import \|^\s*\)\"google.golang.org/grpc/interop/grpc_testing" -- "*.go"

# - Ensure all xds proto imports are renamed to *pb or *grpc.
git grep '"github.com/envoyproxy/go-control-plane/envoy' -- '*.go' ':(exclude)*.pb.go' | not grep -v 'pb "\|grpc "'

misspell -error .

# - gofmt, goimports, go vet, go mod tidy.
# Perform these checks on each module inside gRPC.
for MOD_FILE in $(find . -name 'go.mod'); do
  MOD_DIR=$(dirname ${MOD_FILE})
  pushd ${MOD_DIR}
  go vet -all ./... | fail_on_output
  gofmt -s -d -l . 2>&1 | fail_on_output
  goimports -l . 2>&1 | not grep -vE "\.pb\.go"

  go mod tidy -compat=1.19
  git status --porcelain 2>&1 | fail_on_output || \
    (git status; git --no-pager diff; exit 1)
  popd
done

# - Collection of static analysis checks
SC_OUT="$(mktemp)"
staticcheck -go 1.19 -checks 'all' ./... > "${SC_OUT}" || true

# Error for anything other than checks that need exclusions.
grep -v "(ST1000)" "${SC_OUT}" | grep -v "(SA1019)" | grep -v "(ST1003)" | not grep -v "(ST1019)\|\(other import of\)"

# Exclude underscore checks for generated code.
grep "(ST1003)" "${SC_OUT}" | not grep -v '\(.pb.go:\)\|\(code_string_test.go:\)\|\(grpc_testing_not_regenerate\)'

# Error for duplicate imports not including grpc protos.
grep "(ST1019)\|\(other import of\)" "${SC_OUT}" | not grep -Fv 'XXXXX PleaseIgnoreUnused
channelz/grpc_channelz_v1"
go-control-plane/envoy
grpclb/grpc_lb_v1"
health/grpc_health_v1"
interop/grpc_testing"
orca/v3"
proto/grpc_gcp"
proto/grpc_lookup_v1"
reflection/grpc_reflection_v1"
reflection/grpc_reflection_v1alpha"
XXXXX PleaseIgnoreUnused'

# Error for any package comments not in generated code.
grep "(ST1000)" "${SC_OUT}" | not grep -v "\.pb\.go:"

# Only ignore the following deprecated types/fields/functions and exclude
# generated code.
grep "(SA1019)" "${SC_OUT}" | not grep -Fv 'XXXXX PleaseIgnoreUnused
XXXXX Protobuf related deprecation errors:
"github.com/golang/protobuf
.pb.go:
grpc_testing_not_regenerate
: ptypes.
proto.RegisterType
XXXXX gRPC internal usage deprecation errors:
"google.golang.org/grpc
: grpc.
: v1alpha.
: v1alphareflectionpb.
BalancerAttributes is deprecated:
CredsBundle is deprecated:
Metadata is deprecated: use Attributes instead.
NewSubConn is deprecated:
OverrideServerName is deprecated:
RemoveSubConn is deprecated:
SecurityVersion is deprecated:
Target is deprecated: Use the Target field in the BuildOptions instead.
UpdateAddresses is deprecated:
UpdateSubConnState is deprecated:
balancer.ErrTransientFailure is deprecated:
grpc/reflection/v1alpha/reflection.proto
XXXXX xDS deprecated fields we support
.ExactMatch
.PrefixMatch
.SafeRegexMatch
.SuffixMatch
GetContainsMatch
GetExactMatch
GetMatchSubjectAltNames
GetPrefixMatch
GetSafeRegexMatch
GetSuffixMatch
GetTlsCertificateCertificateProviderInstance
GetValidationContextCertificateProviderInstance
XXXXX PleaseIgnoreUnused'

echo SUCCESS
