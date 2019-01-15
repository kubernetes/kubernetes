#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

LOGFILE=/tmp/make.$TARGET.log

rm -f $LOGFILE

# We need to redirect logs to files to work around traivs 4MB log limit, see
# https://docs.travis-ci.com/user/common-build-problems/#Log-Length-exceeded.
dump_output() {
    if [ ! -f $LOGFILE ]; then
        return
    fi
    local logfile_bytes=$(du -b $LOGFILE | awk '{print $1}')
    local bytes=$((4 * 1024 * 1024 - 50 * 1024))
    if [[ "$bytes" -ge "$logfile_bytes" ]]; then
        echo ">>> log file total $logfile_bytes bytes, show all contents:"
    else
        echo ">>> log file total $logfile_bytes bytes, show last $bytes bytes:"
    fi
    tail -c $bytes $LOGFILE
}

trap 'dump_output' EXIT

kube::test::find_dirs() {
  (
    cd ${KUBE_ROOT}
    find -L . -not \( \
        \( \
          -path './_artifacts/*' \
          -o -path './bazel-*/*' \
          -o -path './_output/*' \
          -o -path './_gopath/*' \
          -o -path './cmd/kubeadm/*' \
          -o -path './contrib/podex/*' \
          -o -path './output/*' \
          -o -path './release/*' \
          -o -path './target/*' \
          -o -path './test/*' \
          -o -path './cluster/*' \
          -o -path './third_party/*' \
          -o -path './staging/*' \
          -o -path './vendor/*' \
        \) -prune \
      \) -name '*_test.go' -print0 | xargs -0n1 dirname | sed "s|^\./|${KUBE_GO_PACKAGE}/|" | LC_ALL=C sort -u
  )
}

args=(
    $TARGET
)

if [[ "$TARGET" == "test-integration" ]]; then
    ./hack/install-etcd.sh
elif [[ "$TARGET" == "test" ]]; then
    TESTS=${TESTS:-}
    if [ -z "$TESTS" ]; then
        TESTS=($(kube::test::find_dirs))
    fi
    args+=(TESTS="${TESTS[@]}")
fi

if [ -n "${TRAVIS:-}" ]; then
    # Resource in travis is low, slow down and increase timeout.
    # See https://docs.travis-ci.com/user/reference/overview/.
    args+=(PARALLEL=1)
    export KUBE_TIMEOUT=${KUBE_TIMEOUT:-"--timeout 300s"}
    # Don't build for all platforms, because it may exceeed travis 2 hours
    # limit.
    export KUBE_FASTBUILD=true
fi

# pass all KUBE_ environments
while IFS='=' read -r -d '' n v; do
    if [[ $n =~ ^KUBE_ ]]; then
        args+=($n="$v")
    fi
done < <(env -0)

if [[ "$TARGET" == "verify" ]]; then
    go get -u github.com/golang/lint/golint
    ./hack/verify-golint.sh
else
    echo "./build/run.sh make \"${args[@]}\" &> $LOGFILE"
    ./build/run.sh make "${args[@]}" &> $LOGFILE
fi
