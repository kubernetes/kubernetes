#!/bin/bash

set -ex

reference_ref=${1:-master}
reference_git=${2:-.}

if ! `hash benchstat 2>/dev/null`; then
    echo "Installing benchstat"
    go get golang.org/x/perf/cmd/benchstat
fi

tempdir=`mktemp -d /tmp/go-toml-benchmark-XXXXXX`
ref_tempdir="${tempdir}/ref"
ref_benchmark="${ref_tempdir}/benchmark-`echo -n ${reference_ref}|tr -s '/' '-'`.txt"
local_benchmark="`pwd`/benchmark-local.txt"

echo "=== ${reference_ref} (${ref_tempdir})"
git clone ${reference_git} ${ref_tempdir} >/dev/null 2>/dev/null
pushd ${ref_tempdir} >/dev/null
git checkout ${reference_ref} >/dev/null 2>/dev/null
go test -bench=. -benchmem | tee ${ref_benchmark}
cd benchmark
go test -bench=. -benchmem | tee -a ${ref_benchmark}
popd >/dev/null

echo ""
echo "=== local"
go test -bench=. -benchmem  | tee ${local_benchmark}
cd benchmark
go test -bench=. -benchmem | tee -a ${local_benchmark}

echo ""
echo "=== diff"
benchstat -delta-test=none ${ref_benchmark} ${local_benchmark}
