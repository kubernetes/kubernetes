#!/bin/bash

PROTOBUF_ROOT=$1
CONFORMANCE_ROOT=$1/conformance
CONFORMANCE_TEST_RUNNER=$CONFORMANCE_ROOT/conformance-test-runner

cd $(dirname $0)

if [[ $PROTOBUF_ROOT == "" ]]; then
  echo "usage: test.sh <protobuf-root>" >/dev/stderr
  exit 1
fi

if [[ ! -x $CONFORMANCE_TEST_RUNNER ]]; then
  echo "SKIP: conformance test runner not installed" >/dev/stderr
  exit 0
fi

a=$CONFORMANCE_ROOT/conformance.proto
b=internal/conformance_proto/conformance.proto
if [[ $(diff $a $b) != "" ]]; then
  cp $a $b
  echo "WARNING: conformance.proto is out of date" >/dev/stderr
fi

$CONFORMANCE_TEST_RUNNER --failure_list failure_list_go.txt ./conformance.sh
