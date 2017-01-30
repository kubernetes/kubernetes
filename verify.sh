#!/usr/bin/env bash

for f in $(find . -name "zz_generated.*.go" | grep -v openapi | grep -v staging); do
    echo $f
    diff $f bazel-genfiles/$(echo $f | sed "s/defaults/defaulter/") 2>&1
done
