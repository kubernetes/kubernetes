#!/bin/bash

# for version in v1.9.1 v1.10.0 v1.10.3 v1.10.12 v1.11.2 v1.11.3 v1.12.0 v1.13.1 v1.14.0 v1.14.1 ; do
for version in v1.10.12 v1.14.1 v1.15.2 ; do
    echo "### $version" > $version.txt
    git checkout -- go.mod && git checkout $version && go test -run=NONE -bench=Benchmark2 >> $version.txt || exit 1
done
