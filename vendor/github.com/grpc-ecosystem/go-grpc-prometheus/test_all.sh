#!/usr/bin/env bash

set -e
echo "" > coverage.txt

for d in $(go list ./... | grep -v vendor); do
    echo -e "TESTS FOR: for \033[0;35m${d}\033[0m"
    go test -race -v -coverprofile=profile.coverage.out -covermode=atomic $d
    if [ -f profile.coverage.out ]; then
        cat profile.coverage.out >> coverage.txt
        rm profile.coverage.out
    fi
    echo ""
done
