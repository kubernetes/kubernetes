#!/usr/bin/env bash
set -eu

go test -p 1 -tags stubpkg ./internal/... \
    > testdata/go-test-quiet.out \
    2> testdata/go-test-quiet.err \
    | true


go test -p 1 -v -tags stubpkg ./internal/... \
    > testdata/go-test-verbose.out \
    2> testdata/go-test-verbose.err \
    | true

go test -p 1 -json -tags stubpkg ./internal/... \
    > testdata/go-test-json.out \
    2> testdata/go-test-json.err \
    | true

go test -p 1 -json -timeout 10ms -tags 'stubpkg timeout' ./internal/... \
    > testdata/go-test-json-with-timeout.out \
    2> testdata/go-test-json-with-timeout.err \
    | true

go test -p 1 -json -tags 'stubpkg panic' ./internal/... \
    > testdata/go-test-json-with-panic.out \
    2> testdata/go-test-json-with-panic.err \
    | true


go test -p 1 -json -tags stubpkg -cover ./internal/... \
    > testdata/go-test-json-with-cover.out \
    2> testdata/go-test-json-with-cover.err \
    | true