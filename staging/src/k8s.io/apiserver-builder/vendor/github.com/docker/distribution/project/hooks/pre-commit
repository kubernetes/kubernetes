#!/bin/sh

REPO_ROOT=$(git rev-parse --show-toplevel)
RESOLVE_REPO_ROOT_STATUS=$?
if [ "$RESOLVE_REPO_ROOT_STATUS" -ne "0" ]; then
	printf "Unable to resolve repository root. Error:\n%s\n" "$RESOLVE_REPO_ROOT_STATUS" > /dev/stderr
	exit $RESOLVE_REPO_ROOT_STATUS
fi

cd $REPO_ROOT

GOFMT_ERRORS=$(gofmt -s -l . 2>&1)
if [ -n "$GOFMT_ERRORS" ]; then
	printf 'gofmt failed for the following files:\n%s\n\nPlease run "gofmt -s -l ." in the root of your repository before committing\n' "$GOFMT_ERRORS" > /dev/stderr
	exit 1
fi

GOLINT_ERRORS=$(golint ./... 2>&1)
if [ -n "$GOLINT_ERRORS" ]; then
	printf "golint failed with the following errors:\n%s\n" "$GOLINT_ERRORS" > /dev/stderr
	exit 1
fi

GOVET_ERRORS=$(go vet ./... 2>&1)
GOVET_STATUS=$?
if [ "$GOVET_STATUS" -ne "0" ]; then
	printf "govet failed with the following errors:\n%s\n" "$GOVET_ERRORS" > /dev/stderr
	exit $GOVET_STATUS
fi
