#!/bin/bash
# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd "$(git rev-parse --show-toplevel)"

BOLD="\x1b[1mRunning: "
PASS="\x1b[32mPASS"
FAIL="\x1b[31mFAIL"
RESET="\x1b[0m"

echo -e "${BOLD}go test ./...${RESET}"
RET_TEST0=$(go test ./... | egrep -v "^(ok|[?])\s+")
if [[ ! -z "$RET_TEST0" ]]; then echo "$RET_TEST0"; echo; fi

echo -e "${BOLD}go test -tags purego ./...${RESET}"
RET_TEST1=$(go test -tags purego ./... | egrep -v "^(ok|[?])\s+")
if [[ ! -z "$RET_TEST1" ]]; then echo "$RET_TEST1"; echo; fi

echo -e "${BOLD}go generate${RESET}"
RET_GEN=$(go run ./internal/cmd/generate-alias 2>&1)
if [[ ! -z "$RET_GEN" ]]; then echo "$RET_GEN"; echo; fi

echo -e "${BOLD}go fmt${RESET}"
RET_FMT=$(gofmt -d $(git ls-files *.go) 2>&1)
if [[ ! -z "$RET_FMT" ]]; then echo "$RET_FMT"; echo; fi

echo -e "${BOLD}git diff${RESET}"
RET_DIFF=$(git diff --no-prefix HEAD 2>&1)
if [[ ! -z "$RET_DIFF" ]]; then echo "$RET_DIFF"; echo; fi

echo -e "${BOLD}git ls-files${RESET}"
RET_FILES=$(git ls-files --others --exclude-standard 2>&1)
if [[ ! -z "$RET_FILES" ]]; then echo "$RET_FILES"; echo; fi

if [[ ! -z "$RET_TEST0" ]] || [[ ! -z "$RET_TEST1" ]] || [[ ! -z "$RET_GEN" ]] || [ ! -z "$RET_FMT" ] || [[ ! -z "$RET_DIFF" ]] || [[ ! -z "$RET_FILES" ]]; then
	echo -e "${FAIL}${RESET}"; exit 1
else
	echo -e "${PASS}${RESET}"; exit 0
fi
