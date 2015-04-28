# Copyright (c) 2014 The mathutil Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: all todo clean nuke

grep=--include=*.go --include=*.run --include=*.y

all: editor
	go build
	go vet || true
	golint .
	go install
	make todo

clean:
	go clean

editor:
	go fmt
	go test -i
	go test

todo:
	@grep -nr $(grep) ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alpha:]][[:alnum:]]* * || true
	@grep -nr $(grep) TODO * || true
	@grep -nr $(grep) BUG * || true
	@grep -nr $(grep) println * || true

nuke: clean
	go clean -i
