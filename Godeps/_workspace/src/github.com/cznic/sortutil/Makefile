# Copyright 2014 The sortutil Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: all clean editor later nuke todo

grep=--include=*.go

all: editor
	go vet
	golint .
	make todo

clean:
	go clean
	rm -f *~

editor:
	go fmt
	go test -i
	go test
	go build

later:
	@grep -n $(grep) LATER * || true
	@grep -n $(grep) MAYBE * || true

nuke: clean
	go clean -i

todo:
	@grep -nr $(grep) ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alpha:]][[:alnum:]]* * || true
	@grep -nr $(grep) TODO * || true
	@grep -nr $(grep) BUG * || true
	@grep -nr $(grep) println * || true
