# Copyright (c) 2014 The ql Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: all clean nuke

all: editor
	go install
	go vet
	golint .
	make todo

bench: all
	go test -run NONE -bench .

clean:
	go clean
	rm -f *~

cover:
	t=$(shell tempfile) ; go test -coverprofile $$t && go tool cover -html $$t && unlink $$t

editor:
	go fmt
	go test -i
	go test
	go build

nuke:
	go clean -i

todo:
	@grep -n ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alpha:]][[:alnum:]]* *.go || true
	@grep -n TODO *.go || true
	@grep -n BUG *.go || true
	@grep -n println *.go || true

later:
	@grep -n LATER *.go || true
	@grep -n MAYBE *.go || true
