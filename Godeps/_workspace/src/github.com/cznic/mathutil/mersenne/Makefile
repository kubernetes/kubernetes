# Copyright (c) 2014 The mersenne Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: editor
	go build
	go vet
	golint .
	go install
	make todo

editor:
	go fmt
	go test -i
	go test

todo:
	@grep -n ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alnum:]] *.go || true
	@grep -n TODO *.go || true
	@grep -n FIXME *.go || true
	@grep -n BUG *.go || true

clean:
	rm -f *~
