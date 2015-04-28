# Copyright 2014 The zappy Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: editor
	go tool vet -printfuncs Log:0,Logf:0 .
	golint .
	go install
	make todo

editor:
	go fmt
	go test -i
	@#go test
	./purego.sh

todo:
	@grep -n ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alnum:]] *.go || true
	@grep -n TODO *.go || true
	@grep -n BUG *.go || true
	@grep -n println *.go || true

clean:
	rm -f *~ cov cov.html

gocov:
	gocov test $(COV) | gocov-html > cov.html

bench:
	go test -run NONE -bench B
