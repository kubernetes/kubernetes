# Copyright 2014 The lldb Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: all editor clean cover nuke

testbin=lldb.test
grep=--include=*.go

all: editor
	go build
	go vet
	golint .
	go install
	make todo

clean:
	go clean
	rm -f *~ cov cov.html bad-dump good-dump lldb.test old.txt new.txt \
		test-acidfiler0-* _test.db _wal

cover:
	t=$(shell tempfile) ; go test -coverprofile $$t && go tool cover -html $$t && unlink $$t

editor:
	go fmt
	go test -i
	go test -timeout 1h
	
mem:
	go test -c
	./$(testbin) -test.bench . -test.memprofile mem.out -test.memprofilerate 1 -test.timeout 24h
	go tool pprof --lines --web --alloc_space $(testbin) mem.out

nuke: clean
	go clean -i

todo:
	@grep -nr $(grep) BUG * || true
	@grep -nr $(grep) LATER * || true
	@grep -nr $(grep) MAYBE * || true
	@grep -nr $(grep) TODO * || true
	@grep -nr $(grep) FIXME * || true
	@grep -nr $(grep) ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alpha:]][[:alnum:]]* * || true
	@grep -nr $(grep) println * || true
