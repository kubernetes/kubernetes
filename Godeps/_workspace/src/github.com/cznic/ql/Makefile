# Copyright (c) 2014 The ql Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY: all clean nuke

all: editor scanner.go parser.go
	go build
	go vet || true
	golint
	go install ./...
	make todo

bench: all
	go test -run NONE -bench .

check: ql.y
	goyacc -v /dev/null -o /dev/null $<

clean:
	go clean
	rm -f *~ y.go y.tab.c *.out ql.test

coerce.go: helper.go
	if [ -f coerce.go ] ; then rm coerce.go ; fi
	go run helper.go | gofmt > $@

cover:
	t=$(shell tempfile) ; go test -coverprofile $$t && go tool cover -html $$t && unlink $$t

cpu: ql.test
	go test -c
	./$< -test.bench . -test.cpuprofile cpu.out
	go tool pprof --lines $< cpu.out

editor: check scanner.go parser.go coerce.go
	go fmt
	go test -i
	go test
	go install

internalError:
	egrep -ho '"internal error.*"' *.go | sort | cat -n

mem: ql.test
	go test -c
	./$< -test.bench . -test.memprofile mem.out
	go tool pprof --lines --web --alloc_space $< mem.out

nuke:
	go clean -i

parser.go: parser.y
	goyacc -o $@ $<
	sed -i -e 's|//line.*||' -e 's/yyEofCode/yyEOFCode/' $@

ql.test: all

ql.y: doc.go
	sed -n '1,/^package/ s/^\/\/  //p' < $< \
		| ebnf2y -o $@ -oe $*.ebnf -start StatementList -pkg $* -p _

scanner.go: scanner.l parser.go
	golex -o $@ $<

todo:
	@grep -n ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alpha:]][[:alnum:]]* *.go *.l parser.y || true
	@grep -n TODO *.go *.l parser.y testdata.ql || true
	@grep -n BUG *.go *.l parser.y || true
	@grep -n println *.go *.l parser.y || true

later:
	@grep -n LATER *.go *.l parser.y || true
	@grep -n MAYBE *.go *.l parser.y || true
