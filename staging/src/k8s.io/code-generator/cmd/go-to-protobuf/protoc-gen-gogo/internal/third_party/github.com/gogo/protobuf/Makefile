# Protocol Buffers for Go with Gadgets
#
# Copyright (c) 2013, The GoGo Authors. All rights reserved.
# http://github.com/gogo/protobuf
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

GO_VERSION:=$(shell go version)
BENCHLIST?=all

# Skip known issues from purego tests
# https://github.com/gogo/protobuf/issues/447
# https://github.com/gogo/protobuf/issues/448
SKIPISSUE:="/jsonpb|/test/casttype/|/test/oneof/combos/"

.PHONY: nuke regenerate tests clean install gofmt vet contributors

all: clean install regenerate install tests errcheck vet

buildserverall: clean install regenerate install tests vet js purego

install:
	go install ./proto
	go install ./gogoproto
	go install ./jsonpb
	go install ./protoc-gen-gogo
	go install ./protoc-gen-gofast
	go install ./protoc-gen-gogofast
	go install ./protoc-gen-gogofaster
	go install ./protoc-gen-gogoslick
	go install ./protoc-gen-gostring
	go install ./protoc-min-version
	go install ./protoc-gen-combo
	go install ./gogoreplace

clean:
	go clean ./...

nuke:
	go clean -i -cache ./...

gofmt:
	gofmt -l -s -w .

regenerate:
	make -C protoc-gen-gogo regenerate
	make -C gogoproto regenerate
	make -C proto/test_proto regenerate
	make -C proto/proto3_proto regenerate
	make -C jsonpb/jsonpb_test_proto regenerate
	make -C conformance regenerate
	make -C protobuf regenerate
	make -C test regenerate
	make -C test/example regenerate
	make -C test/unrecognized regenerate
	make -C test/group regenerate
	make -C test/unrecognizedgroup regenerate
	make -C test/enumstringer regenerate
	make -C test/unmarshalmerge regenerate
	make -C test/moredefaults regenerate
	make -C test/issue8 regenerate
	make -C test/enumprefix regenerate
	make -C test/enumcustomname regenerate
	make -C test/packed regenerate
	make -C test/protosize regenerate
	make -C test/tags regenerate
	make -C test/oneof regenerate
	make -C test/oneof3 regenerate
	make -C test/theproto3 regenerate
	make -C test/mapdefaults regenerate
	make -C test/mapsproto2 regenerate
	make -C test/issue42order regenerate
	make -C proto generate-test-pbs
	make -C test/importdedup regenerate
	make -C test/importduplicate regenerate
	make -C test/custombytesnonstruct regenerate
	make -C test/required regenerate
	make -C test/casttype regenerate
	make -C test/castvalue regenerate
	make -C vanity/test regenerate
	make -C test/sizeunderscore regenerate
	make -C test/issue34 regenerate
	make -C test/empty-issue70 regenerate
	make -C test/indeximport-issue72 regenerate
	make -C test/fuzztests regenerate
	make -C test/oneofembed regenerate
	make -C test/asymetric-issue125 regenerate
	make -C test/filedotname regenerate
	make -C test/nopackage regenerate
	make -C test/types regenerate
	make -C test/proto3extension regenerate
	make -C test/stdtypes regenerate
	make -C test/data regenerate
	make -C test/typedecl regenerate
	make -C test/issue260 regenerate
	make -C test/issue261 regenerate
	make -C test/issue262 regenerate
	make -C test/issue312 regenerate
	make -C test/enumdecl regenerate
	make -C test/typedecl_all regenerate
	make -C test/enumdecl_all regenerate
	make -C test/int64support regenerate
	make -C test/issue322 regenerate
	make -C test/issue330 regenerate
	make -C test/importcustom-issue389 regenerate
	make -C test/merge regenerate
	make -C test/cachedsize regenerate
	make -C test/deterministic regenerate
	make -C test/issue438 regenerate
	make -C test/issue444 regenerate
	make -C test/issue449 regenerate
	make -C test/xxxfields regenerate
	make -C test/issue435 regenerate
	make -C test/issue411 regenerate
	make -C test/issue498 regenerate
	make -C test/issue503 regenerate
	make -C test/issue530 regenerate
	make -C test/issue617 regenerate
	make -C test/issue620 regenerate
	make -C test/protobuffer regenerate
	make -C test/issue630 regenerate

	make gofmt

tests:
	go build ./test/enumprefix
	go test ./...
	(cd test/stdtypes && make test)

vet:
	go get golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow
	go vet ./...
	go vet -vettool=$(shell which shadow) ./...

errcheck:
	go get github.com/kisielk/errcheck
	errcheck ./test/...

drone:
	sudo apt-get install protobuf-compiler
	(cd $(GOPATH)/src/github.com/gogo/protobuf && make buildserverall)

testall:
	go get -u github.com/golang/protobuf/proto
	make -C protoc-gen-gogo test
	make -C vanity/test test
	make -C test/registration test
	make -C conformance test
	make -C test/issue427 test
	make tests

bench:
	go get golang.org/x/tools/cmd/benchcmp
	(cd test/mixbench && go build .)
	./test/mixbench/mixbench -benchlist "${BENCHLIST}"

contributors:
	git log --format='%aN <%aE>' | sort -fu > CONTRIBUTORS

js:
ifeq (go1.12, $(findstring go1.12, $(GO_VERSION)))
	go get -u github.com/gopherjs/gopherjs
	gopherjs build github.com/gogo/protobuf/protoc-gen-gogo
endif

purego:
	go test -tags purego $$(go list ./... | grep -Ev $(SKIPISSUE))

update:
	(cd protobuf && make update)
