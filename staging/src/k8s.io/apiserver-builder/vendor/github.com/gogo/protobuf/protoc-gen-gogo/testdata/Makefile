# Go support for Protocol Buffers - Google's data interchange format
#
# Copyright 2010 The Go Authors.  All rights reserved.
# https://github.com/golang/protobuf
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
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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

all:
	@echo run make test

test:	regenerate testbuild

#test:	regenerate testbuild extension_test
#	./extension_test
#	@echo PASS

my_test/test.pb.go: my_test/test.proto
	protoc --gogo_out=Mmulti/multi1.proto=github.com/gogo/protobuf/protoc-gen-gogo/testdata/multi:. $<

regenerate: my_test/test.pb.go
	go install github.com/gogo/protobuf/protoc-min-version
	protoc-min-version --version="3.0.0" --gogo_out=. ./my_test/test.proto

nuke:	clean

testbuild:	buildprotos
	go test

buildprotos:
	# Invoke protoc once to generate three independent .pb.go files in the same package.
	protoc --gogo_out=. multi/multi{1,2,3}.proto

#extension_test:	extension_test.$O
#	$(LD) -L. -o $@ $<

#multi.a: multi3.pb.$O multi2.pb.$O multi1.pb.$O
#	rm -f multi.a
#	$(QUOTED_GOBIN)/gopack grc $@ $<

#test.pb.go:	imp.pb.go
#multi1.pb.go:	multi2.pb.go multi3.pb.go
#main.$O: imp.pb.$O test.pb.$O multi.a
#extension_test.$O: extension_base.pb.$O extension_extra.pb.$O extension_user.pb.$O
