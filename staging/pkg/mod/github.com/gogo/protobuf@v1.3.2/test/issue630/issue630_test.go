// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2015, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package issue630

import (
	"testing"
)

func TestRepeatedNonNullableGoString(t *testing.T) {
	foo := &Foo{Bar1: []Bar{{Baz: "a"}, {Baz: "b"}}}
	expected := `&issue630.Foo{Bar1: []issue630.Bar{issue630.Bar{Baz:"a", XXX_NoUnkeyedLiteral:struct {}{}, XXX_unrecognized:[]uint8(nil), XXX_sizecache:0}, issue630.Bar{Baz:"b", XXX_NoUnkeyedLiteral:struct {}{}, XXX_unrecognized:[]uint8(nil), XXX_sizecache:0}},
}`
	actual := foo.GoString()
	if expected != actual {
		t.Fatalf("expected:\n%s\ngot:\n%s\n", expected, actual)
	}
}

func TestRepeatedNullableGoString(t *testing.T) {
	qux := &Qux{Bar1: []*Bar{{Baz: "a"}, {Baz: "b"}}}
	expected := `&issue630.Qux{Bar1: []*issue630.Bar{&issue630.Bar{Baz: "a",
}, &issue630.Bar{Baz: "b",
}},
}`
	actual := qux.GoString()
	if expected != actual {
		t.Fatalf("expected:\n%s\ngot:\n%s\n", expected, actual)
	}
}
