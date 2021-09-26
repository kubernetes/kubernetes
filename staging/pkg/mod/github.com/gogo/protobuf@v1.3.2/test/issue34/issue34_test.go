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

package issue34

import (
	"bytes"
	"github.com/gogo/protobuf/proto"
	"testing"
)

func TestZeroLengthOptionalBytes(t *testing.T) {
	roundtrip := func(f *Foo) *Foo {
		data, err := proto.Marshal(f)
		if err != nil {
			panic(err)
		}
		newF := &Foo{}
		err = proto.Unmarshal(data, newF)
		if err != nil {
			panic(err)
		}
		return newF
	}

	f := &Foo{}
	roundtrippedF := roundtrip(f)
	if roundtrippedF.Bar != nil {
		t.Fatalf("should be nil")
	}

	f.Bar = []byte{}
	roundtrippedF = roundtrip(f)
	if roundtrippedF.Bar == nil {
		t.Fatalf("should not be nil")
	}
	if len(roundtrippedF.Bar) != 0 {
		t.Fatalf("should be empty")
	}
}

func TestRepeatedOptional(t *testing.T) {
	repeated := &FooWithRepeated{Bar: [][]byte{[]byte("a"), []byte("b")}}
	data, err := proto.Marshal(repeated)
	if err != nil {
		panic(err)
	}
	optional := &Foo{}
	err = proto.Unmarshal(data, optional)
	if err != nil {
		panic(err)
	}

	if !bytes.Equal(optional.Bar, []byte("b")) {
		t.Fatalf("should return the last entry")
	}
}
