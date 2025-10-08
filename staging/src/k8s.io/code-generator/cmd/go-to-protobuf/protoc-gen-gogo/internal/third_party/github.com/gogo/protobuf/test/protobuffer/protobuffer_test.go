// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2019, The GoGo Authors. All rights reserved.
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

package protobuffer

import (
	"bytes"
	"testing"

	proto "github.com/gogo/protobuf/proto"
)

func TestProtoBufferMarshal12(t *testing.T) {
	i := int32(33)
	m := &PBuffMarshal{Field1: []byte(string("Tester123")), Field2: &i}
	mer := &PBuffMarshaler{Field1: []byte(string("Tester123")), Field2: &i}
	testCases := []struct {
		name string
		size int
		m    proto.Message
	}{
		{"MarshalLarge", 1024, m},
		{"MarshalSmall", 0, m},
		{"MarshalerLarge", 1024, mer},
		{"MarshalerSmall", 0, mer},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			marshalCheck(t, tc.m, tc.size)
		})
	}
}

func marshalCheck(t *testing.T, m proto.Message, size int) {
	buf := proto.NewBuffer(make([]byte, 0, size))
	err := buf.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	err = buf.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	bufferBytes := buf.Bytes()
	protoBytes, err := proto.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	protoBytes = append(protoBytes, protoBytes...)
	if !bytes.Equal(bufferBytes, protoBytes) {
		t.Fatalf("proto.Buffer Marshal != proto.Marshal (%v != %v)\n", bufferBytes, protoBytes)
	}
}
