// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2014 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
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
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
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

package proto_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/gogo/protobuf/proto"
	. "github.com/gogo/protobuf/proto/test_proto"
)

func TestUnmarshalMessageSetWithDuplicate(t *testing.T) {
	/*
		Message{
			Tag{1, StartGroup},
			Message{
				Tag{2, Varint}, Uvarint(12345),
				Tag{3, Bytes}, Bytes("hoo"),
			},
			Tag{1, EndGroup},
			Tag{1, StartGroup},
			Message{
				Tag{2, Varint}, Uvarint(12345),
				Tag{3, Bytes}, Bytes("hah"),
			},
			Tag{1, EndGroup},
		}
	*/
	var in []byte
	fmt.Sscanf("0b10b9601a03686f6f0c0b10b9601a036861680c", "%x", &in)

	/*
		Message{
			Tag{1, StartGroup},
			Message{
				Tag{2, Varint}, Uvarint(12345),
				Tag{3, Bytes}, Bytes("hoohah"),
			},
			Tag{1, EndGroup},
		}
	*/
	var want []byte
	fmt.Sscanf("0b10b9601a06686f6f6861680c", "%x", &want)

	var m MyMessageSet
	if err := proto.Unmarshal(in, &m); err != nil {
		t.Fatalf("unexpected Unmarshal error: %v", err)
	}
	got, err := proto.Marshal(&m)
	if err != nil {
		t.Fatalf("unexpected Marshal error: %v", err)
	}

	if !bytes.Equal(got, want) {
		t.Errorf("output mismatch:\ngot  %x\nwant %x", got, want)
	}
}
