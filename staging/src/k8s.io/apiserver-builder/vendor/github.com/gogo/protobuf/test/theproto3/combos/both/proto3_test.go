// Copyright (c) 2015, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
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

package theproto3

import (
	"reflect"
	"testing"

	"github.com/gogo/protobuf/proto"
)

func TestNilMaps(t *testing.T) {
	m := &AllMaps{StringToMsgMap: map[string]*FloatingPoint{"a": nil}}
	if _, err := proto.Marshal(m); err == nil {
		t.Fatalf("expected error")
	}
}

func TestCustomTypeSize(t *testing.T) {
	m := &Uint128Pair{}
	m.Size() // Should not panic.
}

func TestCustomTypeMarshalUnmarshal(t *testing.T) {
	m1 := &Uint128Pair{}
	if b, err := proto.Marshal(m1); err != nil {
		t.Fatal(err)
	} else {
		m2 := &Uint128Pair{}
		if err := proto.Unmarshal(b, m2); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(m1, m2) {
			t.Errorf("expected %+v, got %+v", m1, m2)
		}
	}
}
