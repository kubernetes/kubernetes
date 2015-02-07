// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2014 The Go Authors.  All rights reserved.
// http://code.google.com/p/goprotobuf/
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

package proto

import (
	"bytes"
	"testing"
)

func TestUnmarshalMessageSetWithDuplicate(t *testing.T) {
	// Check that a repeated message set entry will be concatenated.
	in := &MessageSet{
		Item: []*_MessageSet_Item{
			{TypeId: Int32(12345), Message: []byte("hoo")},
			{TypeId: Int32(12345), Message: []byte("hah")},
		},
	}
	b, err := Marshal(in)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	t.Logf("Marshaled bytes: %q", b)

	m := make(map[int32]Extension)
	if err := UnmarshalMessageSet(b, m); err != nil {
		t.Fatalf("UnmarshalMessageSet: %v", err)
	}
	ext, ok := m[12345]
	if !ok {
		t.Fatalf("Didn't retrieve extension 12345; map is %v", m)
	}
	// Skip wire type/field number and length varints.
	got := skipVarint(skipVarint(ext.enc))
	if want := []byte("hoohah"); !bytes.Equal(got, want) {
		t.Errorf("Combined extension is %q, want %q", got, want)
	}
}
