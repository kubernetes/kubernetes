// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2017, The GoGo Authors. All rights reserved.
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

package mapdefaults

import (
	"testing"

	"github.com/gogo/protobuf/proto"
)

func TestUnmarshalIgnoreUnknownField(t *testing.T) {
	fm := &FakeMap{
		Entries: []*FakeMapEntry{
			{
				Key:   "key",
				Value: "value",
				Other: "other",
			},
		},
	}

	serializedMsg, err := proto.Marshal(fm)
	if err != nil {
		t.Fatalf("Failed to serialize msg: %s", err)
	}

	msg := &MapTest{}
	err = proto.Unmarshal(serializedMsg, msg)

	if err != nil {
		var pb proto.Message = msg
		_, ok := pb.(proto.Unmarshaler)
		if !ok {
			// non-codegen implementation returns error when extra tags are
			// present.
			return
		}
		t.Fatalf("Unexpected error: %s", err)
	}

	strStr := msg.StrStr
	if len(strStr) != 1 {
		t.Fatal("StrStr map should have 1 key/value pairs")
	}

	val, ok := strStr["key"]
	if !ok {
		t.Fatal("\"key\" not found in StrStr map.")
	}
	if val != "value" {
		t.Fatalf("Unexpected value for \"value\": %s", val)
	}
}
