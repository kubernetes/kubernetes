// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
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

package tags

import (
	"reflect"
	"testing"
)

func TestTags(t *testing.T) {

	var tests = []struct {
		value   interface{}
		field   string
		jsontag string
		xmltag  string
	}{
		{
			value:   Inside{},
			field:   "Field1",
			jsontag: "MyField1",
			xmltag:  ",chardata",
		},
		{
			value:   Outside{},
			field:   "Field2",
			jsontag: "MyField2",
			xmltag:  ",comment",
		},
		{
			value:   Outside_Field3{},
			field:   "Field3",
			jsontag: "MyField3",
			xmltag:  ",comment",
		},
	}

	for _, tt := range tests {
		tv := reflect.ValueOf(tt.value).Type()
		f, _ := tv.FieldByName(tt.field)
		if jsontag := f.Tag.Get("json"); jsontag != tt.jsontag {
			t.Fatalf("proto %q type: json tag %s != %s", tv.Name(), jsontag, tt.jsontag)
		}
		if xmltag := f.Tag.Get("xml"); xmltag != tt.xmltag {
			t.Fatalf("proto %q type: xml tag %s != %s", tv.Name(), xmltag, tt.xmltag)
		}
	}
}
