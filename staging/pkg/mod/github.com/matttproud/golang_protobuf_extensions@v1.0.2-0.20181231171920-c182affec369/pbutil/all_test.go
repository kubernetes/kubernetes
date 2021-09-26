// Copyright 2013 Matt T. Proud
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pbutil

import (
	"bytes"
	"testing"

	"github.com/golang/protobuf/proto"

	. "github.com/matttproud/golang_protobuf_extensions/testdata"
)

func TestWriteDelimited(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		msg proto.Message
		buf []byte
		n   int
		err error
	}{
		{
			msg: &Empty{},
			n:   1,
			buf: []byte{0},
		},
		{
			msg: &GoEnum{Foo: FOO_FOO1.Enum()},
			n:   3,
			buf: []byte{2, 8, 1},
		},
		{
			msg: &Strings{
				StringField: proto.String(`This is my gigantic, unhappy string.  It exceeds
the encoding size of a single byte varint.  We are using it to fuzz test the
correctness of the header decoding mechanisms, which may prove problematic.
I expect it may.  Let's hope you enjoy testing as much as we do.`),
			},
			n: 271,
			buf: []byte{141, 2, 10, 138, 2, 84, 104, 105, 115, 32, 105, 115, 32, 109,
				121, 32, 103, 105, 103, 97, 110, 116, 105, 99, 44, 32, 117, 110, 104,
				97, 112, 112, 121, 32, 115, 116, 114, 105, 110, 103, 46, 32, 32, 73,
				116, 32, 101, 120, 99, 101, 101, 100, 115, 10, 116, 104, 101, 32, 101,
				110, 99, 111, 100, 105, 110, 103, 32, 115, 105, 122, 101, 32, 111, 102,
				32, 97, 32, 115, 105, 110, 103, 108, 101, 32, 98, 121, 116, 101, 32,
				118, 97, 114, 105, 110, 116, 46, 32, 32, 87, 101, 32, 97, 114, 101, 32,
				117, 115, 105, 110, 103, 32, 105, 116, 32, 116, 111, 32, 102, 117, 122,
				122, 32, 116, 101, 115, 116, 32, 116, 104, 101, 10, 99, 111, 114, 114,
				101, 99, 116, 110, 101, 115, 115, 32, 111, 102, 32, 116, 104, 101, 32,
				104, 101, 97, 100, 101, 114, 32, 100, 101, 99, 111, 100, 105, 110, 103,
				32, 109, 101, 99, 104, 97, 110, 105, 115, 109, 115, 44, 32, 119, 104,
				105, 99, 104, 32, 109, 97, 121, 32, 112, 114, 111, 118, 101, 32, 112,
				114, 111, 98, 108, 101, 109, 97, 116, 105, 99, 46, 10, 73, 32, 101, 120,
				112, 101, 99, 116, 32, 105, 116, 32, 109, 97, 121, 46, 32, 32, 76, 101,
				116, 39, 115, 32, 104, 111, 112, 101, 32, 121, 111, 117, 32, 101, 110,
				106, 111, 121, 32, 116, 101, 115, 116, 105, 110, 103, 32, 97, 115, 32,
				109, 117, 99, 104, 32, 97, 115, 32, 119, 101, 32, 100, 111, 46},
		},
	} {
		var buf bytes.Buffer
		if n, err := WriteDelimited(&buf, test.msg); n != test.n || err != test.err {
			t.Fatalf("WriteDelimited(buf, %#v) = %v, %v; want %v, %v", test.msg, n, err, test.n, test.err)
		}
		if out := buf.Bytes(); !bytes.Equal(out, test.buf) {
			t.Fatalf("WriteDelimited(buf, %#v); buf = %v; want %v", test.msg, out, test.buf)
		}
	}
}

func TestReadDelimited(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		buf []byte
		msg proto.Message
		n   int
		err error
	}{
		{
			buf: []byte{0},
			msg: &Empty{},
			n:   1,
		},
		{
			n:   3,
			buf: []byte{2, 8, 1},
			msg: &GoEnum{Foo: FOO_FOO1.Enum()},
		},
		{
			buf: []byte{141, 2, 10, 138, 2, 84, 104, 105, 115, 32, 105, 115, 32, 109,
				121, 32, 103, 105, 103, 97, 110, 116, 105, 99, 44, 32, 117, 110, 104,
				97, 112, 112, 121, 32, 115, 116, 114, 105, 110, 103, 46, 32, 32, 73,
				116, 32, 101, 120, 99, 101, 101, 100, 115, 10, 116, 104, 101, 32, 101,
				110, 99, 111, 100, 105, 110, 103, 32, 115, 105, 122, 101, 32, 111, 102,
				32, 97, 32, 115, 105, 110, 103, 108, 101, 32, 98, 121, 116, 101, 32,
				118, 97, 114, 105, 110, 116, 46, 32, 32, 87, 101, 32, 97, 114, 101, 32,
				117, 115, 105, 110, 103, 32, 105, 116, 32, 116, 111, 32, 102, 117, 122,
				122, 32, 116, 101, 115, 116, 32, 116, 104, 101, 10, 99, 111, 114, 114,
				101, 99, 116, 110, 101, 115, 115, 32, 111, 102, 32, 116, 104, 101, 32,
				104, 101, 97, 100, 101, 114, 32, 100, 101, 99, 111, 100, 105, 110, 103,
				32, 109, 101, 99, 104, 97, 110, 105, 115, 109, 115, 44, 32, 119, 104,
				105, 99, 104, 32, 109, 97, 121, 32, 112, 114, 111, 118, 101, 32, 112,
				114, 111, 98, 108, 101, 109, 97, 116, 105, 99, 46, 10, 73, 32, 101, 120,
				112, 101, 99, 116, 32, 105, 116, 32, 109, 97, 121, 46, 32, 32, 76, 101,
				116, 39, 115, 32, 104, 111, 112, 101, 32, 121, 111, 117, 32, 101, 110,
				106, 111, 121, 32, 116, 101, 115, 116, 105, 110, 103, 32, 97, 115, 32,
				109, 117, 99, 104, 32, 97, 115, 32, 119, 101, 32, 100, 111, 46},
			msg: &Strings{
				StringField: proto.String(`This is my gigantic, unhappy string.  It exceeds
the encoding size of a single byte varint.  We are using it to fuzz test the
correctness of the header decoding mechanisms, which may prove problematic.
I expect it may.  Let's hope you enjoy testing as much as we do.`),
			},
			n: 271,
		},
	} {
		msg := proto.Clone(test.msg)
		msg.Reset()
		if n, err := ReadDelimited(bytes.NewBuffer(test.buf), msg); n != test.n || err != test.err {
			t.Fatalf("ReadDelimited(%v, msg) = %v, %v; want %v, %v", test.buf, n, err, test.n, test.err)
		}
		if !proto.Equal(msg, test.msg) {
			t.Fatalf("ReadDelimited(%v, msg); msg = %v; want %v", test.buf, msg, test.msg)
		}
	}
}

func TestEndToEndValid(t *testing.T) {
	t.Parallel()
	for _, test := range [][]proto.Message{
		{&Empty{}},
		{&GoEnum{Foo: FOO_FOO1.Enum()}, &Empty{}, &GoEnum{Foo: FOO_FOO1.Enum()}},
		{&GoEnum{Foo: FOO_FOO1.Enum()}},
		{&Strings{
			StringField: proto.String(`This is my gigantic, unhappy string.  It exceeds
the encoding size of a single byte varint.  We are using it to fuzz test the
correctness of the header decoding mechanisms, which may prove problematic.
I expect it may.  Let's hope you enjoy testing as much as we do.`),
		}},
	} {
		var buf bytes.Buffer
		var written int
		for i, msg := range test {
			n, err := WriteDelimited(&buf, msg)
			if err != nil {
				// Assumption: TestReadDelimited and TestWriteDelimited are sufficient
				//             and inputs for this test are explicitly exercised there.
				t.Fatalf("WriteDelimited(buf, %v[%d]) = ?, %v; wanted ?, nil", test, i, err)
			}
			written += n
		}
		var read int
		for i, msg := range test {
			out := proto.Clone(msg)
			out.Reset()
			n, _ := ReadDelimited(&buf, out)
			// Decide to do EOF checking?
			read += n
			if !proto.Equal(out, msg) {
				t.Fatalf("out = %v; want %v[%d] = %#v", out, test, i, msg)
			}
		}
		if read != written {
			t.Fatalf("%v read = %d; want %d", test, read, written)
		}
	}
}
