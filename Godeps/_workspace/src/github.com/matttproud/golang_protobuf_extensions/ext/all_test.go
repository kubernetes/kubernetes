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

package ext

import (
	"bytes"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	. "code.google.com/p/goprotobuf/proto"
	. "code.google.com/p/goprotobuf/proto/testdata"
)

func TestWriteDelimited(t *testing.T) {
	for _, test := range []struct {
		msg Message
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
				StringField: String(`This is my gigantic, unhappy string.  It exceeds
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
	for _, test := range []struct {
		buf []byte
		msg Message
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
				StringField: String(`This is my gigantic, unhappy string.  It exceeds
the encoding size of a single byte varint.  We are using it to fuzz test the
correctness of the header decoding mechanisms, which may prove problematic.
I expect it may.  Let's hope you enjoy testing as much as we do.`),
			},
			n: 271,
		},
	} {
		msg := Clone(test.msg)
		msg.Reset()
		if n, err := ReadDelimited(bytes.NewBuffer(test.buf), msg); n != test.n || err != test.err {
			t.Fatalf("ReadDelimited(%v, msg) = %v, %v; want %v, %v", test.buf, n, err, test.n, test.err)
		}
		if !Equal(msg, test.msg) {
			t.Fatalf("ReadDelimited(%v, msg); msg = %v; want %v", test.buf, msg, test.msg)
		}
	}
}

func TestEndToEndValid(t *testing.T) {
	for _, test := range [][]Message{
		[]Message{&Empty{}},
		[]Message{&GoEnum{Foo: FOO_FOO1.Enum()}, &Empty{}, &GoEnum{Foo: FOO_FOO1.Enum()}},
		[]Message{&GoEnum{Foo: FOO_FOO1.Enum()}},
		[]Message{&Strings{
			StringField: String(`This is my gigantic, unhappy string.  It exceeds
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
			out := Clone(msg)
			out.Reset()
			n, _ := ReadDelimited(&buf, out)
			// Decide to do EOF checking?
			read += n
			if !Equal(out, msg) {
				t.Fatalf("out = %v; want %v[%d] = %#v", out, test, i, msg)
			}
		}
		if read != written {
			t.Fatalf("%v read = %d; want %d", test, read, written)
		}
	}
}

// visitMessage empties the private state fields of the quick.Value()-generated
// Protocol Buffer messages, for they cause an inordinate amount of problems.
// This is because we are using an automated fuzz generator on a type with
// private fields.
func visitMessage(m Message) {
	t := reflect.TypeOf(m)
	if t.Kind() != reflect.Ptr {
		return
	}
	derefed := t.Elem()
	if derefed.Kind() != reflect.Struct {
		return
	}
	v := reflect.ValueOf(m)
	elem := v.Elem()
	for i := 0; i < elem.NumField(); i++ {
		field := elem.FieldByIndex([]int{i})
		fieldType := field.Type()
		if fieldType.Implements(reflect.TypeOf((*Message)(nil)).Elem()) {
			visitMessage(field.Interface().(Message))
		}
		if field.Kind() == reflect.Slice {
			for i := 0; i < field.Len(); i++ {
				elem := field.Index(i)
				elemType := elem.Type()
				if elemType.Implements(reflect.TypeOf((*Message)(nil)).Elem()) {
					visitMessage(elem.Interface().(Message))
				}
			}
		}
	}
	if field := elem.FieldByName("XXX_unrecognized"); field.IsValid() {
		field.Set(reflect.ValueOf([]byte{}))
	}
	if field := elem.FieldByName("XXX_extensions"); field.IsValid() {
		field.Set(reflect.ValueOf(nil))
	}
}

// rndMessage generates a random valid Protocol Buffer message.
func rndMessage(r *rand.Rand) Message {
	var t reflect.Type
	switch v := rand.Intn(23); v {
	// TODO(br): Uncomment the elements below once fix is incorporated, except
	//           for the elements marked as patently incompatible.
	// case 0:
	// 	t = reflect.TypeOf(&GoEnum{})
	// 	break
	// case 1:
	// 	t = reflect.TypeOf(&GoTestField{})
	// 	break
	case 2:
		t = reflect.TypeOf(&GoTest{})
		break
		// case 3:
		// 	t = reflect.TypeOf(&GoSkipTest{})
		// 	break
		// case 4:
		// 	t = reflect.TypeOf(&NonPackedTest{})
		// 	break
		// case 5:
		// t = reflect.TypeOf(&PackedTest{})
		//	break
	case 6:
		t = reflect.TypeOf(&MaxTag{})
		break
	case 7:
		t = reflect.TypeOf(&OldMessage{})
		break
	case 8:
		t = reflect.TypeOf(&NewMessage{})
		break
	case 9:
		t = reflect.TypeOf(&InnerMessage{})
		break
	case 10:
		t = reflect.TypeOf(&OtherMessage{})
		break
	case 11:
		// PATENTLY INVALID FOR FUZZ GENERATION
		// t = reflect.TypeOf(&MyMessage{})
		break
		// case 12:
		// 	t = reflect.TypeOf(&Ext{})
		// 	break
	case 13:
		// PATENTLY INVALID FOR FUZZ GENERATION
		// t = reflect.TypeOf(&MyMessageSet{})
		break
		// case 14:
		//   t = reflect.TypeOf(&Empty{})
		//   break
		// case 15:
		// t = reflect.TypeOf(&MessageList{})
		// break
		// case 16:
		// 	t = reflect.TypeOf(&Strings{})
		// 	break
		// case 17:
		// 	t = reflect.TypeOf(&Defaults{})
		// 	break
		// case 17:
		// 	t = reflect.TypeOf(&SubDefaults{})
		// 	break
		// case 18:
		// 	t = reflect.TypeOf(&RepeatedEnum{})
		// 	break
	case 19:
		t = reflect.TypeOf(&MoreRepeated{})
		break
		// case 20:
		// 	t = reflect.TypeOf(&GroupOld{})
		// 	break
		// case 21:
		// 	t = reflect.TypeOf(&GroupNew{})
		// 	break
	case 22:
		t = reflect.TypeOf(&FloatingPoint{})
		break
	default:
		// TODO(br): Replace with an unreachable once fixed.
		t = reflect.TypeOf(&GoTest{})
		break
	}
	if t == nil {
		t = reflect.TypeOf(&GoTest{})
	}
	v, ok := quick.Value(t, r)
	if !ok {
		panic("attempt to generate illegal item; consult item 11")
	}
	visitMessage(v.Interface().(Message))
	return v.Interface().(Message)
}

// rndMessages generates several random Protocol Buffer messages.
func rndMessages(r *rand.Rand) []Message {
	n := r.Intn(128)
	out := make([]Message, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, rndMessage(r))
	}
	return out
}

func TestFuzz(t *testing.T) {
	rnd := rand.New(rand.NewSource(42))
	check := func() bool {
		messages := rndMessages(rnd)
		var buf bytes.Buffer
		var written int
		for i, msg := range messages {
			n, err := WriteDelimited(&buf, msg)
			if err != nil {
				t.Fatalf("WriteDelimited(buf, %v[%d]) = ?, %v; wanted ?, nil", messages, i, err)
			}
			written += n
		}
		var read int
		for i, msg := range messages {
			out := Clone(msg)
			out.Reset()
			n, _ := ReadDelimited(&buf, out)
			read += n
			if !Equal(out, msg) {
				t.Fatalf("out = %v; want %v[%d] = %#v", out, messages, i, msg)
			}
		}
		if read != written {
			t.Fatalf("%v read = %d; want %d", messages, read, written)
		}
		return true
	}
	if err := quick.Check(check, nil); err != nil {
		t.Fatal(err)
	}
}
