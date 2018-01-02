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
	"reflect"
	"sort"
	"testing"

	"github.com/golang/protobuf/proto"
	pb "github.com/golang/protobuf/proto/testdata"
	"golang.org/x/sync/errgroup"
)

func TestGetExtensionsWithMissingExtensions(t *testing.T) {
	msg := &pb.MyMessage{}
	ext1 := &pb.Ext{}
	if err := proto.SetExtension(msg, pb.E_Ext_More, ext1); err != nil {
		t.Fatalf("Could not set ext1: %s", err)
	}
	exts, err := proto.GetExtensions(msg, []*proto.ExtensionDesc{
		pb.E_Ext_More,
		pb.E_Ext_Text,
	})
	if err != nil {
		t.Fatalf("GetExtensions() failed: %s", err)
	}
	if exts[0] != ext1 {
		t.Errorf("ext1 not in returned extensions: %T %v", exts[0], exts[0])
	}
	if exts[1] != nil {
		t.Errorf("ext2 in returned extensions: %T %v", exts[1], exts[1])
	}
}

func TestExtensionDescsWithMissingExtensions(t *testing.T) {
	msg := &pb.MyMessage{Count: proto.Int32(0)}
	extdesc1 := pb.E_Ext_More
	if descs, err := proto.ExtensionDescs(msg); len(descs) != 0 || err != nil {
		t.Errorf("proto.ExtensionDescs: got %d descs, error %v; want 0, nil", len(descs), err)
	}

	ext1 := &pb.Ext{}
	if err := proto.SetExtension(msg, extdesc1, ext1); err != nil {
		t.Fatalf("Could not set ext1: %s", err)
	}
	extdesc2 := &proto.ExtensionDesc{
		ExtendedType:  (*pb.MyMessage)(nil),
		ExtensionType: (*bool)(nil),
		Field:         123456789,
		Name:          "a.b",
		Tag:           "varint,123456789,opt",
	}
	ext2 := proto.Bool(false)
	if err := proto.SetExtension(msg, extdesc2, ext2); err != nil {
		t.Fatalf("Could not set ext2: %s", err)
	}

	b, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("Could not marshal msg: %v", err)
	}
	if err := proto.Unmarshal(b, msg); err != nil {
		t.Fatalf("Could not unmarshal into msg: %v", err)
	}

	descs, err := proto.ExtensionDescs(msg)
	if err != nil {
		t.Fatalf("proto.ExtensionDescs: got error %v", err)
	}
	sortExtDescs(descs)
	wantDescs := []*proto.ExtensionDesc{extdesc1, &proto.ExtensionDesc{Field: extdesc2.Field}}
	if !reflect.DeepEqual(descs, wantDescs) {
		t.Errorf("proto.ExtensionDescs(msg) sorted extension ids: got %+v, want %+v", descs, wantDescs)
	}
}

type ExtensionDescSlice []*proto.ExtensionDesc

func (s ExtensionDescSlice) Len() int           { return len(s) }
func (s ExtensionDescSlice) Less(i, j int) bool { return s[i].Field < s[j].Field }
func (s ExtensionDescSlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func sortExtDescs(s []*proto.ExtensionDesc) {
	sort.Sort(ExtensionDescSlice(s))
}

func TestGetExtensionStability(t *testing.T) {
	check := func(m *pb.MyMessage) bool {
		ext1, err := proto.GetExtension(m, pb.E_Ext_More)
		if err != nil {
			t.Fatalf("GetExtension() failed: %s", err)
		}
		ext2, err := proto.GetExtension(m, pb.E_Ext_More)
		if err != nil {
			t.Fatalf("GetExtension() failed: %s", err)
		}
		return ext1 == ext2
	}
	msg := &pb.MyMessage{Count: proto.Int32(4)}
	ext0 := &pb.Ext{}
	if err := proto.SetExtension(msg, pb.E_Ext_More, ext0); err != nil {
		t.Fatalf("Could not set ext1: %s", ext0)
	}
	if !check(msg) {
		t.Errorf("GetExtension() not stable before marshaling")
	}
	bb, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal() failed: %s", err)
	}
	msg1 := &pb.MyMessage{}
	err = proto.Unmarshal(bb, msg1)
	if err != nil {
		t.Fatalf("Unmarshal() failed: %s", err)
	}
	if !check(msg1) {
		t.Errorf("GetExtension() not stable after unmarshaling")
	}
}

func TestGetExtensionDefaults(t *testing.T) {
	var setFloat64 float64 = 1
	var setFloat32 float32 = 2
	var setInt32 int32 = 3
	var setInt64 int64 = 4
	var setUint32 uint32 = 5
	var setUint64 uint64 = 6
	var setBool = true
	var setBool2 = false
	var setString = "Goodnight string"
	var setBytes = []byte("Goodnight bytes")
	var setEnum = pb.DefaultsMessage_TWO

	type testcase struct {
		ext  *proto.ExtensionDesc // Extension we are testing.
		want interface{}          // Expected value of extension, or nil (meaning that GetExtension will fail).
		def  interface{}          // Expected value of extension after ClearExtension().
	}
	tests := []testcase{
		{pb.E_NoDefaultDouble, setFloat64, nil},
		{pb.E_NoDefaultFloat, setFloat32, nil},
		{pb.E_NoDefaultInt32, setInt32, nil},
		{pb.E_NoDefaultInt64, setInt64, nil},
		{pb.E_NoDefaultUint32, setUint32, nil},
		{pb.E_NoDefaultUint64, setUint64, nil},
		{pb.E_NoDefaultSint32, setInt32, nil},
		{pb.E_NoDefaultSint64, setInt64, nil},
		{pb.E_NoDefaultFixed32, setUint32, nil},
		{pb.E_NoDefaultFixed64, setUint64, nil},
		{pb.E_NoDefaultSfixed32, setInt32, nil},
		{pb.E_NoDefaultSfixed64, setInt64, nil},
		{pb.E_NoDefaultBool, setBool, nil},
		{pb.E_NoDefaultBool, setBool2, nil},
		{pb.E_NoDefaultString, setString, nil},
		{pb.E_NoDefaultBytes, setBytes, nil},
		{pb.E_NoDefaultEnum, setEnum, nil},
		{pb.E_DefaultDouble, setFloat64, float64(3.1415)},
		{pb.E_DefaultFloat, setFloat32, float32(3.14)},
		{pb.E_DefaultInt32, setInt32, int32(42)},
		{pb.E_DefaultInt64, setInt64, int64(43)},
		{pb.E_DefaultUint32, setUint32, uint32(44)},
		{pb.E_DefaultUint64, setUint64, uint64(45)},
		{pb.E_DefaultSint32, setInt32, int32(46)},
		{pb.E_DefaultSint64, setInt64, int64(47)},
		{pb.E_DefaultFixed32, setUint32, uint32(48)},
		{pb.E_DefaultFixed64, setUint64, uint64(49)},
		{pb.E_DefaultSfixed32, setInt32, int32(50)},
		{pb.E_DefaultSfixed64, setInt64, int64(51)},
		{pb.E_DefaultBool, setBool, true},
		{pb.E_DefaultBool, setBool2, true},
		{pb.E_DefaultString, setString, "Hello, string"},
		{pb.E_DefaultBytes, setBytes, []byte("Hello, bytes")},
		{pb.E_DefaultEnum, setEnum, pb.DefaultsMessage_ONE},
	}

	checkVal := func(test testcase, msg *pb.DefaultsMessage, valWant interface{}) error {
		val, err := proto.GetExtension(msg, test.ext)
		if err != nil {
			if valWant != nil {
				return fmt.Errorf("GetExtension(): %s", err)
			}
			if want := proto.ErrMissingExtension; err != want {
				return fmt.Errorf("Unexpected error: got %v, want %v", err, want)
			}
			return nil
		}

		// All proto2 extension values are either a pointer to a value or a slice of values.
		ty := reflect.TypeOf(val)
		tyWant := reflect.TypeOf(test.ext.ExtensionType)
		if got, want := ty, tyWant; got != want {
			return fmt.Errorf("unexpected reflect.TypeOf(): got %v want %v", got, want)
		}
		tye := ty.Elem()
		tyeWant := tyWant.Elem()
		if got, want := tye, tyeWant; got != want {
			return fmt.Errorf("unexpected reflect.TypeOf().Elem(): got %v want %v", got, want)
		}

		// Check the name of the type of the value.
		// If it is an enum it will be type int32 with the name of the enum.
		if got, want := tye.Name(), tye.Name(); got != want {
			return fmt.Errorf("unexpected reflect.TypeOf().Elem().Name(): got %v want %v", got, want)
		}

		// Check that value is what we expect.
		// If we have a pointer in val, get the value it points to.
		valExp := val
		if ty.Kind() == reflect.Ptr {
			valExp = reflect.ValueOf(val).Elem().Interface()
		}
		if got, want := valExp, valWant; !reflect.DeepEqual(got, want) {
			return fmt.Errorf("unexpected reflect.DeepEqual(): got %v want %v", got, want)
		}

		return nil
	}

	setTo := func(test testcase) interface{} {
		setTo := reflect.ValueOf(test.want)
		if typ := reflect.TypeOf(test.ext.ExtensionType); typ.Kind() == reflect.Ptr {
			setTo = reflect.New(typ).Elem()
			setTo.Set(reflect.New(setTo.Type().Elem()))
			setTo.Elem().Set(reflect.ValueOf(test.want))
		}
		return setTo.Interface()
	}

	for _, test := range tests {
		msg := &pb.DefaultsMessage{}
		name := test.ext.Name

		// Check the initial value.
		if err := checkVal(test, msg, test.def); err != nil {
			t.Errorf("%s: %v", name, err)
		}

		// Set the per-type value and check value.
		name = fmt.Sprintf("%s (set to %T %v)", name, test.want, test.want)
		if err := proto.SetExtension(msg, test.ext, setTo(test)); err != nil {
			t.Errorf("%s: SetExtension(): %v", name, err)
			continue
		}
		if err := checkVal(test, msg, test.want); err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}

		// Set and check the value.
		name += " (cleared)"
		proto.ClearExtension(msg, test.ext)
		if err := checkVal(test, msg, test.def); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestExtensionsRoundTrip(t *testing.T) {
	msg := &pb.MyMessage{}
	ext1 := &pb.Ext{
		Data: proto.String("hi"),
	}
	ext2 := &pb.Ext{
		Data: proto.String("there"),
	}
	exists := proto.HasExtension(msg, pb.E_Ext_More)
	if exists {
		t.Error("Extension More present unexpectedly")
	}
	if err := proto.SetExtension(msg, pb.E_Ext_More, ext1); err != nil {
		t.Error(err)
	}
	if err := proto.SetExtension(msg, pb.E_Ext_More, ext2); err != nil {
		t.Error(err)
	}
	e, err := proto.GetExtension(msg, pb.E_Ext_More)
	if err != nil {
		t.Error(err)
	}
	x, ok := e.(*pb.Ext)
	if !ok {
		t.Errorf("e has type %T, expected testdata.Ext", e)
	} else if *x.Data != "there" {
		t.Errorf("SetExtension failed to overwrite, got %+v, not 'there'", x)
	}
	proto.ClearExtension(msg, pb.E_Ext_More)
	if _, err = proto.GetExtension(msg, pb.E_Ext_More); err != proto.ErrMissingExtension {
		t.Errorf("got %v, expected ErrMissingExtension", e)
	}
	if _, err := proto.GetExtension(msg, pb.E_X215); err == nil {
		t.Error("expected bad extension error, got nil")
	}
	if err := proto.SetExtension(msg, pb.E_X215, 12); err == nil {
		t.Error("expected extension err")
	}
	if err := proto.SetExtension(msg, pb.E_Ext_More, 12); err == nil {
		t.Error("expected some sort of type mismatch error, got nil")
	}
}

func TestNilExtension(t *testing.T) {
	msg := &pb.MyMessage{
		Count: proto.Int32(1),
	}
	if err := proto.SetExtension(msg, pb.E_Ext_Text, proto.String("hello")); err != nil {
		t.Fatal(err)
	}
	if err := proto.SetExtension(msg, pb.E_Ext_More, (*pb.Ext)(nil)); err == nil {
		t.Error("expected SetExtension to fail due to a nil extension")
	} else if want := "proto: SetExtension called with nil value of type *testdata.Ext"; err.Error() != want {
		t.Errorf("expected error %v, got %v", want, err)
	}
	// Note: if the behavior of Marshal is ever changed to ignore nil extensions, update
	// this test to verify that E_Ext_Text is properly propagated through marshal->unmarshal.
}

func TestMarshalUnmarshalRepeatedExtension(t *testing.T) {
	// Add a repeated extension to the result.
	tests := []struct {
		name string
		ext  []*pb.ComplexExtension
	}{
		{
			"two fields",
			[]*pb.ComplexExtension{
				{First: proto.Int32(7)},
				{Second: proto.Int32(11)},
			},
		},
		{
			"repeated field",
			[]*pb.ComplexExtension{
				{Third: []int32{1000}},
				{Third: []int32{2000}},
			},
		},
		{
			"two fields and repeated field",
			[]*pb.ComplexExtension{
				{Third: []int32{1000}},
				{First: proto.Int32(9)},
				{Second: proto.Int32(21)},
				{Third: []int32{2000}},
			},
		},
	}
	for _, test := range tests {
		// Marshal message with a repeated extension.
		msg1 := new(pb.OtherMessage)
		err := proto.SetExtension(msg1, pb.E_RComplex, test.ext)
		if err != nil {
			t.Fatalf("[%s] Error setting extension: %v", test.name, err)
		}
		b, err := proto.Marshal(msg1)
		if err != nil {
			t.Fatalf("[%s] Error marshaling message: %v", test.name, err)
		}

		// Unmarshal and read the merged proto.
		msg2 := new(pb.OtherMessage)
		err = proto.Unmarshal(b, msg2)
		if err != nil {
			t.Fatalf("[%s] Error unmarshaling message: %v", test.name, err)
		}
		e, err := proto.GetExtension(msg2, pb.E_RComplex)
		if err != nil {
			t.Fatalf("[%s] Error getting extension: %v", test.name, err)
		}
		ext := e.([]*pb.ComplexExtension)
		if ext == nil {
			t.Fatalf("[%s] Invalid extension", test.name)
		}
		if !reflect.DeepEqual(ext, test.ext) {
			t.Errorf("[%s] Wrong value for ComplexExtension: got: %v want: %v\n", test.name, ext, test.ext)
		}
	}
}

func TestUnmarshalRepeatingNonRepeatedExtension(t *testing.T) {
	// We may see multiple instances of the same extension in the wire
	// format. For example, the proto compiler may encode custom options in
	// this way. Here, we verify that we merge the extensions together.
	tests := []struct {
		name string
		ext  []*pb.ComplexExtension
	}{
		{
			"two fields",
			[]*pb.ComplexExtension{
				{First: proto.Int32(7)},
				{Second: proto.Int32(11)},
			},
		},
		{
			"repeated field",
			[]*pb.ComplexExtension{
				{Third: []int32{1000}},
				{Third: []int32{2000}},
			},
		},
		{
			"two fields and repeated field",
			[]*pb.ComplexExtension{
				{Third: []int32{1000}},
				{First: proto.Int32(9)},
				{Second: proto.Int32(21)},
				{Third: []int32{2000}},
			},
		},
	}
	for _, test := range tests {
		var buf bytes.Buffer
		var want pb.ComplexExtension

		// Generate a serialized representation of a repeated extension
		// by catenating bytes together.
		for i, e := range test.ext {
			// Merge to create the wanted proto.
			proto.Merge(&want, e)

			// serialize the message
			msg := new(pb.OtherMessage)
			err := proto.SetExtension(msg, pb.E_Complex, e)
			if err != nil {
				t.Fatalf("[%s] Error setting extension %d: %v", test.name, i, err)
			}
			b, err := proto.Marshal(msg)
			if err != nil {
				t.Fatalf("[%s] Error marshaling message %d: %v", test.name, i, err)
			}
			buf.Write(b)
		}

		// Unmarshal and read the merged proto.
		msg2 := new(pb.OtherMessage)
		err := proto.Unmarshal(buf.Bytes(), msg2)
		if err != nil {
			t.Fatalf("[%s] Error unmarshaling message: %v", test.name, err)
		}
		e, err := proto.GetExtension(msg2, pb.E_Complex)
		if err != nil {
			t.Fatalf("[%s] Error getting extension: %v", test.name, err)
		}
		ext := e.(*pb.ComplexExtension)
		if ext == nil {
			t.Fatalf("[%s] Invalid extension", test.name)
		}
		if !reflect.DeepEqual(*ext, want) {
			t.Errorf("[%s] Wrong value for ComplexExtension: got: %s want: %s\n", test.name, ext, want)
		}
	}
}

func TestClearAllExtensions(t *testing.T) {
	// unregistered extension
	desc := &proto.ExtensionDesc{
		ExtendedType:  (*pb.MyMessage)(nil),
		ExtensionType: (*bool)(nil),
		Field:         101010100,
		Name:          "emptyextension",
		Tag:           "varint,0,opt",
	}
	m := &pb.MyMessage{}
	if proto.HasExtension(m, desc) {
		t.Errorf("proto.HasExtension(%s): got true, want false", proto.MarshalTextString(m))
	}
	if err := proto.SetExtension(m, desc, proto.Bool(true)); err != nil {
		t.Errorf("proto.SetExtension(m, desc, true): got error %q, want nil", err)
	}
	if !proto.HasExtension(m, desc) {
		t.Errorf("proto.HasExtension(%s): got false, want true", proto.MarshalTextString(m))
	}
	proto.ClearAllExtensions(m)
	if proto.HasExtension(m, desc) {
		t.Errorf("proto.HasExtension(%s): got true, want false", proto.MarshalTextString(m))
	}
}

func TestMarshalRace(t *testing.T) {
	// unregistered extension
	desc := &proto.ExtensionDesc{
		ExtendedType:  (*pb.MyMessage)(nil),
		ExtensionType: (*bool)(nil),
		Field:         101010100,
		Name:          "emptyextension",
		Tag:           "varint,0,opt",
	}

	m := &pb.MyMessage{Count: proto.Int32(4)}
	if err := proto.SetExtension(m, desc, proto.Bool(true)); err != nil {
		t.Errorf("proto.SetExtension(m, desc, true): got error %q, want nil", err)
	}

	var g errgroup.Group
	for n := 3; n > 0; n-- {
		g.Go(func() error {
			_, err := proto.Marshal(m)
			return err
		})
	}
	if err := g.Wait(); err != nil {
		t.Fatal(err)
	}
}
