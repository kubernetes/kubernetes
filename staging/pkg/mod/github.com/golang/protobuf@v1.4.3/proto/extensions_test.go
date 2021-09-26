// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/golang/protobuf/proto"

	pb2 "github.com/golang/protobuf/internal/testprotos/proto2_proto"
)

func TestGetExtensionsWithMissingExtensions(t *testing.T) {
	msg := &pb2.MyMessage{}
	ext1 := &pb2.Ext{}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, ext1); err != nil {
		t.Fatalf("Could not set ext1: %s", err)
	}
	exts, err := proto.GetExtensions(msg, []*proto.ExtensionDesc{
		pb2.E_Ext_More,
		pb2.E_Ext_Text,
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

func TestGetExtensionForIncompleteDesc(t *testing.T) {
	msg := &pb2.MyMessage{Count: proto.Int32(0)}
	extdesc1 := &proto.ExtensionDesc{
		ExtendedType:  (*pb2.MyMessage)(nil),
		ExtensionType: (*bool)(nil),
		Field:         123456789,
		Name:          "a.b",
		Tag:           "varint,123456789,opt",
	}
	ext1 := proto.Bool(true)
	if err := proto.SetExtension(msg, extdesc1, ext1); err != nil {
		t.Fatalf("Could not set ext1: %s", err)
	}
	extdesc2 := &proto.ExtensionDesc{
		ExtendedType:  (*pb2.MyMessage)(nil),
		ExtensionType: ([]byte)(nil),
		Field:         123456790,
		Name:          "a.c",
		Tag:           "bytes,123456790,opt",
	}
	ext2 := []byte{0, 1, 2, 3, 4, 5, 6, 7}
	if err := proto.SetExtension(msg, extdesc2, ext2); err != nil {
		t.Fatalf("Could not set ext2: %s", err)
	}
	extdesc3 := &proto.ExtensionDesc{
		ExtendedType:  (*pb2.MyMessage)(nil),
		ExtensionType: (*pb2.Ext)(nil),
		Field:         123456791,
		Name:          "a.d",
		Tag:           "bytes,123456791,opt",
	}
	ext3 := &pb2.Ext{Data: proto.String("foo")}
	if err := proto.SetExtension(msg, extdesc3, ext3); err != nil {
		t.Fatalf("Could not set ext3: %s", err)
	}

	b, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("Could not marshal msg: %v", err)
	}
	if err := proto.Unmarshal(b, msg); err != nil {
		t.Fatalf("Could not unmarshal into msg: %v", err)
	}

	var expected proto.Buffer
	if err := expected.EncodeVarint(uint64((extdesc1.Field << 3) | proto.WireVarint)); err != nil {
		t.Fatalf("failed to compute expected prefix for ext1: %s", err)
	}
	if err := expected.EncodeVarint(1 /* bool true */); err != nil {
		t.Fatalf("failed to compute expected value for ext1: %s", err)
	}

	if b, err := proto.GetExtension(msg, &proto.ExtensionDesc{Field: extdesc1.Field}); err != nil {
		t.Fatalf("Failed to get raw value for ext1: %s", err)
	} else if !reflect.DeepEqual(b, expected.Bytes()) {
		t.Fatalf("Raw value for ext1: got %v, want %v", b, expected.Bytes())
	}

	expected = proto.Buffer{} // reset
	if err := expected.EncodeVarint(uint64((extdesc2.Field << 3) | proto.WireBytes)); err != nil {
		t.Fatalf("failed to compute expected prefix for ext2: %s", err)
	}
	if err := expected.EncodeRawBytes(ext2); err != nil {
		t.Fatalf("failed to compute expected value for ext2: %s", err)
	}

	if b, err := proto.GetExtension(msg, &proto.ExtensionDesc{Field: extdesc2.Field}); err != nil {
		t.Fatalf("Failed to get raw value for ext2: %s", err)
	} else if !reflect.DeepEqual(b, expected.Bytes()) {
		t.Fatalf("Raw value for ext2: got %v, want %v", b, expected.Bytes())
	}

	expected = proto.Buffer{} // reset
	if err := expected.EncodeVarint(uint64((extdesc3.Field << 3) | proto.WireBytes)); err != nil {
		t.Fatalf("failed to compute expected prefix for ext3: %s", err)
	}
	if b, err := proto.Marshal(ext3); err != nil {
		t.Fatalf("failed to compute expected value for ext3: %s", err)
	} else if err := expected.EncodeRawBytes(b); err != nil {
		t.Fatalf("failed to compute expected value for ext3: %s", err)
	}

	if b, err := proto.GetExtension(msg, &proto.ExtensionDesc{Field: extdesc3.Field}); err != nil {
		t.Fatalf("Failed to get raw value for ext3: %s", err)
	} else if !reflect.DeepEqual(b, expected.Bytes()) {
		t.Fatalf("Raw value for ext3: got %v, want %v", b, expected.Bytes())
	}
}

func TestExtensionDescsWithUnregisteredExtensions(t *testing.T) {
	msg := &pb2.MyMessage{Count: proto.Int32(0)}
	extdesc1 := pb2.E_Ext_More
	if descs, err := proto.ExtensionDescs(msg); len(descs) != 0 || err != nil {
		t.Errorf("proto.ExtensionDescs: got %d descs, error %v; want 0, nil", len(descs), err)
	}

	ext1 := &pb2.Ext{}
	if err := proto.SetExtension(msg, extdesc1, ext1); err != nil {
		t.Fatalf("Could not set ext1: %s", err)
	}
	extdesc2 := &proto.ExtensionDesc{
		ExtendedType:  (*pb2.MyMessage)(nil),
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
	wantDescs := []*proto.ExtensionDesc{extdesc1, {Field: extdesc2.Field}}
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
	check := func(m *pb2.MyMessage) bool {
		ext1, err := proto.GetExtension(m, pb2.E_Ext_More)
		if err != nil {
			t.Fatalf("GetExtension() failed: %s", err)
		}
		ext2, err := proto.GetExtension(m, pb2.E_Ext_More)
		if err != nil {
			t.Fatalf("GetExtension() failed: %s", err)
		}
		return ext1 == ext2
	}
	msg := &pb2.MyMessage{Count: proto.Int32(4)}
	ext0 := &pb2.Ext{}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, ext0); err != nil {
		t.Fatalf("Could not set ext1: %s", ext0)
	}
	if !check(msg) {
		t.Errorf("GetExtension() not stable before marshaling")
	}
	bb, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal() failed: %s", err)
	}
	msg1 := &pb2.MyMessage{}
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
	var setEnum = pb2.DefaultsMessage_TWO

	type testcase struct {
		ext  *proto.ExtensionDesc // Extension we are testing.
		want interface{}          // Expected value of extension, or nil (meaning that GetExtension will fail).
		def  interface{}          // Expected value of extension after ClearExtension().
	}
	tests := []testcase{
		{pb2.E_NoDefaultDouble, setFloat64, nil},
		{pb2.E_NoDefaultFloat, setFloat32, nil},
		{pb2.E_NoDefaultInt32, setInt32, nil},
		{pb2.E_NoDefaultInt64, setInt64, nil},
		{pb2.E_NoDefaultUint32, setUint32, nil},
		{pb2.E_NoDefaultUint64, setUint64, nil},
		{pb2.E_NoDefaultSint32, setInt32, nil},
		{pb2.E_NoDefaultSint64, setInt64, nil},
		{pb2.E_NoDefaultFixed32, setUint32, nil},
		{pb2.E_NoDefaultFixed64, setUint64, nil},
		{pb2.E_NoDefaultSfixed32, setInt32, nil},
		{pb2.E_NoDefaultSfixed64, setInt64, nil},
		{pb2.E_NoDefaultBool, setBool, nil},
		{pb2.E_NoDefaultBool, setBool2, nil},
		{pb2.E_NoDefaultString, setString, nil},
		{pb2.E_NoDefaultBytes, setBytes, nil},
		{pb2.E_NoDefaultEnum, setEnum, nil},
		{pb2.E_DefaultDouble, setFloat64, float64(3.1415)},
		{pb2.E_DefaultFloat, setFloat32, float32(3.14)},
		{pb2.E_DefaultInt32, setInt32, int32(42)},
		{pb2.E_DefaultInt64, setInt64, int64(43)},
		{pb2.E_DefaultUint32, setUint32, uint32(44)},
		{pb2.E_DefaultUint64, setUint64, uint64(45)},
		{pb2.E_DefaultSint32, setInt32, int32(46)},
		{pb2.E_DefaultSint64, setInt64, int64(47)},
		{pb2.E_DefaultFixed32, setUint32, uint32(48)},
		{pb2.E_DefaultFixed64, setUint64, uint64(49)},
		{pb2.E_DefaultSfixed32, setInt32, int32(50)},
		{pb2.E_DefaultSfixed64, setInt64, int64(51)},
		{pb2.E_DefaultBool, setBool, true},
		{pb2.E_DefaultBool, setBool2, true},
		{pb2.E_DefaultString, setString, "Hello, string,def=foo"},
		{pb2.E_DefaultBytes, setBytes, []byte("Hello, bytes")},
		{pb2.E_DefaultEnum, setEnum, pb2.DefaultsMessage_ONE},
	}

	checkVal := func(t *testing.T, name string, test testcase, msg *pb2.DefaultsMessage, valWant interface{}) {
		t.Run(name, func(t *testing.T) {
			val, err := proto.GetExtension(msg, test.ext)
			if err != nil {
				if valWant != nil {
					t.Errorf("GetExtension(): %s", err)
					return
				}
				if want := proto.ErrMissingExtension; err != want {
					t.Errorf("Unexpected error: got %v, want %v", err, want)
					return
				}
				return
			}

			// All proto2 extension values are either a pointer to a value or a slice of values.
			ty := reflect.TypeOf(val)
			tyWant := reflect.TypeOf(test.ext.ExtensionType)
			if got, want := ty, tyWant; got != want {
				t.Errorf("unexpected reflect.TypeOf(): got %v want %v", got, want)
				return
			}
			tye := ty.Elem()
			tyeWant := tyWant.Elem()
			if got, want := tye, tyeWant; got != want {
				t.Errorf("unexpected reflect.TypeOf().Elem(): got %v want %v", got, want)
				return
			}

			// Check the name of the type of the value.
			// If it is an enum it will be type int32 with the name of the enum.
			if got, want := tye.Name(), tye.Name(); got != want {
				t.Errorf("unexpected reflect.TypeOf().Elem().Name(): got %v want %v", got, want)
				return
			}

			// Check that value is what we expect.
			// If we have a pointer in val, get the value it points to.
			valExp := val
			if ty.Kind() == reflect.Ptr {
				valExp = reflect.ValueOf(val).Elem().Interface()
			}
			if got, want := valExp, valWant; !reflect.DeepEqual(got, want) {
				t.Errorf("unexpected reflect.DeepEqual(): got %v want %v", got, want)
				return
			}
		})
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
		msg := &pb2.DefaultsMessage{}
		name := test.ext.Name

		// Check the initial value.
		checkVal(t, name+"/initial", test, msg, test.def)

		// Set the per-type value and check value.
		if err := proto.SetExtension(msg, test.ext, setTo(test)); err != nil {
			t.Errorf("%s: SetExtension(): %v", name, err)
			continue
		}
		checkVal(t, name+"/set", test, msg, test.want)

		// Set and check the value.
		proto.ClearExtension(msg, test.ext)
		checkVal(t, name+"/cleared", test, msg, test.def)
	}
}

func TestNilMessage(t *testing.T) {
	name := "nil interface"
	if got, err := proto.GetExtension(nil, pb2.E_Ext_More); err == nil {
		t.Errorf("%s: got %T %v, expected to fail", name, got, got)
	} else if !strings.Contains(err.Error(), "extendable") {
		t.Errorf("%s: got error %v, expected not-extendable error", name, err)
	}

	// Regression tests: all functions of the Extension API
	// used to panic when passed (*M)(nil), where M is a concrete message
	// type.  Now they handle this gracefully as a no-op or reported error.
	var nilMsg *pb2.MyMessage
	desc := pb2.E_Ext_More

	isNotExtendable := func(err error) bool {
		return strings.Contains(fmt.Sprint(err), "not an extendable")
	}

	if proto.HasExtension(nilMsg, desc) {
		t.Error("HasExtension(nil) = true")
	}

	if _, err := proto.GetExtensions(nilMsg, []*proto.ExtensionDesc{desc}); !isNotExtendable(err) {
		t.Errorf("GetExtensions(nil) = %q (wrong error)", err)
	}

	if _, err := proto.ExtensionDescs(nilMsg); !isNotExtendable(err) {
		t.Errorf("ExtensionDescs(nil) = %q (wrong error)", err)
	}

	if err := proto.SetExtension(nilMsg, desc, nil); !isNotExtendable(err) {
		t.Errorf("SetExtension(nil) = %q (wrong error)", err)
	}

	proto.ClearExtension(nilMsg, desc) // no-op
	proto.ClearAllExtensions(nilMsg)   // no-op
}

func TestExtensionsRoundTrip(t *testing.T) {
	msg := &pb2.MyMessage{}
	ext1 := &pb2.Ext{
		Data: proto.String("hi"),
	}
	ext2 := &pb2.Ext{
		Data: proto.String("there"),
	}
	exists := proto.HasExtension(msg, pb2.E_Ext_More)
	if exists {
		t.Error("Extension More present unexpectedly")
	}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, ext1); err != nil {
		t.Error(err)
	}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, ext2); err != nil {
		t.Error(err)
	}
	e, err := proto.GetExtension(msg, pb2.E_Ext_More)
	if err != nil {
		t.Error(err)
	}
	x, ok := e.(*pb2.Ext)
	if !ok {
		t.Errorf("e has type %T, expected test_proto.Ext", e)
	} else if *x.Data != "there" {
		t.Errorf("SetExtension failed to overwrite, got %+v, not 'there'", x)
	}
	proto.ClearExtension(msg, pb2.E_Ext_More)
	if _, err = proto.GetExtension(msg, pb2.E_Ext_More); err != proto.ErrMissingExtension {
		t.Errorf("got %v, expected ErrMissingExtension", e)
	}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, 12); err == nil {
		t.Error("expected some sort of type mismatch error, got nil")
	}
}

func TestNilExtension(t *testing.T) {
	msg := &pb2.MyMessage{
		Count: proto.Int32(1),
	}
	if err := proto.SetExtension(msg, pb2.E_Ext_Text, proto.String("hello")); err != nil {
		t.Fatal(err)
	}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, (*pb2.Ext)(nil)); err == nil {
		t.Error("expected SetExtension to fail due to a nil extension")
	} else if want := fmt.Sprintf("proto: SetExtension called with nil value of type %T", new(pb2.Ext)); err.Error() != want {
		t.Errorf("expected error %v, got %v", want, err)
	}
	// Note: if the behavior of Marshal is ever changed to ignore nil extensions, update
	// this test to verify that E_Ext_Text is properly propagated through marshal->unmarshal.
}

func TestMarshalUnmarshalRepeatedExtension(t *testing.T) {
	// Add a repeated extension to the result.
	tests := []struct {
		name string
		ext  []*pb2.ComplexExtension
	}{
		{
			"two fields",
			[]*pb2.ComplexExtension{
				{First: proto.Int32(7)},
				{Second: proto.Int32(11)},
			},
		},
		{
			"repeated field",
			[]*pb2.ComplexExtension{
				{Third: []int32{1000}},
				{Third: []int32{2000}},
			},
		},
		{
			"two fields and repeated field",
			[]*pb2.ComplexExtension{
				{Third: []int32{1000}},
				{First: proto.Int32(9)},
				{Second: proto.Int32(21)},
				{Third: []int32{2000}},
			},
		},
	}
	for _, test := range tests {
		// Marshal message with a repeated extension.
		msg1 := new(pb2.OtherMessage)
		err := proto.SetExtension(msg1, pb2.E_RComplex, test.ext)
		if err != nil {
			t.Fatalf("[%s] Error setting extension: %v", test.name, err)
		}
		b, err := proto.Marshal(msg1)
		if err != nil {
			t.Fatalf("[%s] Error marshaling message: %v", test.name, err)
		}

		// Unmarshal and read the merged proto.
		msg2 := new(pb2.OtherMessage)
		err = proto.Unmarshal(b, msg2)
		if err != nil {
			t.Fatalf("[%s] Error unmarshaling message: %v", test.name, err)
		}
		e, err := proto.GetExtension(msg2, pb2.E_RComplex)
		if err != nil {
			t.Fatalf("[%s] Error getting extension: %v", test.name, err)
		}
		ext := e.([]*pb2.ComplexExtension)
		if ext == nil {
			t.Fatalf("[%s] Invalid extension", test.name)
		}
		if len(ext) != len(test.ext) {
			t.Errorf("[%s] Wrong length of ComplexExtension: got: %v want: %v\n", test.name, len(ext), len(test.ext))
		}
		for i := range test.ext {
			if !proto.Equal(ext[i], test.ext[i]) {
				t.Errorf("[%s] Wrong value for ComplexExtension[%d]: got: %v want: %v\n", test.name, i, ext[i], test.ext[i])
			}
		}
	}
}

func TestUnmarshalRepeatingNonRepeatedExtension(t *testing.T) {
	// We may see multiple instances of the same extension in the wire
	// format. For example, the proto compiler may encode custom options in
	// this way. Here, we verify that we merge the extensions together.
	tests := []struct {
		name string
		ext  []*pb2.ComplexExtension
	}{
		{
			"two fields",
			[]*pb2.ComplexExtension{
				{First: proto.Int32(7)},
				{Second: proto.Int32(11)},
			},
		},
		{
			"repeated field",
			[]*pb2.ComplexExtension{
				{Third: []int32{1000}},
				{Third: []int32{2000}},
			},
		},
		{
			"two fields and repeated field",
			[]*pb2.ComplexExtension{
				{Third: []int32{1000}},
				{First: proto.Int32(9)},
				{Second: proto.Int32(21)},
				{Third: []int32{2000}},
			},
		},
	}
	for _, test := range tests {
		var buf bytes.Buffer
		var want pb2.ComplexExtension

		// Generate a serialized representation of a repeated extension
		// by catenating bytes together.
		for i, e := range test.ext {
			// Merge to create the wanted proto.
			proto.Merge(&want, e)

			// serialize the message
			msg := new(pb2.OtherMessage)
			err := proto.SetExtension(msg, pb2.E_Complex, e)
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
		msg2 := new(pb2.OtherMessage)
		err := proto.Unmarshal(buf.Bytes(), msg2)
		if err != nil {
			t.Fatalf("[%s] Error unmarshaling message: %v", test.name, err)
		}
		e, err := proto.GetExtension(msg2, pb2.E_Complex)
		if err != nil {
			t.Fatalf("[%s] Error getting extension: %v", test.name, err)
		}
		ext := e.(*pb2.ComplexExtension)
		if ext == nil {
			t.Fatalf("[%s] Invalid extension", test.name)
		}
		if !proto.Equal(ext, &want) {
			t.Errorf("[%s] Wrong value for ComplexExtension: got: %s want: %s\n", test.name, ext, &want)
		}
	}
}

func TestClearAllExtensions(t *testing.T) {
	// unregistered extension
	desc := &proto.ExtensionDesc{
		ExtendedType:  (*pb2.MyMessage)(nil),
		ExtensionType: (*bool)(nil),
		Field:         101010100,
		Name:          "emptyextension",
		Tag:           "varint,0,opt",
	}
	m := &pb2.MyMessage{}
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
	ext := &pb2.Ext{}
	m := &pb2.MyMessage{Count: proto.Int32(4)}
	if err := proto.SetExtension(m, pb2.E_Ext_More, ext); err != nil {
		t.Fatalf("proto.SetExtension(m, desc, true): got error %q, want nil", err)
	}

	b, err := proto.Marshal(m)
	if err != nil {
		t.Fatalf("Could not marshal message: %v", err)
	}
	if err := proto.Unmarshal(b, m); err != nil {
		t.Fatalf("Could not unmarshal message: %v", err)
	}
	// after Unmarshal, the extension is in undecoded form.
	// GetExtension will decode it lazily. Make sure this does
	// not race against Marshal.

	wg := sync.WaitGroup{}
	errs := make(chan error, 3)
	for n := 3; n > 0; n-- {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := proto.Marshal(m)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)

	for err = range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}
