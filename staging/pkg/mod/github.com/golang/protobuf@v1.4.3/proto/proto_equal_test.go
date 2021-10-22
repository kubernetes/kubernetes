// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"testing"

	"github.com/golang/protobuf/proto"

	pb2 "github.com/golang/protobuf/internal/testprotos/proto2_proto"
	pb3 "github.com/golang/protobuf/internal/testprotos/proto3_proto"
)

// Four identical base messages.
// The init function adds extensions to some of them.
var messageWithoutExtension = &pb2.MyMessage{Count: proto.Int32(7)}
var messageWithExtension1a = &pb2.MyMessage{Count: proto.Int32(7)}
var messageWithExtension1b = &pb2.MyMessage{Count: proto.Int32(7)}
var messageWithExtension2 = &pb2.MyMessage{Count: proto.Int32(7)}
var messageWithExtension3a = &pb2.MyMessage{Count: proto.Int32(7)}
var messageWithExtension3b = &pb2.MyMessage{Count: proto.Int32(7)}
var messageWithExtension3c = &pb2.MyMessage{Count: proto.Int32(7)}

// Two messages with non-message extensions.
var messageWithInt32Extension1 = &pb2.MyMessage{Count: proto.Int32(8)}
var messageWithInt32Extension2 = &pb2.MyMessage{Count: proto.Int32(8)}

func init() {
	ext1 := &pb2.Ext{Data: proto.String("Kirk")}
	ext2 := &pb2.Ext{Data: proto.String("Picard")}

	// messageWithExtension1a has ext1, but never marshals it.
	if err := proto.SetExtension(messageWithExtension1a, pb2.E_Ext_More, ext1); err != nil {
		panic("proto.SetExtension on 1a failed: " + err.Error())
	}

	// messageWithExtension1b is the unmarshaled form of messageWithExtension1a.
	if err := proto.SetExtension(messageWithExtension1b, pb2.E_Ext_More, ext1); err != nil {
		panic("proto.SetExtension on 1b failed: " + err.Error())
	}
	buf, err := proto.Marshal(messageWithExtension1b)
	if err != nil {
		panic("proto.Marshal of 1b failed: " + err.Error())
	}
	messageWithExtension1b.Reset()
	if err := proto.Unmarshal(buf, messageWithExtension1b); err != nil {
		panic("proto.Unmarshal of 1b failed: " + err.Error())
	}

	// messageWithExtension2 has ext2.
	if err := proto.SetExtension(messageWithExtension2, pb2.E_Ext_More, ext2); err != nil {
		panic("proto.SetExtension on 2 failed: " + err.Error())
	}

	if err := proto.SetExtension(messageWithInt32Extension1, pb2.E_Ext_Number, proto.Int32(23)); err != nil {
		panic("proto.SetExtension on Int32-1 failed: " + err.Error())
	}
	if err := proto.SetExtension(messageWithInt32Extension1, pb2.E_Ext_Number, proto.Int32(24)); err != nil {
		panic("proto.SetExtension on Int32-2 failed: " + err.Error())
	}

	// messageWithExtension3{a,b,c} has unregistered extension.
	if proto.RegisteredExtensions(messageWithExtension3a)[200] != nil {
		panic("expect extension 200 unregistered")
	}
	bytes := []byte{
		0xc0, 0x0c, 0x01, // id=200, wiretype=0 (varint), data=1
	}
	bytes2 := []byte{
		0xc0, 0x0c, 0x02, // id=200, wiretype=0 (varint), data=2
	}
	proto.SetRawExtension(messageWithExtension3a, 200, bytes)
	proto.SetRawExtension(messageWithExtension3b, 200, bytes)
	proto.SetRawExtension(messageWithExtension3c, 200, bytes2)
}

var EqualTests = []struct {
	desc string
	a, b proto.Message
	exp  bool
}{
	{"different types", &pb2.GoEnum{}, &pb2.GoTestField{}, false},
	{"equal empty", &pb2.GoEnum{}, &pb2.GoEnum{}, true},
	{"nil vs nil", nil, nil, true},
	{"typed nil vs typed nil", (*pb2.GoEnum)(nil), (*pb2.GoEnum)(nil), true},
	{"typed nil vs empty", (*pb2.GoEnum)(nil), &pb2.GoEnum{}, false},
	{"different typed nil", (*pb2.GoEnum)(nil), (*pb2.GoTestField)(nil), false},

	{"one set field, one unset field", &pb2.GoTestField{Label: proto.String("foo")}, &pb2.GoTestField{}, false},
	{"one set field zero, one unset field", &pb2.GoTest{Param: proto.Int32(0)}, &pb2.GoTest{}, false},
	{"different set fields", &pb2.GoTestField{Label: proto.String("foo")}, &pb2.GoTestField{Label: proto.String("bar")}, false},
	{"equal set", &pb2.GoTestField{Label: proto.String("foo")}, &pb2.GoTestField{Label: proto.String("foo")}, true},

	{"repeated, one set", &pb2.GoTest{F_Int32Repeated: []int32{2, 3}}, &pb2.GoTest{}, false},
	{"repeated, different length", &pb2.GoTest{F_Int32Repeated: []int32{2, 3}}, &pb2.GoTest{F_Int32Repeated: []int32{2}}, false},
	{"repeated, different value", &pb2.GoTest{F_Int32Repeated: []int32{2}}, &pb2.GoTest{F_Int32Repeated: []int32{3}}, false},
	{"repeated, equal", &pb2.GoTest{F_Int32Repeated: []int32{2, 4}}, &pb2.GoTest{F_Int32Repeated: []int32{2, 4}}, true},
	{"repeated, nil equal nil", &pb2.GoTest{F_Int32Repeated: nil}, &pb2.GoTest{F_Int32Repeated: nil}, true},
	{"repeated, nil equal empty", &pb2.GoTest{F_Int32Repeated: nil}, &pb2.GoTest{F_Int32Repeated: []int32{}}, true},
	{"repeated, empty equal nil", &pb2.GoTest{F_Int32Repeated: []int32{}}, &pb2.GoTest{F_Int32Repeated: nil}, true},

	{
		"nested, different",
		&pb2.GoTest{RequiredField: &pb2.GoTestField{Label: proto.String("foo")}},
		&pb2.GoTest{RequiredField: &pb2.GoTestField{Label: proto.String("bar")}},
		false,
	},
	{
		"nested, equal",
		&pb2.GoTest{RequiredField: &pb2.GoTestField{Label: proto.String("wow")}},
		&pb2.GoTest{RequiredField: &pb2.GoTestField{Label: proto.String("wow")}},
		true,
	},

	{"bytes", &pb2.OtherMessage{Value: []byte("foo")}, &pb2.OtherMessage{Value: []byte("foo")}, true},
	{"bytes, empty", &pb2.OtherMessage{Value: []byte{}}, &pb2.OtherMessage{Value: []byte{}}, true},
	{"bytes, empty vs nil", &pb2.OtherMessage{Value: []byte{}}, &pb2.OtherMessage{Value: nil}, false},
	{
		"repeated bytes",
		&pb2.MyMessage{RepBytes: [][]byte{[]byte("sham"), []byte("wow")}},
		&pb2.MyMessage{RepBytes: [][]byte{[]byte("sham"), []byte("wow")}},
		true,
	},
	// In proto3, []byte{} and []byte(nil) are equal.
	{"proto3 bytes, empty vs nil", &pb3.Message{Data: []byte{}}, &pb3.Message{Data: nil}, true},

	{"extension vs. no extension", messageWithoutExtension, messageWithExtension1a, false},
	{"extension vs. same extension", messageWithExtension1a, messageWithExtension1b, true},
	{"extension vs. different extension", messageWithExtension1a, messageWithExtension2, false},

	{"int32 extension vs. itself", messageWithInt32Extension1, messageWithInt32Extension1, true},
	{"int32 extension vs. a different int32", messageWithInt32Extension1, messageWithInt32Extension2, false},

	{"unregistered extension same", messageWithExtension3a, messageWithExtension3b, true},
	{"unregistered extension different", messageWithExtension3a, messageWithExtension3c, false},

	{
		"message with group",
		&pb2.MyMessage{
			Count: proto.Int32(1),
			Somegroup: &pb2.MyMessage_SomeGroup{
				GroupField: proto.Int32(5),
			},
		},
		&pb2.MyMessage{
			Count: proto.Int32(1),
			Somegroup: &pb2.MyMessage_SomeGroup{
				GroupField: proto.Int32(5),
			},
		},
		true,
	},

	{
		"map same",
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Ken"}},
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Ken"}},
		true,
	},
	{
		"map different entry",
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Ken"}},
		&pb2.MessageWithMap{NameMapping: map[int32]string{2: "Rob"}},
		false,
	},
	{
		"map different key only",
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Ken"}},
		&pb2.MessageWithMap{NameMapping: map[int32]string{2: "Ken"}},
		false,
	},
	{
		"map different value only",
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Ken"}},
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Rob"}},
		false,
	},
	{
		"zero-length maps same",
		&pb2.MessageWithMap{NameMapping: map[int32]string{}},
		&pb2.MessageWithMap{NameMapping: nil},
		true,
	},
	{
		"orders in map don't matter",
		&pb2.MessageWithMap{NameMapping: map[int32]string{1: "Ken", 2: "Rob"}},
		&pb2.MessageWithMap{NameMapping: map[int32]string{2: "Rob", 1: "Ken"}},
		true,
	},
	{
		"oneof same",
		&pb2.Communique{Union: &pb2.Communique_Number{41}},
		&pb2.Communique{Union: &pb2.Communique_Number{41}},
		true,
	},
	{
		"oneof one nil",
		&pb2.Communique{Union: &pb2.Communique_Number{41}},
		&pb2.Communique{},
		false,
	},
	{
		"oneof different",
		&pb2.Communique{Union: &pb2.Communique_Number{41}},
		&pb2.Communique{Union: &pb2.Communique_Name{"Bobby Tables"}},
		false,
	},
}

func TestEqual(t *testing.T) {
	for _, tc := range EqualTests {
		if res := proto.Equal(tc.a, tc.b); res != tc.exp {
			t.Errorf("%v: Equal(%v, %v) = %v, want %v", tc.desc, tc.a, tc.b, res, tc.exp)
		}
	}
}
