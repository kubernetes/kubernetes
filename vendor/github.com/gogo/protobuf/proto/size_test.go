// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2012 The Go Authors.  All rights reserved.
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
	"log"
	"strings"
	"testing"

	. "github.com/gogo/protobuf/proto"
	proto3pb "github.com/gogo/protobuf/proto/proto3_proto"
	pb "github.com/gogo/protobuf/proto/testdata"
)

var messageWithExtension1 = &pb.MyMessage{Count: Int32(7)}

// messageWithExtension2 is in equal_test.go.
var messageWithExtension3 = &pb.MyMessage{Count: Int32(8)}

func init() {
	if err := SetExtension(messageWithExtension1, pb.E_Ext_More, &pb.Ext{Data: String("Abbott")}); err != nil {
		log.Panicf("SetExtension: %v", err)
	}
	if err := SetExtension(messageWithExtension3, pb.E_Ext_More, &pb.Ext{Data: String("Costello")}); err != nil {
		log.Panicf("SetExtension: %v", err)
	}

	// Force messageWithExtension3 to have the extension encoded.
	Marshal(messageWithExtension3)

}

var SizeTests = []struct {
	desc string
	pb   Message
}{
	{"empty", &pb.OtherMessage{}},
	// Basic types.
	{"bool", &pb.Defaults{F_Bool: Bool(true)}},
	{"int32", &pb.Defaults{F_Int32: Int32(12)}},
	{"negative int32", &pb.Defaults{F_Int32: Int32(-1)}},
	{"small int64", &pb.Defaults{F_Int64: Int64(1)}},
	{"big int64", &pb.Defaults{F_Int64: Int64(1 << 20)}},
	{"negative int64", &pb.Defaults{F_Int64: Int64(-1)}},
	{"fixed32", &pb.Defaults{F_Fixed32: Uint32(71)}},
	{"fixed64", &pb.Defaults{F_Fixed64: Uint64(72)}},
	{"uint32", &pb.Defaults{F_Uint32: Uint32(123)}},
	{"uint64", &pb.Defaults{F_Uint64: Uint64(124)}},
	{"float", &pb.Defaults{F_Float: Float32(12.6)}},
	{"double", &pb.Defaults{F_Double: Float64(13.9)}},
	{"string", &pb.Defaults{F_String: String("niles")}},
	{"bytes", &pb.Defaults{F_Bytes: []byte("wowsa")}},
	{"bytes, empty", &pb.Defaults{F_Bytes: []byte{}}},
	{"sint32", &pb.Defaults{F_Sint32: Int32(65)}},
	{"sint64", &pb.Defaults{F_Sint64: Int64(67)}},
	{"enum", &pb.Defaults{F_Enum: pb.Defaults_BLUE.Enum()}},
	// Repeated.
	{"empty repeated bool", &pb.MoreRepeated{Bools: []bool{}}},
	{"repeated bool", &pb.MoreRepeated{Bools: []bool{false, true, true, false}}},
	{"packed repeated bool", &pb.MoreRepeated{BoolsPacked: []bool{false, true, true, false, true, true, true}}},
	{"repeated int32", &pb.MoreRepeated{Ints: []int32{1, 12203, 1729, -1}}},
	{"repeated int32 packed", &pb.MoreRepeated{IntsPacked: []int32{1, 12203, 1729}}},
	{"repeated int64 packed", &pb.MoreRepeated{Int64SPacked: []int64{
		// Need enough large numbers to verify that the header is counting the number of bytes
		// for the field, not the number of elements.
		1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62,
		1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62, 1 << 62,
	}}},
	{"repeated string", &pb.MoreRepeated{Strings: []string{"r", "ken", "gri"}}},
	{"repeated fixed", &pb.MoreRepeated{Fixeds: []uint32{1, 2, 3, 4}}},
	// Nested.
	{"nested", &pb.OldMessage{Nested: &pb.OldMessage_Nested{Name: String("whatever")}}},
	{"group", &pb.GroupOld{G: &pb.GroupOld_G{X: Int32(12345)}}},
	// Other things.
	{"unrecognized", &pb.MoreRepeated{XXX_unrecognized: []byte{13<<3 | 0, 4}}},
	{"extension (unencoded)", messageWithExtension1},
	{"extension (encoded)", messageWithExtension3},
	// proto3 message
	{"proto3 empty", &proto3pb.Message{}},
	{"proto3 bool", &proto3pb.Message{TrueScotsman: true}},
	{"proto3 int64", &proto3pb.Message{ResultCount: 1}},
	{"proto3 uint32", &proto3pb.Message{HeightInCm: 123}},
	{"proto3 float", &proto3pb.Message{Score: 12.6}},
	{"proto3 string", &proto3pb.Message{Name: "Snezana"}},
	{"proto3 bytes", &proto3pb.Message{Data: []byte("wowsa")}},
	{"proto3 bytes, empty", &proto3pb.Message{Data: []byte{}}},
	{"proto3 enum", &proto3pb.Message{Hilarity: proto3pb.Message_PUNS}},
	{"proto3 map field with empty bytes", &proto3pb.MessageWithMap{ByteMapping: map[bool][]byte{false: {}}}},

	{"map field", &pb.MessageWithMap{NameMapping: map[int32]string{1: "Rob", 7: "Andrew"}}},
	{"map field with message", &pb.MessageWithMap{MsgMapping: map[int64]*pb.FloatingPoint{0x7001: {F: Float64(2.0)}}}},
	{"map field with bytes", &pb.MessageWithMap{ByteMapping: map[bool][]byte{true: []byte("this time for sure")}}},
	{"map field with empty bytes", &pb.MessageWithMap{ByteMapping: map[bool][]byte{true: {}}}},

	{"map field with big entry", &pb.MessageWithMap{NameMapping: map[int32]string{8: strings.Repeat("x", 125)}}},
	{"map field with big key and val", &pb.MessageWithMap{StrToStr: map[string]string{strings.Repeat("x", 70): strings.Repeat("y", 70)}}},
	{"map field with big numeric key", &pb.MessageWithMap{NameMapping: map[int32]string{0xf00d: "om nom nom"}}},

	{"oneof not set", &pb.Communique{}},
	{"oneof zero int32", &pb.Communique{Union: &pb.Communique_Number{Number: 0}}},
	{"oneof int32", &pb.Communique{Union: &pb.Communique_Number{Number: 3}}},
	{"oneof string", &pb.Communique{Union: &pb.Communique_Name{Name: "Rhythmic Fman"}}},
}

func TestSize(t *testing.T) {
	for _, tc := range SizeTests {
		size := Size(tc.pb)
		b, err := Marshal(tc.pb)
		if err != nil {
			t.Errorf("%v: Marshal failed: %v", tc.desc, err)
			continue
		}
		if size != len(b) {
			t.Errorf("%v: Size(%v) = %d, want %d", tc.desc, tc.pb, size, len(b))
			t.Logf("%v: bytes: %#v", tc.desc, b)
		}
	}
}
