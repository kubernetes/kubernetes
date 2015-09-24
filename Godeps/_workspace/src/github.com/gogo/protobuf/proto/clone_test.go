// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2011 The Go Authors.  All rights reserved.
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
	"testing"

	"github.com/gogo/protobuf/proto"

	proto3pb "github.com/gogo/protobuf/proto/proto3_proto"
	pb "github.com/gogo/protobuf/proto/testdata"
)

var cloneTestMessage = &pb.MyMessage{
	Count: proto.Int32(42),
	Name:  proto.String("Dave"),
	Pet:   []string{"bunny", "kitty", "horsey"},
	Inner: &pb.InnerMessage{
		Host:      proto.String("niles"),
		Port:      proto.Int32(9099),
		Connected: proto.Bool(true),
	},
	Others: []*pb.OtherMessage{
		{
			Value: []byte("some bytes"),
		},
	},
	Somegroup: &pb.MyMessage_SomeGroup{
		GroupField: proto.Int32(6),
	},
	RepBytes: [][]byte{[]byte("sham"), []byte("wow")},
}

func init() {
	ext := &pb.Ext{
		Data: proto.String("extension"),
	}
	if err := proto.SetExtension(cloneTestMessage, pb.E_Ext_More, ext); err != nil {
		panic("SetExtension: " + err.Error())
	}
}

func TestClone(t *testing.T) {
	m := proto.Clone(cloneTestMessage).(*pb.MyMessage)
	if !proto.Equal(m, cloneTestMessage) {
		t.Errorf("Clone(%v) = %v", cloneTestMessage, m)
	}

	// Verify it was a deep copy.
	*m.Inner.Port++
	if proto.Equal(m, cloneTestMessage) {
		t.Error("Mutating clone changed the original")
	}
	// Byte fields and repeated fields should be copied.
	if &m.Pet[0] == &cloneTestMessage.Pet[0] {
		t.Error("Pet: repeated field not copied")
	}
	if &m.Others[0] == &cloneTestMessage.Others[0] {
		t.Error("Others: repeated field not copied")
	}
	if &m.Others[0].Value[0] == &cloneTestMessage.Others[0].Value[0] {
		t.Error("Others[0].Value: bytes field not copied")
	}
	if &m.RepBytes[0] == &cloneTestMessage.RepBytes[0] {
		t.Error("RepBytes: repeated field not copied")
	}
	if &m.RepBytes[0][0] == &cloneTestMessage.RepBytes[0][0] {
		t.Error("RepBytes[0]: bytes field not copied")
	}
}

func TestCloneNil(t *testing.T) {
	var m *pb.MyMessage
	if c := proto.Clone(m); !proto.Equal(m, c) {
		t.Errorf("Clone(%v) = %v", m, c)
	}
}

var mergeTests = []struct {
	src, dst, want proto.Message
}{
	{
		src: &pb.MyMessage{
			Count: proto.Int32(42),
		},
		dst: &pb.MyMessage{
			Name: proto.String("Dave"),
		},
		want: &pb.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("Dave"),
		},
	},
	{
		src: &pb.MyMessage{
			Inner: &pb.InnerMessage{
				Host:      proto.String("hey"),
				Connected: proto.Bool(true),
			},
			Pet: []string{"horsey"},
			Others: []*pb.OtherMessage{
				{
					Value: []byte("some bytes"),
				},
			},
		},
		dst: &pb.MyMessage{
			Inner: &pb.InnerMessage{
				Host: proto.String("niles"),
				Port: proto.Int32(9099),
			},
			Pet: []string{"bunny", "kitty"},
			Others: []*pb.OtherMessage{
				{
					Key: proto.Int64(31415926535),
				},
				{
					// Explicitly test a src=nil field
					Inner: nil,
				},
			},
		},
		want: &pb.MyMessage{
			Inner: &pb.InnerMessage{
				Host:      proto.String("hey"),
				Connected: proto.Bool(true),
				Port:      proto.Int32(9099),
			},
			Pet: []string{"bunny", "kitty", "horsey"},
			Others: []*pb.OtherMessage{
				{
					Key: proto.Int64(31415926535),
				},
				{},
				{
					Value: []byte("some bytes"),
				},
			},
		},
	},
	{
		src: &pb.MyMessage{
			RepBytes: [][]byte{[]byte("wow")},
		},
		dst: &pb.MyMessage{
			Somegroup: &pb.MyMessage_SomeGroup{
				GroupField: proto.Int32(6),
			},
			RepBytes: [][]byte{[]byte("sham")},
		},
		want: &pb.MyMessage{
			Somegroup: &pb.MyMessage_SomeGroup{
				GroupField: proto.Int32(6),
			},
			RepBytes: [][]byte{[]byte("sham"), []byte("wow")},
		},
	},
	// Check that a scalar bytes field replaces rather than appends.
	{
		src:  &pb.OtherMessage{Value: []byte("foo")},
		dst:  &pb.OtherMessage{Value: []byte("bar")},
		want: &pb.OtherMessage{Value: []byte("foo")},
	},
	{
		src: &pb.MessageWithMap{
			NameMapping: map[int32]string{6: "Nigel"},
			MsgMapping: map[int64]*pb.FloatingPoint{
				0x4001: {F: proto.Float64(2.0)},
			},
			ByteMapping: map[bool][]byte{true: []byte("wowsa")},
		},
		dst: &pb.MessageWithMap{
			NameMapping: map[int32]string{
				6: "Bruce", // should be overwritten
				7: "Andrew",
			},
		},
		want: &pb.MessageWithMap{
			NameMapping: map[int32]string{
				6: "Nigel",
				7: "Andrew",
			},
			MsgMapping: map[int64]*pb.FloatingPoint{
				0x4001: {F: proto.Float64(2.0)},
			},
			ByteMapping: map[bool][]byte{true: []byte("wowsa")},
		},
	},
	// proto3 shouldn't merge zero values,
	// in the same way that proto2 shouldn't merge nils.
	{
		src: &proto3pb.Message{
			Name: "Aaron",
			Data: []byte(""), // zero value, but not nil
		},
		dst: &proto3pb.Message{
			HeightInCm: 176,
			Data:       []byte("texas!"),
		},
		want: &proto3pb.Message{
			Name:       "Aaron",
			HeightInCm: 176,
			Data:       []byte("texas!"),
		},
	},
	// Oneof fields should merge by assignment.
	{
		src: &pb.Communique{
			Union: &pb.Communique_Number{Number: 41},
		},
		dst: &pb.Communique{
			Union: &pb.Communique_Name{Name: "Bobby Tables"},
		},
		want: &pb.Communique{
			Union: &pb.Communique_Number{Number: 41},
		},
	},
	// Oneof nil is the same as not set.
	{
		src: &pb.Communique{},
		dst: &pb.Communique{
			Union: &pb.Communique_Name{Name: "Bobby Tables"},
		},
		want: &pb.Communique{
			Union: &pb.Communique_Name{Name: "Bobby Tables"},
		},
	},
}

func TestMerge(t *testing.T) {
	for _, m := range mergeTests {
		got := proto.Clone(m.dst)
		proto.Merge(got, m.src)
		if !proto.Equal(got, m.want) {
			t.Errorf("Merge(%v, %v)\n got %v\nwant %v\n", m.dst, m.src, got, m.want)
		}
	}
}
