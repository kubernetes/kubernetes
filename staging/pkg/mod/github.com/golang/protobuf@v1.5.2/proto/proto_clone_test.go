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

var cloneTestMessage = &pb2.MyMessage{
	Count: proto.Int32(42),
	Name:  proto.String("Dave"),
	Pet:   []string{"bunny", "kitty", "horsey"},
	Inner: &pb2.InnerMessage{
		Host:      proto.String("niles"),
		Port:      proto.Int32(9099),
		Connected: proto.Bool(true),
	},
	Others: []*pb2.OtherMessage{
		{
			Value: []byte("some bytes"),
		},
	},
	Somegroup: &pb2.MyMessage_SomeGroup{
		GroupField: proto.Int32(6),
	},
	RepBytes: [][]byte{[]byte("sham"), []byte("wow")},
}

func init() {
	ext := &pb2.Ext{
		Data: proto.String("extension"),
	}
	if err := proto.SetExtension(cloneTestMessage, pb2.E_Ext_More, ext); err != nil {
		panic("SetExtension: " + err.Error())
	}
	if err := proto.SetExtension(cloneTestMessage, pb2.E_Ext_Text, proto.String("hello")); err != nil {
		panic("SetExtension: " + err.Error())
	}
	if err := proto.SetExtension(cloneTestMessage, pb2.E_Greeting, []string{"one", "two"}); err != nil {
		panic("SetExtension: " + err.Error())
	}
}

func TestClone(t *testing.T) {
	// Create a clone using a marshal/unmarshal roundtrip.
	vanilla := new(pb2.MyMessage)
	b, err := proto.Marshal(cloneTestMessage)
	if err != nil {
		t.Errorf("unexpected Marshal error: %v", err)
	}
	if err := proto.Unmarshal(b, vanilla); err != nil {
		t.Errorf("unexpected Unarshal error: %v", err)
	}

	// Create a clone using Clone and verify that it is equal to the original.
	m := proto.Clone(cloneTestMessage).(*pb2.MyMessage)
	if !proto.Equal(m, cloneTestMessage) {
		t.Fatalf("Clone(%v) = %v", cloneTestMessage, m)
	}

	// Mutate the clone, which should not affect the original.
	x1, err := proto.GetExtension(m, pb2.E_Ext_More)
	if err != nil {
		t.Errorf("unexpected GetExtension(%v) error: %v", pb2.E_Ext_More.Name, err)
	}
	x2, err := proto.GetExtension(m, pb2.E_Ext_Text)
	if err != nil {
		t.Errorf("unexpected GetExtension(%v) error: %v", pb2.E_Ext_Text.Name, err)
	}
	x3, err := proto.GetExtension(m, pb2.E_Greeting)
	if err != nil {
		t.Errorf("unexpected GetExtension(%v) error: %v", pb2.E_Greeting.Name, err)
	}
	*m.Inner.Port++
	*(x1.(*pb2.Ext)).Data = "blah blah"
	*(x2.(*string)) = "goodbye"
	x3.([]string)[0] = "zero"
	if !proto.Equal(cloneTestMessage, vanilla) {
		t.Fatalf("mutation on original detected:\ngot  %v\nwant %v", cloneTestMessage, vanilla)
	}
}

func TestCloneNil(t *testing.T) {
	var m *pb2.MyMessage
	if c := proto.Clone(m); !proto.Equal(m, c) {
		t.Errorf("Clone(%v) = %v", m, c)
	}
}

var mergeTests = []struct {
	src, dst, want proto.Message
}{
	{
		src: &pb2.MyMessage{
			Count: proto.Int32(42),
		},
		dst: &pb2.MyMessage{
			Name: proto.String("Dave"),
		},
		want: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("Dave"),
		},
	},
	{
		src: &pb2.MyMessage{
			Inner: &pb2.InnerMessage{
				Host:      proto.String("hey"),
				Connected: proto.Bool(true),
			},
			Pet: []string{"horsey"},
			Others: []*pb2.OtherMessage{
				{
					Value: []byte("some bytes"),
				},
			},
		},
		dst: &pb2.MyMessage{
			Inner: &pb2.InnerMessage{
				Host: proto.String("niles"),
				Port: proto.Int32(9099),
			},
			Pet: []string{"bunny", "kitty"},
			Others: []*pb2.OtherMessage{
				{
					Key: proto.Int64(31415926535),
				},
				{
					// Explicitly test a src=nil field
					Inner: nil,
				},
			},
		},
		want: &pb2.MyMessage{
			Inner: &pb2.InnerMessage{
				Host:      proto.String("hey"),
				Connected: proto.Bool(true),
				Port:      proto.Int32(9099),
			},
			Pet: []string{"bunny", "kitty", "horsey"},
			Others: []*pb2.OtherMessage{
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
		src: &pb2.MyMessage{
			RepBytes: [][]byte{[]byte("wow")},
		},
		dst: &pb2.MyMessage{
			Somegroup: &pb2.MyMessage_SomeGroup{
				GroupField: proto.Int32(6),
			},
			RepBytes: [][]byte{[]byte("sham")},
		},
		want: &pb2.MyMessage{
			Somegroup: &pb2.MyMessage_SomeGroup{
				GroupField: proto.Int32(6),
			},
			RepBytes: [][]byte{[]byte("sham"), []byte("wow")},
		},
	},
	// Check that a scalar bytes field replaces rather than appends.
	{
		src:  &pb2.OtherMessage{Value: []byte("foo")},
		dst:  &pb2.OtherMessage{Value: []byte("bar")},
		want: &pb2.OtherMessage{Value: []byte("foo")},
	},
	{
		src: &pb2.MessageWithMap{
			NameMapping: map[int32]string{6: "Nigel"},
			MsgMapping: map[int64]*pb2.FloatingPoint{
				0x4001: &pb2.FloatingPoint{F: proto.Float64(2.0)},
				0x4002: &pb2.FloatingPoint{
					F: proto.Float64(2.0),
				},
			},
			ByteMapping: map[bool][]byte{true: []byte("wowsa")},
		},
		dst: &pb2.MessageWithMap{
			NameMapping: map[int32]string{
				6: "Bruce", // should be overwritten
				7: "Andrew",
			},
			MsgMapping: map[int64]*pb2.FloatingPoint{
				0x4002: &pb2.FloatingPoint{
					F:     proto.Float64(3.0),
					Exact: proto.Bool(true),
				}, // the entire message should be overwritten
			},
		},
		want: &pb2.MessageWithMap{
			NameMapping: map[int32]string{
				6: "Nigel",
				7: "Andrew",
			},
			MsgMapping: map[int64]*pb2.FloatingPoint{
				0x4001: &pb2.FloatingPoint{F: proto.Float64(2.0)},
				0x4002: &pb2.FloatingPoint{
					F: proto.Float64(2.0),
				},
			},
			ByteMapping: map[bool][]byte{true: []byte("wowsa")},
		},
	},
	// proto3 shouldn't merge zero values,
	// in the same way that proto2 shouldn't merge nils.
	{
		src: &pb3.Message{
			Name: "Aaron",
			Data: []byte(""), // zero value, but not nil
		},
		dst: &pb3.Message{
			HeightInCm: 176,
			Data:       []byte("texas!"),
		},
		want: &pb3.Message{
			Name:       "Aaron",
			HeightInCm: 176,
			Data:       []byte("texas!"),
		},
	},
	{ // Oneof fields should merge by assignment.
		src:  &pb2.Communique{Union: &pb2.Communique_Number{41}},
		dst:  &pb2.Communique{Union: &pb2.Communique_Name{"Bobby Tables"}},
		want: &pb2.Communique{Union: &pb2.Communique_Number{41}},
	},
	{ // Oneof nil is the same as not set.
		src:  &pb2.Communique{},
		dst:  &pb2.Communique{Union: &pb2.Communique_Name{"Bobby Tables"}},
		want: &pb2.Communique{Union: &pb2.Communique_Name{"Bobby Tables"}},
	},
	{
		src:  &pb2.Communique{Union: &pb2.Communique_Number{1337}},
		dst:  &pb2.Communique{},
		want: &pb2.Communique{Union: &pb2.Communique_Number{1337}},
	},
	{
		src:  &pb2.Communique{Union: &pb2.Communique_Col{pb2.MyMessage_RED}},
		dst:  &pb2.Communique{},
		want: &pb2.Communique{Union: &pb2.Communique_Col{pb2.MyMessage_RED}},
	},
	{
		src:  &pb2.Communique{Union: &pb2.Communique_Data{[]byte("hello")}},
		dst:  &pb2.Communique{},
		want: &pb2.Communique{Union: &pb2.Communique_Data{[]byte("hello")}},
	},
	{
		src:  &pb2.Communique{Union: &pb2.Communique_Msg{&pb2.Strings{BytesField: []byte{1, 2, 3}}}},
		dst:  &pb2.Communique{},
		want: &pb2.Communique{Union: &pb2.Communique_Msg{&pb2.Strings{BytesField: []byte{1, 2, 3}}}},
	},
	{
		src:  &pb2.Communique{Union: &pb2.Communique_Msg{}},
		dst:  &pb2.Communique{},
		want: &pb2.Communique{Union: &pb2.Communique_Msg{}},
	},
	{
		src:  &pb2.Communique{Union: &pb2.Communique_Msg{&pb2.Strings{StringField: proto.String("123")}}},
		dst:  &pb2.Communique{Union: &pb2.Communique_Msg{&pb2.Strings{BytesField: []byte{1, 2, 3}}}},
		want: &pb2.Communique{Union: &pb2.Communique_Msg{&pb2.Strings{StringField: proto.String("123"), BytesField: []byte{1, 2, 3}}}},
	},
	{
		src: &pb3.Message{
			Terrain: map[string]*pb3.Nested{
				"kay_a": &pb3.Nested{Cute: true},      // replace
				"kay_b": &pb3.Nested{Bunny: "rabbit"}, // insert
			},
		},
		dst: &pb3.Message{
			Terrain: map[string]*pb3.Nested{
				"kay_a": &pb3.Nested{Bunny: "lost"},  // replaced
				"kay_c": &pb3.Nested{Bunny: "bunny"}, // keep
			},
		},
		want: &pb3.Message{
			Terrain: map[string]*pb3.Nested{
				"kay_a": &pb3.Nested{Cute: true},
				"kay_b": &pb3.Nested{Bunny: "rabbit"},
				"kay_c": &pb3.Nested{Bunny: "bunny"},
			},
		},
	},
	{
		src: &pb2.GoTest{
			F_BoolRepeated:   []bool{},
			F_Int32Repeated:  []int32{},
			F_Int64Repeated:  []int64{},
			F_Uint32Repeated: []uint32{},
			F_Uint64Repeated: []uint64{},
			F_FloatRepeated:  []float32{},
			F_DoubleRepeated: []float64{},
			F_StringRepeated: []string{},
			F_BytesRepeated:  [][]byte{},
		},
		dst: &pb2.GoTest{},
		want: &pb2.GoTest{
			F_BoolRepeated:   []bool{},
			F_Int32Repeated:  []int32{},
			F_Int64Repeated:  []int64{},
			F_Uint32Repeated: []uint32{},
			F_Uint64Repeated: []uint64{},
			F_FloatRepeated:  []float32{},
			F_DoubleRepeated: []float64{},
			F_StringRepeated: []string{},
			F_BytesRepeated:  [][]byte{},
		},
	},
	{
		src: &pb2.GoTest{},
		dst: &pb2.GoTest{
			F_BoolRepeated:   []bool{},
			F_Int32Repeated:  []int32{},
			F_Int64Repeated:  []int64{},
			F_Uint32Repeated: []uint32{},
			F_Uint64Repeated: []uint64{},
			F_FloatRepeated:  []float32{},
			F_DoubleRepeated: []float64{},
			F_StringRepeated: []string{},
			F_BytesRepeated:  [][]byte{},
		},
		want: &pb2.GoTest{
			F_BoolRepeated:   []bool{},
			F_Int32Repeated:  []int32{},
			F_Int64Repeated:  []int64{},
			F_Uint32Repeated: []uint32{},
			F_Uint64Repeated: []uint64{},
			F_FloatRepeated:  []float32{},
			F_DoubleRepeated: []float64{},
			F_StringRepeated: []string{},
			F_BytesRepeated:  [][]byte{},
		},
	},
	{
		src: &pb2.GoTest{
			F_BytesRepeated: [][]byte{nil, []byte{}, []byte{0}},
		},
		dst: &pb2.GoTest{},
		want: &pb2.GoTest{
			F_BytesRepeated: [][]byte{nil, []byte{}, []byte{0}},
		},
	},
	{
		src: &pb2.MyMessage{
			Others: []*pb2.OtherMessage{},
		},
		dst: &pb2.MyMessage{},
		want: &pb2.MyMessage{
			Others: []*pb2.OtherMessage{},
		},
	},
}

func TestMerge(t *testing.T) {
	for _, m := range mergeTests {
		got := proto.Clone(m.dst)
		if !proto.Equal(got, m.dst) {
			t.Errorf("Clone()\ngot  %v\nwant %v", got, m.dst)
			continue
		}
		proto.Merge(got, m.src)
		if !proto.Equal(got, m.want) {
			t.Errorf("Merge(%v, %v)\ngot  %v\nwant %v", m.dst, m.src, got, m.want)
		}
	}
}
