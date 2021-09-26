// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"testing"

	"github.com/golang/protobuf/proto"
	"google.golang.org/protobuf/testing/protopack"

	pb2 "github.com/golang/protobuf/internal/testprotos/proto2_proto"
	pb3 "github.com/golang/protobuf/internal/testprotos/proto3_proto"
)

var rawFields = protopack.Message{
	protopack.Tag{5, protopack.Fixed32Type}, protopack.Uint32(4041331395),
}.Marshal()

func TestDiscardUnknown(t *testing.T) {
	tests := []struct {
		desc     string
		in, want proto.Message
	}{{
		desc: "Nil",
		in:   nil, want: nil, // Should not panic
	}, {
		desc: "NilPtr",
		in:   (*pb3.Message)(nil), want: (*pb3.Message)(nil), // Should not panic
	}, {
		desc: "Nested",
		in: &pb3.Message{
			Name:             "Aaron",
			Nested:           &pb3.Nested{Cute: true, XXX_unrecognized: []byte(rawFields)},
			XXX_unrecognized: []byte(rawFields),
		},
		want: &pb3.Message{
			Name:   "Aaron",
			Nested: &pb3.Nested{Cute: true},
		},
	}, {
		desc: "Slice",
		in: &pb3.Message{
			Name: "Aaron",
			Children: []*pb3.Message{
				{Name: "Sarah", XXX_unrecognized: []byte(rawFields)},
				{Name: "Abraham", XXX_unrecognized: []byte(rawFields)},
			},
			XXX_unrecognized: []byte(rawFields),
		},
		want: &pb3.Message{
			Name: "Aaron",
			Children: []*pb3.Message{
				{Name: "Sarah"},
				{Name: "Abraham"},
			},
		},
	}, {
		desc: "OneOf",
		in: &pb2.Communique{
			Union: &pb2.Communique_Msg{&pb2.Strings{
				StringField:      proto.String("123"),
				XXX_unrecognized: []byte(rawFields),
			}},
			XXX_unrecognized: []byte(rawFields),
		},
		want: &pb2.Communique{
			Union: &pb2.Communique_Msg{&pb2.Strings{StringField: proto.String("123")}},
		},
	}, {
		desc: "Map",
		in: &pb2.MessageWithMap{MsgMapping: map[int64]*pb2.FloatingPoint{
			0x4002: &pb2.FloatingPoint{
				Exact:            proto.Bool(true),
				XXX_unrecognized: []byte(rawFields),
			},
		}},
		want: &pb2.MessageWithMap{MsgMapping: map[int64]*pb2.FloatingPoint{
			0x4002: &pb2.FloatingPoint{Exact: proto.Bool(true)},
		}},
	}, {
		desc: "Extension",
		in: func() proto.Message {
			m := &pb2.MyMessage{
				Count: proto.Int32(42),
				Somegroup: &pb2.MyMessage_SomeGroup{
					GroupField:       proto.Int32(6),
					XXX_unrecognized: []byte(rawFields),
				},
				XXX_unrecognized: []byte(rawFields),
			}
			proto.SetExtension(m, pb2.E_Ext_More, &pb2.Ext{
				Data:             proto.String("extension"),
				XXX_unrecognized: []byte(rawFields),
			})
			return m
		}(),
		want: func() proto.Message {
			m := &pb2.MyMessage{
				Count:     proto.Int32(42),
				Somegroup: &pb2.MyMessage_SomeGroup{GroupField: proto.Int32(6)},
			}
			proto.SetExtension(m, pb2.E_Ext_More, &pb2.Ext{Data: proto.String("extension")})
			return m
		}(),
	}}

	for _, tt := range tests {
		proto.DiscardUnknown(tt.in)
		if !proto.Equal(tt.in, tt.want) {
			t.Errorf("test %s, expected unknown fields to be discarded\ngot  %v\nwant %v", tt.desc, tt.in, tt.want)
		}
	}
}
