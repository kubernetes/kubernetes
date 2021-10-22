// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2018, The GoGo Authors. All rights reserved.
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

package proto

type float64Value struct {
	Value float64 `protobuf:"fixed64,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *float64Value) Reset()       { *m = float64Value{} }
func (*float64Value) ProtoMessage()  {}
func (*float64Value) String() string { return "float64<string>" }

type float32Value struct {
	Value float32 `protobuf:"fixed32,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *float32Value) Reset()       { *m = float32Value{} }
func (*float32Value) ProtoMessage()  {}
func (*float32Value) String() string { return "float32<string>" }

type int64Value struct {
	Value int64 `protobuf:"varint,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *int64Value) Reset()       { *m = int64Value{} }
func (*int64Value) ProtoMessage()  {}
func (*int64Value) String() string { return "int64<string>" }

type uint64Value struct {
	Value uint64 `protobuf:"varint,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *uint64Value) Reset()       { *m = uint64Value{} }
func (*uint64Value) ProtoMessage()  {}
func (*uint64Value) String() string { return "uint64<string>" }

type int32Value struct {
	Value int32 `protobuf:"varint,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *int32Value) Reset()       { *m = int32Value{} }
func (*int32Value) ProtoMessage()  {}
func (*int32Value) String() string { return "int32<string>" }

type uint32Value struct {
	Value uint32 `protobuf:"varint,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *uint32Value) Reset()       { *m = uint32Value{} }
func (*uint32Value) ProtoMessage()  {}
func (*uint32Value) String() string { return "uint32<string>" }

type boolValue struct {
	Value bool `protobuf:"varint,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *boolValue) Reset()       { *m = boolValue{} }
func (*boolValue) ProtoMessage()  {}
func (*boolValue) String() string { return "bool<string>" }

type stringValue struct {
	Value string `protobuf:"bytes,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *stringValue) Reset()       { *m = stringValue{} }
func (*stringValue) ProtoMessage()  {}
func (*stringValue) String() string { return "string<string>" }

type bytesValue struct {
	Value []byte `protobuf:"bytes,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *bytesValue) Reset()       { *m = bytesValue{} }
func (*bytesValue) ProtoMessage()  {}
func (*bytesValue) String() string { return "[]byte<string>" }

func init() {
	RegisterType((*float64Value)(nil), "gogo.protobuf.proto.DoubleValue")
	RegisterType((*float32Value)(nil), "gogo.protobuf.proto.FloatValue")
	RegisterType((*int64Value)(nil), "gogo.protobuf.proto.Int64Value")
	RegisterType((*uint64Value)(nil), "gogo.protobuf.proto.UInt64Value")
	RegisterType((*int32Value)(nil), "gogo.protobuf.proto.Int32Value")
	RegisterType((*uint32Value)(nil), "gogo.protobuf.proto.UInt32Value")
	RegisterType((*boolValue)(nil), "gogo.protobuf.proto.BoolValue")
	RegisterType((*stringValue)(nil), "gogo.protobuf.proto.StringValue")
	RegisterType((*bytesValue)(nil), "gogo.protobuf.proto.BytesValue")
}
