// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
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

/*
The marshalto plugin generates a Marshal and MarshalTo method for each message.
The `Marshal() ([]byte, error)` method results in the fact that the message
implements the Marshaler interface.
This allows proto.Marshal to be faster by calling the generated Marshal method rather than using reflect to Marshal the struct.

If is enabled by the following extensions:

  - marshaler
  - marshaler_all

Or the following extensions:

  - unsafe_marshaler
  - unsafe_marshaler_all

That is if you want to use the unsafe package in your generated code.
The speed up using the unsafe package is not very significant.

The generation of marshalling tests are enabled using one of the following extensions:

  - testgen
  - testgen_all

And benchmarks given it is enabled using one of the following extensions:

  - benchgen
  - benchgen_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

option (gogoproto.marshaler_all) = true;

message B {
	option (gogoproto.description) = true;
	optional A A = 1 [(gogoproto.nullable) = false, (gogoproto.embed) = true];
	repeated bytes G = 2 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uint128", (gogoproto.nullable) = false];
}

given to the marshalto plugin, will generate the following code:

  func (m *B) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
  }

  func (m *B) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	dAtA[i] = 0xa
	i++
	i = encodeVarintExample(dAtA, i, uint64(m.A.Size()))
	n2, err := m.A.MarshalTo(dAtA[i:])
	if err != nil {
		return 0, err
	}
	i += n2
	if len(m.G) > 0 {
		for _, msg := range m.G {
			dAtA[i] = 0x12
			i++
			i = encodeVarintExample(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if m.XXX_unrecognized != nil {
		i += copy(dAtA[i:], m.XXX_unrecognized)
	}
	return i, nil
  }

As shown above Marshal calculates the size of the not yet marshalled message
and allocates the appropriate buffer.
This is followed by calling the MarshalTo method which requires a preallocated buffer.
The MarshalTo method allows a user to rather preallocated a reusable buffer.

The Size method is generated using the size plugin and the gogoproto.sizer, gogoproto.sizer_all extensions.
The user can also using the generated Size method to check that his reusable buffer is still big enough.

The generated tests and benchmarks will keep you safe and show that this is really a significant speed improvement.

An additional message-level option `stable_marshaler` (and the file-level
option `stable_marshaler_all`) exists which causes the generated marshalling
code to behave deterministically. Today, this only changes the serialization of
maps; they are serialized in sort order.
*/
package marshalto

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
	"github.com/gogo/protobuf/vanity"
)

type NumGen interface {
	Next() string
	Current() string
}

type numGen struct {
	index int
}

func NewNumGen() NumGen {
	return &numGen{0}
}

func (this *numGen) Next() string {
	this.index++
	return this.Current()
}

func (this *numGen) Current() string {
	return strconv.Itoa(this.index)
}

type marshalto struct {
	*generator.Generator
	generator.PluginImports
	atleastOne  bool
	unsafePkg   generator.Single
	errorsPkg   generator.Single
	protoPkg    generator.Single
	sortKeysPkg generator.Single
	mathPkg     generator.Single
	typesPkg    generator.Single
	localName   string
	unsafe      bool
}

func NewMarshal() *marshalto {
	return &marshalto{}
}

func NewUnsafeMarshal() *marshalto {
	return &marshalto{unsafe: true}
}

func (p *marshalto) Name() string {
	if p.unsafe {
		return "unsafemarshaler"
	}
	return "marshalto"
}

func (p *marshalto) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *marshalto) callFixed64(varName ...string) {
	p.P(`i = encodeFixed64`, p.localName, `(dAtA, i, uint64(`, strings.Join(varName, ""), `))`)
}

func (p *marshalto) callFixed32(varName ...string) {
	p.P(`i = encodeFixed32`, p.localName, `(dAtA, i, uint32(`, strings.Join(varName, ""), `))`)
}

func (p *marshalto) callVarint(varName ...string) {
	p.P(`i = encodeVarint`, p.localName, `(dAtA, i, uint64(`, strings.Join(varName, ""), `))`)
}

func (p *marshalto) encodeVarint(varName string) {
	p.P(`for `, varName, ` >= 1<<7 {`)
	p.In()
	p.P(`dAtA[i] = uint8(uint64(`, varName, `)&0x7f|0x80)`)
	p.P(varName, ` >>= 7`)
	p.P(`i++`)
	p.Out()
	p.P(`}`)
	p.P(`dAtA[i] = uint8(`, varName, `)`)
	p.P(`i++`)
}

func (p *marshalto) encodeFixed64(varName string) {
	p.P(`dAtA[i] = uint8(`, varName, `)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 8)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 16)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 24)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 32)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 40)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 48)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 56)`)
	p.P(`i++`)
}

func (p *marshalto) unsafeFixed64(varName string, someType string) {
	p.P(`*(*`, someType, `)(`, p.unsafePkg.Use(), `.Pointer(&dAtA[i])) = `, varName)
	p.P(`i+=8`)
}

func (p *marshalto) encodeFixed32(varName string) {
	p.P(`dAtA[i] = uint8(`, varName, `)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 8)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 16)`)
	p.P(`i++`)
	p.P(`dAtA[i] = uint8(`, varName, ` >> 24)`)
	p.P(`i++`)
}

func (p *marshalto) unsafeFixed32(varName string, someType string) {
	p.P(`*(*`, someType, `)(`, p.unsafePkg.Use(), `.Pointer(&dAtA[i])) = `, varName)
	p.P(`i+=4`)
}

func (p *marshalto) encodeKey(fieldNumber int32, wireType int) {
	x := uint32(fieldNumber)<<3 | uint32(wireType)
	i := 0
	keybuf := make([]byte, 0)
	for i = 0; x > 127; i++ {
		keybuf = append(keybuf, 0x80|uint8(x&0x7F))
		x >>= 7
	}
	keybuf = append(keybuf, uint8(x))
	for _, b := range keybuf {
		p.P(`dAtA[i] = `, fmt.Sprintf("%#v", b))
		p.P(`i++`)
	}
}

func keySize(fieldNumber int32, wireType int) int {
	x := uint32(fieldNumber)<<3 | uint32(wireType)
	size := 0
	for size = 0; x > 127; size++ {
		x >>= 7
	}
	size++
	return size
}

func wireToType(wire string) int {
	switch wire {
	case "fixed64":
		return proto.WireFixed64
	case "fixed32":
		return proto.WireFixed32
	case "varint":
		return proto.WireVarint
	case "bytes":
		return proto.WireBytes
	case "group":
		return proto.WireBytes
	case "zigzag32":
		return proto.WireVarint
	case "zigzag64":
		return proto.WireVarint
	}
	panic("unreachable")
}

func (p *marshalto) mapField(numGen NumGen, field *descriptor.FieldDescriptorProto, kvField *descriptor.FieldDescriptorProto, varName string, protoSizer bool) {
	switch kvField.GetType() {
	case descriptor.FieldDescriptorProto_TYPE_DOUBLE:
		p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(`, varName, `))`)
	case descriptor.FieldDescriptorProto_TYPE_FLOAT:
		p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(`, varName, `))`)
	case descriptor.FieldDescriptorProto_TYPE_INT64,
		descriptor.FieldDescriptorProto_TYPE_UINT64,
		descriptor.FieldDescriptorProto_TYPE_INT32,
		descriptor.FieldDescriptorProto_TYPE_UINT32,
		descriptor.FieldDescriptorProto_TYPE_ENUM:
		p.callVarint(varName)
	case descriptor.FieldDescriptorProto_TYPE_FIXED64,
		descriptor.FieldDescriptorProto_TYPE_SFIXED64:
		p.callFixed64(varName)
	case descriptor.FieldDescriptorProto_TYPE_FIXED32,
		descriptor.FieldDescriptorProto_TYPE_SFIXED32:
		p.callFixed32(varName)
	case descriptor.FieldDescriptorProto_TYPE_BOOL:
		p.P(`if `, varName, ` {`)
		p.In()
		p.P(`dAtA[i] = 1`)
		p.Out()
		p.P(`} else {`)
		p.In()
		p.P(`dAtA[i] = 0`)
		p.Out()
		p.P(`}`)
		p.P(`i++`)
	case descriptor.FieldDescriptorProto_TYPE_STRING,
		descriptor.FieldDescriptorProto_TYPE_BYTES:
		if gogoproto.IsCustomType(field) && kvField.IsBytes() {
			p.callVarint(varName, `.Size()`)
			p.P(`n`, numGen.Next(), `, err := `, varName, `.MarshalTo(dAtA[i:])`)
			p.P(`if err != nil {`)
			p.In()
			p.P(`return 0, err`)
			p.Out()
			p.P(`}`)
			p.P(`i+=n`, numGen.Current())
		} else {
			p.callVarint(`len(`, varName, `)`)
			p.P(`i+=copy(dAtA[i:], `, varName, `)`)
		}
	case descriptor.FieldDescriptorProto_TYPE_SINT32:
		p.callVarint(`(uint32(`, varName, `) << 1) ^ uint32((`, varName, ` >> 31))`)
	case descriptor.FieldDescriptorProto_TYPE_SINT64:
		p.callVarint(`(uint64(`, varName, `) << 1) ^ uint64((`, varName, ` >> 63))`)
	case descriptor.FieldDescriptorProto_TYPE_MESSAGE:
		if gogoproto.IsStdTime(field) {
			p.callVarint(p.typesPkg.Use(), `.SizeOfStdTime(*`, varName, `)`)
			p.P(`n`, numGen.Next(), `, err := `, p.typesPkg.Use(), `.StdTimeMarshalTo(*`, varName, `, dAtA[i:])`)
		} else if gogoproto.IsStdDuration(field) {
			p.callVarint(p.typesPkg.Use(), `.SizeOfStdDuration(*`, varName, `)`)
			p.P(`n`, numGen.Next(), `, err := `, p.typesPkg.Use(), `.StdDurationMarshalTo(*`, varName, `, dAtA[i:])`)
		} else if protoSizer {
			p.callVarint(varName, `.ProtoSize()`)
			p.P(`n`, numGen.Next(), `, err := `, varName, `.MarshalTo(dAtA[i:])`)
		} else {
			p.callVarint(varName, `.Size()`)
			p.P(`n`, numGen.Next(), `, err := `, varName, `.MarshalTo(dAtA[i:])`)
		}
		p.P(`if err != nil {`)
		p.In()
		p.P(`return 0, err`)
		p.Out()
		p.P(`}`)
		p.P(`i+=n`, numGen.Current())
	}
}

type orderFields []*descriptor.FieldDescriptorProto

func (this orderFields) Len() int {
	return len(this)
}

func (this orderFields) Less(i, j int) bool {
	return this[i].GetNumber() < this[j].GetNumber()
}

func (this orderFields) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}

func (p *marshalto) generateField(proto3 bool, numGen NumGen, file *generator.FileDescriptor, message *generator.Descriptor, field *descriptor.FieldDescriptorProto) {
	fieldname := p.GetOneOfFieldName(message, field)
	nullable := gogoproto.IsNullable(field)
	repeated := field.IsRepeated()
	required := field.IsRequired()

	protoSizer := gogoproto.IsProtoSizer(file.FileDescriptorProto, message.DescriptorProto)
	doNilCheck := gogoproto.NeedsNilCheck(proto3, field)
	if required && nullable {
		p.P(`if m.`, fieldname, `== nil {`)
		p.In()
		if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
			p.P(`return 0, new(`, p.protoPkg.Use(), `.RequiredNotSetError)`)
		} else {
			p.P(`return 0, `, p.protoPkg.Use(), `.NewRequiredNotSetError("`, field.GetName(), `")`)
		}
		p.Out()
		p.P(`} else {`)
	} else if repeated {
		p.P(`if len(m.`, fieldname, `) > 0 {`)
		p.In()
	} else if doNilCheck {
		p.P(`if m.`, fieldname, ` != nil {`)
		p.In()
	}
	packed := field.IsPacked() || (proto3 && field.IsPacked3())
	wireType := field.WireType()
	fieldNumber := field.GetNumber()
	if packed {
		wireType = proto.WireBytes
	}
	switch *field.Type {
	case descriptor.FieldDescriptorProto_TYPE_DOUBLE:
		if !p.unsafe || gogoproto.IsCastType(field) {
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 8`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float64bits(float64(num))`)
				p.encodeFixed64("f" + numGen.Current())
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float64bits(float64(num))`)
				p.encodeFixed64("f" + numGen.Current())
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(m.`+fieldname, `))`)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(m.`+fieldname, `))`)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(*m.`+fieldname, `))`)
			}
		} else {
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 8`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.unsafeFixed64("num", "float64")
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64("num", "float64")
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64(`m.`+fieldname, "float64")
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64(`m.`+fieldname, "float64")
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64(`*m.`+fieldname, `float64`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_FLOAT:
		if !p.unsafe || gogoproto.IsCastType(field) {
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 4`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float32bits(float32(num))`)
				p.encodeFixed32("f" + numGen.Current())
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float32bits(float32(num))`)
				p.encodeFixed32("f" + numGen.Current())
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(m.`+fieldname, `))`)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(m.`+fieldname, `))`)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(*m.`+fieldname, `))`)
			}
		} else {
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 4`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.unsafeFixed32("num", "float32")
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32("num", "float32")
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32(`m.`+fieldname, `float32`)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32(`m.`+fieldname, `float32`)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32(`*m.`+fieldname, "float32")
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_INT64,
		descriptor.FieldDescriptorProto_TYPE_UINT64,
		descriptor.FieldDescriptorProto_TYPE_INT32,
		descriptor.FieldDescriptorProto_TYPE_UINT32,
		descriptor.FieldDescriptorProto_TYPE_ENUM:
		if packed {
			jvar := "j" + numGen.Next()
			p.P(`dAtA`, numGen.Next(), ` := make([]byte, len(m.`, fieldname, `)*10)`)
			p.P(`var `, jvar, ` int`)
			if *field.Type == descriptor.FieldDescriptorProto_TYPE_INT64 ||
				*field.Type == descriptor.FieldDescriptorProto_TYPE_INT32 {
				p.P(`for _, num1 := range m.`, fieldname, ` {`)
				p.In()
				p.P(`num := uint64(num1)`)
			} else {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
			}
			p.P(`for num >= 1<<7 {`)
			p.In()
			p.P(`dAtA`, numGen.Current(), `[`, jvar, `] = uint8(uint64(num)&0x7f|0x80)`)
			p.P(`num >>= 7`)
			p.P(jvar, `++`)
			p.Out()
			p.P(`}`)
			p.P(`dAtA`, numGen.Current(), `[`, jvar, `] = uint8(num)`)
			p.P(jvar, `++`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(jvar)
			p.P(`i += copy(dAtA[i:], dAtA`, numGen.Current(), `[:`, jvar, `])`)
		} else if repeated {
			p.P(`for _, num := range m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.callVarint("num")
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`m.`, fieldname)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`m.`, fieldname)
		} else {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`*m.`, fieldname)
		}
	case descriptor.FieldDescriptorProto_TYPE_FIXED64,
		descriptor.FieldDescriptorProto_TYPE_SFIXED64:
		if !p.unsafe {
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 8`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeFixed64("num")
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.encodeFixed64("num")
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.callFixed64("m." + fieldname)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed64("m." + fieldname)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed64("*m." + fieldname)
			}
		} else {
			typeName := "int64"
			if *field.Type == descriptor.FieldDescriptorProto_TYPE_FIXED64 {
				typeName = "uint64"
			}
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 8`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.unsafeFixed64("num", typeName)
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64("num", typeName)
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64("m."+fieldname, typeName)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64("m."+fieldname, typeName)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed64("*m."+fieldname, typeName)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_FIXED32,
		descriptor.FieldDescriptorProto_TYPE_SFIXED32:
		if !p.unsafe {
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 4`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeFixed32("num")
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.encodeFixed32("num")
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.callFixed32("m." + fieldname)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed32("m." + fieldname)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.callFixed32("*m." + fieldname)
			}
		} else {
			typeName := "int32"
			if *field.Type == descriptor.FieldDescriptorProto_TYPE_FIXED32 {
				typeName = "uint32"
			}
			if packed {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `) * 4`)
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.unsafeFixed32("num", typeName)
				p.Out()
				p.P(`}`)
			} else if repeated {
				p.P(`for _, num := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32("num", typeName)
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if m.`, fieldname, ` != 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32("m."+fieldname, typeName)
				p.Out()
				p.P(`}`)
			} else if !nullable {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32("m."+fieldname, typeName)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.unsafeFixed32("*m."+fieldname, typeName)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_BOOL:
		if packed {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`len(m.`, fieldname, `)`)
			p.P(`for _, b := range m.`, fieldname, ` {`)
			p.In()
			p.P(`if b {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.P(`i++`)
			p.Out()
			p.P(`}`)
		} else if repeated {
			p.P(`for _, b := range m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.P(`if b {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.P(`i++`)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.P(`if m.`, fieldname, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.P(`i++`)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.encodeKey(fieldNumber, wireType)
			p.P(`if m.`, fieldname, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.P(`i++`)
		} else {
			p.encodeKey(fieldNumber, wireType)
			p.P(`if *m.`, fieldname, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.P(`i++`)
		}
	case descriptor.FieldDescriptorProto_TYPE_STRING:
		if repeated {
			p.P(`for _, s := range m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.P(`l = len(s)`)
			p.encodeVarint("l")
			p.P(`i+=copy(dAtA[i:], s)`)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if len(m.`, fieldname, `) > 0 {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`len(m.`, fieldname, `)`)
			p.P(`i+=copy(dAtA[i:], m.`, fieldname, `)`)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`len(m.`, fieldname, `)`)
			p.P(`i+=copy(dAtA[i:], m.`, fieldname, `)`)
		} else {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`len(*m.`, fieldname, `)`)
			p.P(`i+=copy(dAtA[i:], *m.`, fieldname, `)`)
		}
	case descriptor.FieldDescriptorProto_TYPE_GROUP:
		panic(fmt.Errorf("marshaler does not support group %v", fieldname))
	case descriptor.FieldDescriptorProto_TYPE_MESSAGE:
		if p.IsMap(field) {
			m := p.GoMapType(nil, field)
			keygoTyp, keywire := p.GoType(nil, m.KeyField)
			keygoAliasTyp, _ := p.GoType(nil, m.KeyAliasField)
			// keys may not be pointers
			keygoTyp = strings.Replace(keygoTyp, "*", "", 1)
			keygoAliasTyp = strings.Replace(keygoAliasTyp, "*", "", 1)
			keyCapTyp := generator.CamelCase(keygoTyp)
			valuegoTyp, valuewire := p.GoType(nil, m.ValueField)
			valuegoAliasTyp, _ := p.GoType(nil, m.ValueAliasField)
			nullable, valuegoTyp, valuegoAliasTyp = generator.GoMapValueTypes(field, m.ValueField, valuegoTyp, valuegoAliasTyp)
			keyKeySize := keySize(1, wireToType(keywire))
			valueKeySize := keySize(2, wireToType(valuewire))
			if gogoproto.IsStableMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				keysName := `keysFor` + fieldname
				p.P(keysName, ` := make([]`, keygoTyp, `, 0, len(m.`, fieldname, `))`)
				p.P(`for k, _ := range m.`, fieldname, ` {`)
				p.In()
				p.P(keysName, ` = append(`, keysName, `, `, keygoTyp, `(k))`)
				p.Out()
				p.P(`}`)
				p.P(p.sortKeysPkg.Use(), `.`, keyCapTyp, `s(`, keysName, `)`)
				p.P(`for _, k := range `, keysName, ` {`)
			} else {
				p.P(`for k, _ := range m.`, fieldname, ` {`)
			}
			p.In()
			p.encodeKey(fieldNumber, wireType)
			sum := []string{strconv.Itoa(keyKeySize)}
			switch m.KeyField.GetType() {
			case descriptor.FieldDescriptorProto_TYPE_DOUBLE,
				descriptor.FieldDescriptorProto_TYPE_FIXED64,
				descriptor.FieldDescriptorProto_TYPE_SFIXED64:
				sum = append(sum, `8`)
			case descriptor.FieldDescriptorProto_TYPE_FLOAT,
				descriptor.FieldDescriptorProto_TYPE_FIXED32,
				descriptor.FieldDescriptorProto_TYPE_SFIXED32:
				sum = append(sum, `4`)
			case descriptor.FieldDescriptorProto_TYPE_INT64,
				descriptor.FieldDescriptorProto_TYPE_UINT64,
				descriptor.FieldDescriptorProto_TYPE_UINT32,
				descriptor.FieldDescriptorProto_TYPE_ENUM,
				descriptor.FieldDescriptorProto_TYPE_INT32:
				sum = append(sum, `sov`+p.localName+`(uint64(k))`)
			case descriptor.FieldDescriptorProto_TYPE_BOOL:
				sum = append(sum, `1`)
			case descriptor.FieldDescriptorProto_TYPE_STRING,
				descriptor.FieldDescriptorProto_TYPE_BYTES:
				sum = append(sum, `len(k)+sov`+p.localName+`(uint64(len(k)))`)
			case descriptor.FieldDescriptorProto_TYPE_SINT32,
				descriptor.FieldDescriptorProto_TYPE_SINT64:
				sum = append(sum, `soz`+p.localName+`(uint64(k))`)
			}
			if gogoproto.IsStableMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				p.P(`v := m.`, fieldname, `[`, keygoAliasTyp, `(k)]`)
			} else {
				p.P(`v := m.`, fieldname, `[k]`)
			}
			accessor := `v`
			switch m.ValueField.GetType() {
			case descriptor.FieldDescriptorProto_TYPE_DOUBLE,
				descriptor.FieldDescriptorProto_TYPE_FIXED64,
				descriptor.FieldDescriptorProto_TYPE_SFIXED64:
				sum = append(sum, strconv.Itoa(valueKeySize))
				sum = append(sum, strconv.Itoa(8))
			case descriptor.FieldDescriptorProto_TYPE_FLOAT,
				descriptor.FieldDescriptorProto_TYPE_FIXED32,
				descriptor.FieldDescriptorProto_TYPE_SFIXED32:
				sum = append(sum, strconv.Itoa(valueKeySize))
				sum = append(sum, strconv.Itoa(4))
			case descriptor.FieldDescriptorProto_TYPE_INT64,
				descriptor.FieldDescriptorProto_TYPE_UINT64,
				descriptor.FieldDescriptorProto_TYPE_UINT32,
				descriptor.FieldDescriptorProto_TYPE_ENUM,
				descriptor.FieldDescriptorProto_TYPE_INT32:
				sum = append(sum, strconv.Itoa(valueKeySize))
				sum = append(sum, `sov`+p.localName+`(uint64(v))`)
			case descriptor.FieldDescriptorProto_TYPE_BOOL:
				sum = append(sum, strconv.Itoa(valueKeySize))
				sum = append(sum, `1`)
			case descriptor.FieldDescriptorProto_TYPE_STRING:
				sum = append(sum, strconv.Itoa(valueKeySize))
				sum = append(sum, `len(v)+sov`+p.localName+`(uint64(len(v)))`)
			case descriptor.FieldDescriptorProto_TYPE_BYTES:
				if gogoproto.IsCustomType(field) {
					p.P(`cSize := 0`)
					if gogoproto.IsNullable(field) {
						p.P(`if `, accessor, ` != nil {`)
						p.In()
					}
					p.P(`cSize = `, accessor, `.Size()`)
					p.P(`cSize += `, strconv.Itoa(valueKeySize), ` + sov`+p.localName+`(uint64(cSize))`)
					if gogoproto.IsNullable(field) {
						p.Out()
						p.P(`}`)
					}
					sum = append(sum, `cSize`)
				} else {
					p.P(`byteSize := 0`)
					if proto3 {
						p.P(`if len(v) > 0 {`)
					} else {
						p.P(`if v != nil {`)
					}
					p.In()
					p.P(`byteSize = `, strconv.Itoa(valueKeySize), ` + len(v)+sov`+p.localName+`(uint64(len(v)))`)
					p.Out()
					p.P(`}`)
					sum = append(sum, `byteSize`)
				}
			case descriptor.FieldDescriptorProto_TYPE_SINT32,
				descriptor.FieldDescriptorProto_TYPE_SINT64:
				sum = append(sum, strconv.Itoa(valueKeySize))
				sum = append(sum, `soz`+p.localName+`(uint64(v))`)
			case descriptor.FieldDescriptorProto_TYPE_MESSAGE:
				if valuegoTyp != valuegoAliasTyp &&
					!gogoproto.IsStdTime(field) &&
					!gogoproto.IsStdDuration(field) {
					if nullable {
						// cast back to the type that has the generated methods on it
						accessor = `((` + valuegoTyp + `)(` + accessor + `))`
					} else {
						accessor = `((*` + valuegoTyp + `)(&` + accessor + `))`
					}
				} else if !nullable {
					accessor = `(&v)`
				}
				p.P(`msgSize := 0`)
				p.P(`if `, accessor, ` != nil {`)
				p.In()
				if gogoproto.IsStdTime(field) {
					p.P(`msgSize = `, p.typesPkg.Use(), `.SizeOfStdTime(*`, accessor, `)`)
				} else if gogoproto.IsStdDuration(field) {
					p.P(`msgSize = `, p.typesPkg.Use(), `.SizeOfStdDuration(*`, accessor, `)`)
				} else if protoSizer {
					p.P(`msgSize = `, accessor, `.ProtoSize()`)
				} else {
					p.P(`msgSize = `, accessor, `.Size()`)
				}
				p.P(`msgSize += `, strconv.Itoa(valueKeySize), ` + sov`+p.localName+`(uint64(msgSize))`)
				p.Out()
				p.P(`}`)
				sum = append(sum, `msgSize`)
			}
			p.P(`mapSize := `, strings.Join(sum, " + "))
			p.callVarint("mapSize")
			p.encodeKey(1, wireToType(keywire))
			p.mapField(numGen, field, m.KeyField, "k", protoSizer)
			nullableMsg := nullable && (m.ValueField.GetType() == descriptor.FieldDescriptorProto_TYPE_MESSAGE ||
				gogoproto.IsCustomType(field) && m.ValueField.IsBytes())
			plainBytes := m.ValueField.IsBytes() && !gogoproto.IsCustomType(field)
			if nullableMsg {
				p.P(`if `, accessor, ` != nil { `)
				p.In()
			} else if plainBytes {
				if proto3 {
					p.P(`if len(`, accessor, `) > 0 {`)
				} else {
					p.P(`if `, accessor, ` != nil {`)
				}
				p.In()
			}
			p.encodeKey(2, wireToType(valuewire))
			p.mapField(numGen, field, m.ValueField, accessor, protoSizer)
			if nullableMsg || plainBytes {
				p.Out()
				p.P(`}`)
			}
			p.Out()
			p.P(`}`)
		} else if repeated {
			p.P(`for _, msg := range m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			varName := "msg"
			if gogoproto.IsStdTime(field) {
				if gogoproto.IsNullable(field) {
					varName = "*" + varName
				}
				p.callVarint(p.typesPkg.Use(), `.SizeOfStdTime(`, varName, `)`)
				p.P(`n, err := `, p.typesPkg.Use(), `.StdTimeMarshalTo(`, varName, `, dAtA[i:])`)
			} else if gogoproto.IsStdDuration(field) {
				if gogoproto.IsNullable(field) {
					varName = "*" + varName
				}
				p.callVarint(p.typesPkg.Use(), `.SizeOfStdDuration(`, varName, `)`)
				p.P(`n, err := `, p.typesPkg.Use(), `.StdDurationMarshalTo(`, varName, `, dAtA[i:])`)
			} else if protoSizer {
				p.callVarint(varName, ".ProtoSize()")
				p.P(`n, err := `, varName, `.MarshalTo(dAtA[i:])`)
			} else {
				p.callVarint(varName, ".Size()")
				p.P(`n, err := `, varName, `.MarshalTo(dAtA[i:])`)
			}
			p.P(`if err != nil {`)
			p.In()
			p.P(`return 0, err`)
			p.Out()
			p.P(`}`)
			p.P(`i+=n`)
			p.Out()
			p.P(`}`)
		} else {
			p.encodeKey(fieldNumber, wireType)
			varName := `m.` + fieldname
			if gogoproto.IsStdTime(field) {
				if gogoproto.IsNullable(field) {
					varName = "*" + varName
				}
				p.callVarint(p.typesPkg.Use(), `.SizeOfStdTime(`, varName, `)`)
				p.P(`n`, numGen.Next(), `, err := `, p.typesPkg.Use(), `.StdTimeMarshalTo(`, varName, `, dAtA[i:])`)
			} else if gogoproto.IsStdDuration(field) {
				if gogoproto.IsNullable(field) {
					varName = "*" + varName
				}
				p.callVarint(p.typesPkg.Use(), `.SizeOfStdDuration(`, varName, `)`)
				p.P(`n`, numGen.Next(), `, err := `, p.typesPkg.Use(), `.StdDurationMarshalTo(`, varName, `, dAtA[i:])`)
			} else if protoSizer {
				p.callVarint(varName, `.ProtoSize()`)
				p.P(`n`, numGen.Next(), `, err := `, varName, `.MarshalTo(dAtA[i:])`)
			} else {
				p.callVarint(varName, `.Size()`)
				p.P(`n`, numGen.Next(), `, err := `, varName, `.MarshalTo(dAtA[i:])`)
			}
			p.P(`if err != nil {`)
			p.In()
			p.P(`return 0, err`)
			p.Out()
			p.P(`}`)
			p.P(`i+=n`, numGen.Current())
		}
	case descriptor.FieldDescriptorProto_TYPE_BYTES:
		if !gogoproto.IsCustomType(field) {
			if repeated {
				p.P(`for _, b := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.callVarint("len(b)")
				p.P(`i+=copy(dAtA[i:], b)`)
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if len(m.`, fieldname, `) > 0 {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `)`)
				p.P(`i+=copy(dAtA[i:], m.`, fieldname, `)`)
				p.Out()
				p.P(`}`)
			} else {
				p.encodeKey(fieldNumber, wireType)
				p.callVarint(`len(m.`, fieldname, `)`)
				p.P(`i+=copy(dAtA[i:], m.`, fieldname, `)`)
			}
		} else {
			if repeated {
				p.P(`for _, msg := range m.`, fieldname, ` {`)
				p.In()
				p.encodeKey(fieldNumber, wireType)
				if protoSizer {
					p.callVarint(`msg.ProtoSize()`)
				} else {
					p.callVarint(`msg.Size()`)
				}
				p.P(`n, err := msg.MarshalTo(dAtA[i:])`)
				p.P(`if err != nil {`)
				p.In()
				p.P(`return 0, err`)
				p.Out()
				p.P(`}`)
				p.P(`i+=n`)
				p.Out()
				p.P(`}`)
			} else {
				p.encodeKey(fieldNumber, wireType)
				if protoSizer {
					p.callVarint(`m.`, fieldname, `.ProtoSize()`)
				} else {
					p.callVarint(`m.`, fieldname, `.Size()`)
				}
				p.P(`n`, numGen.Next(), `, err := m.`, fieldname, `.MarshalTo(dAtA[i:])`)
				p.P(`if err != nil {`)
				p.In()
				p.P(`return 0, err`)
				p.Out()
				p.P(`}`)
				p.P(`i+=n`, numGen.Current())
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_SINT32:
		if packed {
			datavar := "dAtA" + numGen.Next()
			jvar := "j" + numGen.Next()
			p.P(datavar, ` := make([]byte, len(m.`, fieldname, ")*5)")
			p.P(`var `, jvar, ` int`)
			p.P(`for _, num := range m.`, fieldname, ` {`)
			p.In()
			xvar := "x" + numGen.Next()
			p.P(xvar, ` := (uint32(num) << 1) ^ uint32((num >> 31))`)
			p.P(`for `, xvar, ` >= 1<<7 {`)
			p.In()
			p.P(datavar, `[`, jvar, `] = uint8(uint64(`, xvar, `)&0x7f|0x80)`)
			p.P(jvar, `++`)
			p.P(xvar, ` >>= 7`)
			p.Out()
			p.P(`}`)
			p.P(datavar, `[`, jvar, `] = uint8(`, xvar, `)`)
			p.P(jvar, `++`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(jvar)
			p.P(`i+=copy(dAtA[i:], `, datavar, `[:`, jvar, `])`)
		} else if repeated {
			p.P(`for _, num := range m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.P(`x`, numGen.Next(), ` := (uint32(num) << 1) ^ uint32((num >> 31))`)
			p.encodeVarint("x" + numGen.Current())
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`(uint32(m.`, fieldname, `) << 1) ^ uint32((m.`, fieldname, ` >> 31))`)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`(uint32(m.`, fieldname, `) << 1) ^ uint32((m.`, fieldname, ` >> 31))`)
		} else {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`(uint32(*m.`, fieldname, `) << 1) ^ uint32((*m.`, fieldname, ` >> 31))`)
		}
	case descriptor.FieldDescriptorProto_TYPE_SINT64:
		if packed {
			jvar := "j" + numGen.Next()
			xvar := "x" + numGen.Next()
			datavar := "dAtA" + numGen.Next()
			p.P(`var `, jvar, ` int`)
			p.P(datavar, ` := make([]byte, len(m.`, fieldname, `)*10)`)
			p.P(`for _, num := range m.`, fieldname, ` {`)
			p.In()
			p.P(xvar, ` := (uint64(num) << 1) ^ uint64((num >> 63))`)
			p.P(`for `, xvar, ` >= 1<<7 {`)
			p.In()
			p.P(datavar, `[`, jvar, `] = uint8(uint64(`, xvar, `)&0x7f|0x80)`)
			p.P(jvar, `++`)
			p.P(xvar, ` >>= 7`)
			p.Out()
			p.P(`}`)
			p.P(datavar, `[`, jvar, `] = uint8(`, xvar, `)`)
			p.P(jvar, `++`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(jvar)
			p.P(`i+=copy(dAtA[i:], `, datavar, `[:`, jvar, `])`)
		} else if repeated {
			p.P(`for _, num := range m.`, fieldname, ` {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.P(`x`, numGen.Next(), ` := (uint64(num) << 1) ^ uint64((num >> 63))`)
			p.encodeVarint("x" + numGen.Current())
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`(uint64(m.`, fieldname, `) << 1) ^ uint64((m.`, fieldname, ` >> 63))`)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`(uint64(m.`, fieldname, `) << 1) ^ uint64((m.`, fieldname, ` >> 63))`)
		} else {
			p.encodeKey(fieldNumber, wireType)
			p.callVarint(`(uint64(*m.`, fieldname, `) << 1) ^ uint64((*m.`, fieldname, ` >> 63))`)
		}
	default:
		panic("not implemented")
	}
	if (required && nullable) || repeated || doNilCheck {
		p.Out()
		p.P(`}`)
	}
}

func (p *marshalto) Generate(file *generator.FileDescriptor) {
	numGen := NewNumGen()
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.atleastOne = false
	p.localName = generator.FileName(file)

	p.mathPkg = p.NewImport("math")
	p.sortKeysPkg = p.NewImport("github.com/gogo/protobuf/sortkeys")
	p.protoPkg = p.NewImport("github.com/gogo/protobuf/proto")
	if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
		p.protoPkg = p.NewImport("github.com/golang/protobuf/proto")
	}
	p.unsafePkg = p.NewImport("unsafe")
	p.errorsPkg = p.NewImport("errors")
	p.typesPkg = p.NewImport("github.com/gogo/protobuf/types")

	for _, message := range file.Messages() {
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		if p.unsafe {
			if !gogoproto.IsUnsafeMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				continue
			}
			if gogoproto.IsMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				panic(fmt.Sprintf("unsafe_marshaler and marshalto enabled for %v", ccTypeName))
			}
		}
		if !p.unsafe {
			if !gogoproto.IsMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				continue
			}
			if gogoproto.IsUnsafeMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				panic(fmt.Sprintf("unsafe_marshaler and marshalto enabled for %v", ccTypeName))
			}
		}
		p.atleastOne = true

		p.P(`func (m *`, ccTypeName, `) Marshal() (dAtA []byte, err error) {`)
		p.In()
		if gogoproto.IsProtoSizer(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`size := m.ProtoSize()`)
		} else {
			p.P(`size := m.Size()`)
		}
		p.P(`dAtA = make([]byte, size)`)
		p.P(`n, err := m.MarshalTo(dAtA)`)
		p.P(`if err != nil {`)
		p.In()
		p.P(`return nil, err`)
		p.Out()
		p.P(`}`)
		p.P(`return dAtA[:n], nil`)
		p.Out()
		p.P(`}`)
		p.P(``)
		p.P(`func (m *`, ccTypeName, `) MarshalTo(dAtA []byte) (int, error) {`)
		p.In()
		p.P(`var i int`)
		p.P(`_ = i`)
		p.P(`var l int`)
		p.P(`_ = l`)
		fields := orderFields(message.GetField())
		sort.Sort(fields)
		oneofs := make(map[string]struct{})
		for _, field := range message.Field {
			oneof := field.OneofIndex != nil
			if !oneof {
				proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
				p.generateField(proto3, numGen, file, message, field)
			} else {
				fieldname := p.GetFieldName(message, field)
				if _, ok := oneofs[fieldname]; !ok {
					oneofs[fieldname] = struct{}{}
					p.P(`if m.`, fieldname, ` != nil {`)
					p.In()
					p.P(`nn`, numGen.Next(), `, err := m.`, fieldname, `.MarshalTo(dAtA[i:])`)
					p.P(`if err != nil {`)
					p.In()
					p.P(`return 0, err`)
					p.Out()
					p.P(`}`)
					p.P(`i+=nn`, numGen.Current())
					p.Out()
					p.P(`}`)
				}
			}
		}
		if message.DescriptorProto.HasExtension() {
			if gogoproto.HasExtensionsMap(file.FileDescriptorProto, message.DescriptorProto) {
				p.P(`n, err := `, p.protoPkg.Use(), `.EncodeInternalExtension(m, dAtA[i:])`)
				p.P(`if err != nil {`)
				p.In()
				p.P(`return 0, err`)
				p.Out()
				p.P(`}`)
				p.P(`i+=n`)
			} else {
				p.P(`if m.XXX_extensions != nil {`)
				p.In()
				p.P(`i+=copy(dAtA[i:], m.XXX_extensions)`)
				p.Out()
				p.P(`}`)
			}
		}
		if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`if m.XXX_unrecognized != nil {`)
			p.In()
			p.P(`i+=copy(dAtA[i:], m.XXX_unrecognized)`)
			p.Out()
			p.P(`}`)
		}

		p.P(`return i, nil`)
		p.Out()
		p.P(`}`)
		p.P()

		//Generate MarshalTo methods for oneof fields
		m := proto.Clone(message.DescriptorProto).(*descriptor.DescriptorProto)
		for _, field := range m.Field {
			oneof := field.OneofIndex != nil
			if !oneof {
				continue
			}
			ccTypeName := p.OneOfTypeName(message, field)
			p.P(`func (m *`, ccTypeName, `) MarshalTo(dAtA []byte) (int, error) {`)
			p.In()
			p.P(`i := 0`)
			vanity.TurnOffNullableForNativeTypesWithoutDefaultsOnly(field)
			p.generateField(false, numGen, file, message, field)
			p.P(`return i, nil`)
			p.Out()
			p.P(`}`)
		}
	}

	if p.atleastOne {
		p.P(`func encodeFixed64`, p.localName, `(dAtA []byte, offset int, v uint64) int {`)
		p.In()
		p.P(`dAtA[offset] = uint8(v)`)
		p.P(`dAtA[offset+1] = uint8(v >> 8)`)
		p.P(`dAtA[offset+2] = uint8(v >> 16)`)
		p.P(`dAtA[offset+3] = uint8(v >> 24)`)
		p.P(`dAtA[offset+4] = uint8(v >> 32)`)
		p.P(`dAtA[offset+5] = uint8(v >> 40)`)
		p.P(`dAtA[offset+6] = uint8(v >> 48)`)
		p.P(`dAtA[offset+7] = uint8(v >> 56)`)
		p.P(`return offset+8`)
		p.Out()
		p.P(`}`)

		p.P(`func encodeFixed32`, p.localName, `(dAtA []byte, offset int, v uint32) int {`)
		p.In()
		p.P(`dAtA[offset] = uint8(v)`)
		p.P(`dAtA[offset+1] = uint8(v >> 8)`)
		p.P(`dAtA[offset+2] = uint8(v >> 16)`)
		p.P(`dAtA[offset+3] = uint8(v >> 24)`)
		p.P(`return offset+4`)
		p.Out()
		p.P(`}`)

		p.P(`func encodeVarint`, p.localName, `(dAtA []byte, offset int, v uint64) int {`)
		p.In()
		p.P(`for v >= 1<<7 {`)
		p.In()
		p.P(`dAtA[offset] = uint8(v&0x7f|0x80)`)
		p.P(`v >>= 7`)
		p.P(`offset++`)
		p.Out()
		p.P(`}`)
		p.P(`dAtA[offset] = uint8(v)`)
		p.P(`return offset+1`)
		p.Out()
		p.P(`}`)
	}

}

func init() {
	generator.RegisterPlugin(NewMarshal())
	generator.RegisterPlugin(NewUnsafeMarshal())
}
