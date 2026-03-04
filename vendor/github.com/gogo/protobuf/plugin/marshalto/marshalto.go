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
          n, err := m.MarshalToSizedBuffer(dAtA[:size])
          if err != nil {
                  return nil, err
          }
          return dAtA[:n], nil
  }

  func (m *B) MarshalTo(dAtA []byte) (int, error) {
          size := m.Size()
          return m.MarshalToSizedBuffer(dAtA[:size])
  }

  func (m *B) MarshalToSizedBuffer(dAtA []byte) (int, error) {
          i := len(dAtA)
          _ = i
          var l int
          _ = l
          if m.XXX_unrecognized != nil {
                  i -= len(m.XXX_unrecognized)
                  copy(dAtA[i:], m.XXX_unrecognized)
          }
          if len(m.G) > 0 {
                  for iNdEx := len(m.G) - 1; iNdEx >= 0; iNdEx-- {
                          {
                                  size := m.G[iNdEx].Size()
                                  i -= size
                                  if _, err := m.G[iNdEx].MarshalTo(dAtA[i:]); err != nil {
                                          return 0, err
                                  }
                                  i = encodeVarintExample(dAtA, i, uint64(size))
                          }
                          i--
                          dAtA[i] = 0x12
                  }
          }
          {
                  size, err := m.A.MarshalToSizedBuffer(dAtA[:i])
                  if err != nil {
                          return 0, err
                  }
                  i -= size
                  i = encodeVarintExample(dAtA, i, uint64(size))
          }
          i--
          dAtA[i] = 0xa
          return len(dAtA) - i, nil
  }

As shown above Marshal calculates the size of the not yet marshalled message
and allocates the appropriate buffer.
This is followed by calling the MarshalToSizedBuffer method which requires a preallocated buffer, and marshals backwards.
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
	errorsPkg   generator.Single
	protoPkg    generator.Single
	sortKeysPkg generator.Single
	mathPkg     generator.Single
	typesPkg    generator.Single
	binaryPkg   generator.Single
	localName   string
}

func NewMarshal() *marshalto {
	return &marshalto{}
}

func (p *marshalto) Name() string {
	return "marshalto"
}

func (p *marshalto) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *marshalto) callFixed64(varName ...string) {
	p.P(`i -= 8`)
	p.P(p.binaryPkg.Use(), `.LittleEndian.PutUint64(dAtA[i:], uint64(`, strings.Join(varName, ""), `))`)
}

func (p *marshalto) callFixed32(varName ...string) {
	p.P(`i -= 4`)
	p.P(p.binaryPkg.Use(), `.LittleEndian.PutUint32(dAtA[i:], uint32(`, strings.Join(varName, ""), `))`)
}

func (p *marshalto) callVarint(varName ...string) {
	p.P(`i = encodeVarint`, p.localName, `(dAtA, i, uint64(`, strings.Join(varName, ""), `))`)
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
	for i = len(keybuf) - 1; i >= 0; i-- {
		p.P(`i--`)
		p.P(`dAtA[i] = `, fmt.Sprintf("%#v", keybuf[i]))
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
		p.P(`i--`)
		p.P(`if `, varName, ` {`)
		p.In()
		p.P(`dAtA[i] = 1`)
		p.Out()
		p.P(`} else {`)
		p.In()
		p.P(`dAtA[i] = 0`)
		p.Out()
		p.P(`}`)
	case descriptor.FieldDescriptorProto_TYPE_STRING,
		descriptor.FieldDescriptorProto_TYPE_BYTES:
		if gogoproto.IsCustomType(field) && kvField.IsBytes() {
			p.forward(varName, true, protoSizer)
		} else {
			p.P(`i -= len(`, varName, `)`)
			p.P(`copy(dAtA[i:], `, varName, `)`)
			p.callVarint(`len(`, varName, `)`)
		}
	case descriptor.FieldDescriptorProto_TYPE_SINT32:
		p.callVarint(`(uint32(`, varName, `) << 1) ^ uint32((`, varName, ` >> 31))`)
	case descriptor.FieldDescriptorProto_TYPE_SINT64:
		p.callVarint(`(uint64(`, varName, `) << 1) ^ uint64((`, varName, ` >> 63))`)
	case descriptor.FieldDescriptorProto_TYPE_MESSAGE:
		if !p.marshalAllSizeOf(kvField, `(*`+varName+`)`, numGen.Next()) {
			if gogoproto.IsCustomType(field) {
				p.forward(varName, true, protoSizer)
			} else {
				p.backward(varName, true)
			}
		}

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
		if packed {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float64bits(float64(`, val, `))`)
			p.callFixed64("f" + numGen.Current())
			p.Out()
			p.P(`}`)
			p.callVarint(`len(m.`, fieldname, `) * 8`)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float64bits(float64(`, val, `))`)
			p.callFixed64("f" + numGen.Current())
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(m.`+fieldname, `))`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(m.`+fieldname, `))`)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callFixed64(p.mathPkg.Use(), `.Float64bits(float64(*m.`+fieldname, `))`)
			p.encodeKey(fieldNumber, wireType)
		}
	case descriptor.FieldDescriptorProto_TYPE_FLOAT:
		if packed {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float32bits(float32(`, val, `))`)
			p.callFixed32("f" + numGen.Current())
			p.Out()
			p.P(`}`)
			p.callVarint(`len(m.`, fieldname, `) * 4`)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`f`, numGen.Next(), ` := `, p.mathPkg.Use(), `.Float32bits(float32(`, val, `))`)
			p.callFixed32("f" + numGen.Current())
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(m.`+fieldname, `))`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(m.`+fieldname, `))`)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callFixed32(p.mathPkg.Use(), `.Float32bits(float32(*m.`+fieldname, `))`)
			p.encodeKey(fieldNumber, wireType)
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
			p.P(`i -= `, jvar)
			p.P(`copy(dAtA[i:], dAtA`, numGen.Current(), `[:`, jvar, `])`)
			p.callVarint(jvar)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.callVarint(val)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callVarint(`m.`, fieldname)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callVarint(`m.`, fieldname)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callVarint(`*m.`, fieldname)
			p.encodeKey(fieldNumber, wireType)
		}
	case descriptor.FieldDescriptorProto_TYPE_FIXED64,
		descriptor.FieldDescriptorProto_TYPE_SFIXED64:
		if packed {
			val := p.reverseListRange(`m.`, fieldname)
			p.callFixed64(val)
			p.Out()
			p.P(`}`)
			p.callVarint(`len(m.`, fieldname, `) * 8`)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.callFixed64(val)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callFixed64("m." + fieldname)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callFixed64("m." + fieldname)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callFixed64("*m." + fieldname)
			p.encodeKey(fieldNumber, wireType)
		}
	case descriptor.FieldDescriptorProto_TYPE_FIXED32,
		descriptor.FieldDescriptorProto_TYPE_SFIXED32:
		if packed {
			val := p.reverseListRange(`m.`, fieldname)
			p.callFixed32(val)
			p.Out()
			p.P(`}`)
			p.callVarint(`len(m.`, fieldname, `) * 4`)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.callFixed32(val)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callFixed32("m." + fieldname)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callFixed32("m." + fieldname)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callFixed32("*m." + fieldname)
			p.encodeKey(fieldNumber, wireType)
		}
	case descriptor.FieldDescriptorProto_TYPE_BOOL:
		if packed {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`i--`)
			p.P(`if `, val, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)
			p.callVarint(`len(m.`, fieldname, `)`)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`i--`)
			p.P(`if `, val, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` {`)
			p.In()
			p.P(`i--`)
			p.P(`if m.`, fieldname, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.P(`i--`)
			p.P(`if m.`, fieldname, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.P(`i--`)
			p.P(`if *m.`, fieldname, ` {`)
			p.In()
			p.P(`dAtA[i] = 1`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`dAtA[i] = 0`)
			p.Out()
			p.P(`}`)
			p.encodeKey(fieldNumber, wireType)
		}
	case descriptor.FieldDescriptorProto_TYPE_STRING:
		if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`i -= len(`, val, `)`)
			p.P(`copy(dAtA[i:], `, val, `)`)
			p.callVarint(`len(`, val, `)`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if len(m.`, fieldname, `) > 0 {`)
			p.In()
			p.P(`i -= len(m.`, fieldname, `)`)
			p.P(`copy(dAtA[i:], m.`, fieldname, `)`)
			p.callVarint(`len(m.`, fieldname, `)`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.P(`i -= len(m.`, fieldname, `)`)
			p.P(`copy(dAtA[i:], m.`, fieldname, `)`)
			p.callVarint(`len(m.`, fieldname, `)`)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.P(`i -= len(*m.`, fieldname, `)`)
			p.P(`copy(dAtA[i:], *m.`, fieldname, `)`)
			p.callVarint(`len(*m.`, fieldname, `)`)
			p.encodeKey(fieldNumber, wireType)
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
			var val string
			if gogoproto.IsStableMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				keysName := `keysFor` + fieldname
				p.P(keysName, ` := make([]`, keygoTyp, `, 0, len(m.`, fieldname, `))`)
				p.P(`for k := range m.`, fieldname, ` {`)
				p.In()
				p.P(keysName, ` = append(`, keysName, `, `, keygoTyp, `(k))`)
				p.Out()
				p.P(`}`)
				p.P(p.sortKeysPkg.Use(), `.`, keyCapTyp, `s(`, keysName, `)`)
				val = p.reverseListRange(keysName)
			} else {
				p.P(`for k := range m.`, fieldname, ` {`)
				val = "k"
				p.In()
			}
			if gogoproto.IsStableMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				p.P(`v := m.`, fieldname, `[`, keygoAliasTyp, `(`, val, `)]`)
			} else {
				p.P(`v := m.`, fieldname, `[`, val, `]`)
			}
			p.P(`baseI := i`)
			accessor := `v`

			if m.ValueField.GetType() == descriptor.FieldDescriptorProto_TYPE_MESSAGE {
				if valuegoTyp != valuegoAliasTyp && !gogoproto.IsStdType(m.ValueAliasField) {
					if nullable {
						// cast back to the type that has the generated methods on it
						accessor = `((` + valuegoTyp + `)(` + accessor + `))`
					} else {
						accessor = `((*` + valuegoTyp + `)(&` + accessor + `))`
					}
				} else if !nullable {
					accessor = `(&v)`
				}
			}

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
			p.mapField(numGen, field, m.ValueAliasField, accessor, protoSizer)
			p.encodeKey(2, wireToType(valuewire))
			if nullableMsg || plainBytes {
				p.Out()
				p.P(`}`)
			}

			p.mapField(numGen, field, m.KeyField, val, protoSizer)
			p.encodeKey(1, wireToType(keywire))

			p.callVarint(`baseI - i`)

			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			sizeOfVarName := val
			if gogoproto.IsNullable(field) {
				sizeOfVarName = `*` + val
			}
			if !p.marshalAllSizeOf(field, sizeOfVarName, ``) {
				if gogoproto.IsCustomType(field) {
					p.forward(val, true, protoSizer)
				} else {
					p.backward(val, true)
				}
			}
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else {
			sizeOfVarName := `m.` + fieldname
			if gogoproto.IsNullable(field) {
				sizeOfVarName = `*` + sizeOfVarName
			}
			if !p.marshalAllSizeOf(field, sizeOfVarName, numGen.Next()) {
				if gogoproto.IsCustomType(field) {
					p.forward(`m.`+fieldname, true, protoSizer)
				} else {
					p.backward(`m.`+fieldname, true)
				}
			}
			p.encodeKey(fieldNumber, wireType)
		}
	case descriptor.FieldDescriptorProto_TYPE_BYTES:
		if !gogoproto.IsCustomType(field) {
			if repeated {
				val := p.reverseListRange(`m.`, fieldname)
				p.P(`i -= len(`, val, `)`)
				p.P(`copy(dAtA[i:], `, val, `)`)
				p.callVarint(`len(`, val, `)`)
				p.encodeKey(fieldNumber, wireType)
				p.Out()
				p.P(`}`)
			} else if proto3 {
				p.P(`if len(m.`, fieldname, `) > 0 {`)
				p.In()
				p.P(`i -= len(m.`, fieldname, `)`)
				p.P(`copy(dAtA[i:], m.`, fieldname, `)`)
				p.callVarint(`len(m.`, fieldname, `)`)
				p.encodeKey(fieldNumber, wireType)
				p.Out()
				p.P(`}`)
			} else {
				p.P(`i -= len(m.`, fieldname, `)`)
				p.P(`copy(dAtA[i:], m.`, fieldname, `)`)
				p.callVarint(`len(m.`, fieldname, `)`)
				p.encodeKey(fieldNumber, wireType)
			}
		} else {
			if repeated {
				val := p.reverseListRange(`m.`, fieldname)
				p.forward(val, true, protoSizer)
				p.encodeKey(fieldNumber, wireType)
				p.Out()
				p.P(`}`)
			} else {
				p.forward(`m.`+fieldname, true, protoSizer)
				p.encodeKey(fieldNumber, wireType)
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
			p.P(`i -= `, jvar)
			p.P(`copy(dAtA[i:], `, datavar, `[:`, jvar, `])`)
			p.callVarint(jvar)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`x`, numGen.Next(), ` := (uint32(`, val, `) << 1) ^ uint32((`, val, ` >> 31))`)
			p.callVarint(`x`, numGen.Current())
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callVarint(`(uint32(m.`, fieldname, `) << 1) ^ uint32((m.`, fieldname, ` >> 31))`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callVarint(`(uint32(m.`, fieldname, `) << 1) ^ uint32((m.`, fieldname, ` >> 31))`)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callVarint(`(uint32(*m.`, fieldname, `) << 1) ^ uint32((*m.`, fieldname, ` >> 31))`)
			p.encodeKey(fieldNumber, wireType)
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
			p.P(`i -= `, jvar)
			p.P(`copy(dAtA[i:], `, datavar, `[:`, jvar, `])`)
			p.callVarint(jvar)
			p.encodeKey(fieldNumber, wireType)
		} else if repeated {
			val := p.reverseListRange(`m.`, fieldname)
			p.P(`x`, numGen.Next(), ` := (uint64(`, val, `) << 1) ^ uint64((`, val, ` >> 63))`)
			p.callVarint("x" + numGen.Current())
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if proto3 {
			p.P(`if m.`, fieldname, ` != 0 {`)
			p.In()
			p.callVarint(`(uint64(m.`, fieldname, `) << 1) ^ uint64((m.`, fieldname, ` >> 63))`)
			p.encodeKey(fieldNumber, wireType)
			p.Out()
			p.P(`}`)
		} else if !nullable {
			p.callVarint(`(uint64(m.`, fieldname, `) << 1) ^ uint64((m.`, fieldname, ` >> 63))`)
			p.encodeKey(fieldNumber, wireType)
		} else {
			p.callVarint(`(uint64(*m.`, fieldname, `) << 1) ^ uint64((*m.`, fieldname, ` >> 63))`)
			p.encodeKey(fieldNumber, wireType)
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
	p.errorsPkg = p.NewImport("errors")
	p.binaryPkg = p.NewImport("encoding/binary")
	p.typesPkg = p.NewImport("github.com/gogo/protobuf/types")

	for _, message := range file.Messages() {
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		if !gogoproto.IsMarshaler(file.FileDescriptorProto, message.DescriptorProto) &&
			!gogoproto.IsUnsafeMarshaler(file.FileDescriptorProto, message.DescriptorProto) {
			continue
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
		p.P(`n, err := m.MarshalToSizedBuffer(dAtA[:size])`)
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
		if gogoproto.IsProtoSizer(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`size := m.ProtoSize()`)
		} else {
			p.P(`size := m.Size()`)
		}
		p.P(`return m.MarshalToSizedBuffer(dAtA[:size])`)
		p.Out()
		p.P(`}`)
		p.P(``)
		p.P(`func (m *`, ccTypeName, `) MarshalToSizedBuffer(dAtA []byte) (int, error) {`)
		p.In()
		p.P(`i := len(dAtA)`)
		p.P(`_ = i`)
		p.P(`var l int`)
		p.P(`_ = l`)
		if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`if m.XXX_unrecognized != nil {`)
			p.In()
			p.P(`i -= len(m.XXX_unrecognized)`)
			p.P(`copy(dAtA[i:], m.XXX_unrecognized)`)
			p.Out()
			p.P(`}`)
		}
		if message.DescriptorProto.HasExtension() {
			if gogoproto.HasExtensionsMap(file.FileDescriptorProto, message.DescriptorProto) {
				p.P(`if n, err := `, p.protoPkg.Use(), `.EncodeInternalExtensionBackwards(m, dAtA[:i]); err != nil {`)
				p.In()
				p.P(`return 0, err`)
				p.Out()
				p.P(`} else {`)
				p.In()
				p.P(`i -= n`)
				p.Out()
				p.P(`}`)
			} else {
				p.P(`if m.XXX_extensions != nil {`)
				p.In()
				p.P(`i -= len(m.XXX_extensions)`)
				p.P(`copy(dAtA[i:], m.XXX_extensions)`)
				p.Out()
				p.P(`}`)
			}
		}
		fields := orderFields(message.GetField())
		sort.Sort(fields)
		oneofs := make(map[string]struct{})
		for i := len(message.Field) - 1; i >= 0; i-- {
			field := message.Field[i]
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
					p.forward(`m.`+fieldname, false, gogoproto.IsProtoSizer(file.FileDescriptorProto, message.DescriptorProto))
					p.Out()
					p.P(`}`)
				}
			}
		}
		p.P(`return len(dAtA) - i, nil`)
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
			if gogoproto.IsProtoSizer(file.FileDescriptorProto, message.DescriptorProto) {
				p.P(`size := m.ProtoSize()`)
			} else {
				p.P(`size := m.Size()`)
			}
			p.P(`return m.MarshalToSizedBuffer(dAtA[:size])`)
			p.Out()
			p.P(`}`)
			p.P(``)
			p.P(`func (m *`, ccTypeName, `) MarshalToSizedBuffer(dAtA []byte) (int, error) {`)
			p.In()
			p.P(`i := len(dAtA)`)
			vanity.TurnOffNullableForNativeTypes(field)
			p.generateField(false, numGen, file, message, field)
			p.P(`return len(dAtA) - i, nil`)
			p.Out()
			p.P(`}`)
		}
	}

	if p.atleastOne {
		p.P(`func encodeVarint`, p.localName, `(dAtA []byte, offset int, v uint64) int {`)
		p.In()
		p.P(`offset -= sov`, p.localName, `(v)`)
		p.P(`base := offset`)
		p.P(`for v >= 1<<7 {`)
		p.In()
		p.P(`dAtA[offset] = uint8(v&0x7f|0x80)`)
		p.P(`v >>= 7`)
		p.P(`offset++`)
		p.Out()
		p.P(`}`)
		p.P(`dAtA[offset] = uint8(v)`)
		p.P(`return base`)
		p.Out()
		p.P(`}`)
	}

}

func (p *marshalto) reverseListRange(expression ...string) string {
	exp := strings.Join(expression, "")
	p.P(`for iNdEx := len(`, exp, `) - 1; iNdEx >= 0; iNdEx-- {`)
	p.In()
	return exp + `[iNdEx]`
}

func (p *marshalto) marshalAllSizeOf(field *descriptor.FieldDescriptorProto, varName, num string) bool {
	if gogoproto.IsStdTime(field) {
		p.marshalSizeOf(`StdTimeMarshalTo`, `SizeOfStdTime`, varName, num)
	} else if gogoproto.IsStdDuration(field) {
		p.marshalSizeOf(`StdDurationMarshalTo`, `SizeOfStdDuration`, varName, num)
	} else if gogoproto.IsStdDouble(field) {
		p.marshalSizeOf(`StdDoubleMarshalTo`, `SizeOfStdDouble`, varName, num)
	} else if gogoproto.IsStdFloat(field) {
		p.marshalSizeOf(`StdFloatMarshalTo`, `SizeOfStdFloat`, varName, num)
	} else if gogoproto.IsStdInt64(field) {
		p.marshalSizeOf(`StdInt64MarshalTo`, `SizeOfStdInt64`, varName, num)
	} else if gogoproto.IsStdUInt64(field) {
		p.marshalSizeOf(`StdUInt64MarshalTo`, `SizeOfStdUInt64`, varName, num)
	} else if gogoproto.IsStdInt32(field) {
		p.marshalSizeOf(`StdInt32MarshalTo`, `SizeOfStdInt32`, varName, num)
	} else if gogoproto.IsStdUInt32(field) {
		p.marshalSizeOf(`StdUInt32MarshalTo`, `SizeOfStdUInt32`, varName, num)
	} else if gogoproto.IsStdBool(field) {
		p.marshalSizeOf(`StdBoolMarshalTo`, `SizeOfStdBool`, varName, num)
	} else if gogoproto.IsStdString(field) {
		p.marshalSizeOf(`StdStringMarshalTo`, `SizeOfStdString`, varName, num)
	} else if gogoproto.IsStdBytes(field) {
		p.marshalSizeOf(`StdBytesMarshalTo`, `SizeOfStdBytes`, varName, num)
	} else {
		return false
	}
	return true
}

func (p *marshalto) marshalSizeOf(marshal, size, varName, num string) {
	p.P(`n`, num, `, err`, num, ` := `, p.typesPkg.Use(), `.`, marshal, `(`, varName, `, dAtA[i-`, p.typesPkg.Use(), `.`, size, `(`, varName, `):])`)
	p.P(`if err`, num, ` != nil {`)
	p.In()
	p.P(`return 0, err`, num)
	p.Out()
	p.P(`}`)
	p.P(`i -= n`, num)
	p.callVarint(`n`, num)
}

func (p *marshalto) backward(varName string, varInt bool) {
	p.P(`{`)
	p.In()
	p.P(`size, err := `, varName, `.MarshalToSizedBuffer(dAtA[:i])`)
	p.P(`if err != nil {`)
	p.In()
	p.P(`return 0, err`)
	p.Out()
	p.P(`}`)
	p.P(`i -= size`)
	if varInt {
		p.callVarint(`size`)
	}
	p.Out()
	p.P(`}`)
}

func (p *marshalto) forward(varName string, varInt, protoSizer bool) {
	p.P(`{`)
	p.In()
	if protoSizer {
		p.P(`size := `, varName, `.ProtoSize()`)
	} else {
		p.P(`size := `, varName, `.Size()`)
	}
	p.P(`i -= size`)
	p.P(`if _, err := `, varName, `.MarshalTo(dAtA[i:]); err != nil {`)
	p.In()
	p.P(`return 0, err`)
	p.Out()
	p.P(`}`)
	p.Out()
	if varInt {
		p.callVarint(`size`)
	}
	p.P(`}`)
}

func init() {
	generator.RegisterPlugin(NewMarshal())
}
