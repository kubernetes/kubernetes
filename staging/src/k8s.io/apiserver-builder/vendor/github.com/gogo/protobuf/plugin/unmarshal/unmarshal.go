// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
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
The unmarshal plugin generates a Unmarshal method for each message.
The `Unmarshal([]byte) error` method results in the fact that the message
implements the Unmarshaler interface.
The allows proto.Unmarshal to be faster by calling the generated Unmarshal method rather than using reflect.

If is enabled by the following extensions:

  - unmarshaler
  - unmarshaler_all

Or the following extensions:

  - unsafe_unmarshaler
  - unsafe_unmarshaler_all

That is if you want to use the unsafe package in your generated code.
The speed up using the unsafe package is not very significant.

The generation of unmarshalling tests are enabled using one of the following extensions:

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

  option (gogoproto.unmarshaler_all) = true;

  message B {
	option (gogoproto.description) = true;
	optional A A = 1 [(gogoproto.nullable) = false, (gogoproto.embed) = true];
	repeated bytes G = 2 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uint128", (gogoproto.nullable) = false];
  }

given to the unmarshal plugin, will generate the following code:

  func (m *B) Unmarshal(data []byte) error {
	l := len(data)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := data[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return proto.ErrWrongType
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if err := m.A.Unmarshal(data[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return proto.ErrWrongType
			}
			var byteLen int
			for shift := uint(0); ; shift += 7 {
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				byteLen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			postIndex := iNdEx + byteLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.G = append(m.G, github_com_gogo_protobuf_test_custom.Uint128{})
			if err := m.G[len(m.G)-1].Unmarshal(data[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			var sizeOfWire int
			for {
				sizeOfWire++
				wire >>= 7
				if wire == 0 {
					break
				}
			}
			iNdEx -= sizeOfWire
			skippy, err := skip(data[iNdEx:])
			if err != nil {
				return err
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			m.XXX_unrecognized = append(m.XXX_unrecognized, data[iNdEx:iNdEx+skippy]...)
			iNdEx += skippy
		}
	}
	return nil
  }

Remember when using this code to call proto.Unmarshal.
This will call m.Reset and invoke the generated Unmarshal method for you.
If you call m.Unmarshal without m.Reset you could be merging protocol buffers.

*/
package unmarshal

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type unmarshal struct {
	*generator.Generator
	unsafe bool
	generator.PluginImports
	atleastOne bool
	ioPkg      generator.Single
	mathPkg    generator.Single
	unsafePkg  generator.Single
	localName  string
}

func NewUnmarshal() *unmarshal {
	return &unmarshal{}
}

func NewUnsafeUnmarshal() *unmarshal {
	return &unmarshal{unsafe: true}
}

func (p *unmarshal) Name() string {
	if p.unsafe {
		return "unsafeunmarshaler"
	}
	return "unmarshal"
}

func (p *unmarshal) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *unmarshal) decodeVarint(varName string, typName string) {
	p.P(`for shift := uint(0); ; shift += 7 {`)
	p.In()
	p.P(`if shift >= 64 {`)
	p.In()
	p.P(`return ErrIntOverflow` + p.localName)
	p.Out()
	p.P(`}`)
	p.P(`if iNdEx >= l {`)
	p.In()
	p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
	p.Out()
	p.P(`}`)
	p.P(`b := data[iNdEx]`)
	p.P(`iNdEx++`)
	p.P(varName, ` |= (`, typName, `(b) & 0x7F) << shift`)
	p.P(`if b < 0x80 {`)
	p.In()
	p.P(`break`)
	p.Out()
	p.P(`}`)
	p.Out()
	p.P(`}`)
}

func (p *unmarshal) decodeFixed32(varName string, typeName string) {
	p.P(`if (iNdEx+4) > l {`)
	p.In()
	p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
	p.Out()
	p.P(`}`)
	p.P(`iNdEx += 4`)
	p.P(varName, ` = `, typeName, `(data[iNdEx-4])`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-3]) << 8`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-2]) << 16`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-1]) << 24`)
}

func (p *unmarshal) unsafeFixed32(varName string, typeName string) {
	p.P(`if iNdEx + 4 > l {`)
	p.In()
	p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
	p.Out()
	p.P(`}`)
	p.P(varName, ` = *(*`, typeName, `)(`, p.unsafePkg.Use(), `.Pointer(&data[iNdEx]))`)
	p.P(`iNdEx += 4`)
}

func (p *unmarshal) decodeFixed64(varName string, typeName string) {
	p.P(`if (iNdEx+8) > l {`)
	p.In()
	p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
	p.Out()
	p.P(`}`)
	p.P(`iNdEx += 8`)
	p.P(varName, ` = `, typeName, `(data[iNdEx-8])`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-7]) << 8`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-6]) << 16`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-5]) << 24`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-4]) << 32`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-3]) << 40`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-2]) << 48`)
	p.P(varName, ` |= `, typeName, `(data[iNdEx-1]) << 56`)
}

func (p *unmarshal) unsafeFixed64(varName string, typeName string) {
	p.P(`if iNdEx + 8 > l {`)
	p.In()
	p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
	p.Out()
	p.P(`}`)
	p.P(varName, ` = *(*`, typeName, `)(`, p.unsafePkg.Use(), `.Pointer(&data[iNdEx]))`)
	p.P(`iNdEx += 8`)
}

func (p *unmarshal) mapField(varName string, field *descriptor.FieldDescriptorProto) {
	switch field.GetType() {
	case descriptor.FieldDescriptorProto_TYPE_DOUBLE:
		p.P(`var `, varName, `temp uint64`)
		p.decodeFixed64(varName+"temp", "uint64")
		p.P(varName, ` := `, p.mathPkg.Use(), `.Float64frombits(`, varName, `temp)`)
	case descriptor.FieldDescriptorProto_TYPE_FLOAT:
		p.P(`var `, varName, `temp uint32`)
		p.decodeFixed32(varName+"temp", "uint32")
		p.P(varName, ` := `, p.mathPkg.Use(), `.Float32frombits(`, varName, `temp)`)
	case descriptor.FieldDescriptorProto_TYPE_INT64:
		p.P(`var `, varName, ` int64`)
		p.decodeVarint(varName, "int64")
	case descriptor.FieldDescriptorProto_TYPE_UINT64:
		p.P(`var `, varName, ` uint64`)
		p.decodeVarint(varName, "uint64")
	case descriptor.FieldDescriptorProto_TYPE_INT32:
		p.P(`var `, varName, ` int32`)
		p.decodeVarint(varName, "int32")
	case descriptor.FieldDescriptorProto_TYPE_FIXED64:
		p.P(`var `, varName, ` uint64`)
		p.decodeFixed64(varName, "uint64")
	case descriptor.FieldDescriptorProto_TYPE_FIXED32:
		p.P(`var `, varName, ` uint32`)
		p.decodeFixed32(varName, "uint32")
	case descriptor.FieldDescriptorProto_TYPE_BOOL:
		p.P(`var `, varName, `temp int`)
		p.decodeVarint(varName+"temp", "int")
		p.P(varName, ` := bool(`, varName, `temp != 0)`)
	case descriptor.FieldDescriptorProto_TYPE_STRING:
		p.P(`var stringLen`, varName, ` uint64`)
		p.decodeVarint("stringLen"+varName, "uint64")
		p.P(`intStringLen`, varName, ` := int(stringLen`, varName, `)`)
		p.P(`if intStringLen`, varName, ` < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`postStringIndex`, varName, ` := iNdEx + intStringLen`, varName)
		p.P(`if postStringIndex`, varName, ` > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		cast, _ := p.GoType(nil, field)
		cast = strings.Replace(cast, "*", "", 1)
		p.P(varName, ` := `, cast, `(data[iNdEx:postStringIndex`, varName, `])`)
		p.P(`iNdEx = postStringIndex`, varName)
	case descriptor.FieldDescriptorProto_TYPE_MESSAGE:
		p.P(`var mapmsglen int`)
		p.decodeVarint("mapmsglen", "int")
		p.P(`if mapmsglen < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`postmsgIndex := iNdEx + mapmsglen`)
		p.P(`if mapmsglen < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`if postmsgIndex > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		desc := p.ObjectNamed(field.GetTypeName())
		msgname := p.TypeName(desc)
		p.P(varName, ` := &`, msgname, `{}`)
		p.P(`if err := `, varName, `.Unmarshal(data[iNdEx:postmsgIndex]); err != nil {`)
		p.In()
		p.P(`return err`)
		p.Out()
		p.P(`}`)
		p.P(`iNdEx = postmsgIndex`)
	case descriptor.FieldDescriptorProto_TYPE_BYTES:
		p.P(`var mapbyteLen uint64`)
		p.decodeVarint("mapbyteLen", "uint64")
		p.P(`intMapbyteLen := int(mapbyteLen)`)
		p.P(`if intMapbyteLen < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`postbytesIndex := iNdEx + intMapbyteLen`)
		p.P(`if postbytesIndex > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		p.P(varName, ` := make([]byte, mapbyteLen)`)
		p.P(`copy(`, varName, `, data[iNdEx:postbytesIndex])`)
		p.P(`iNdEx = postbytesIndex`)
	case descriptor.FieldDescriptorProto_TYPE_UINT32:
		p.P(`var `, varName, ` uint32`)
		p.decodeVarint(varName, "uint32")
	case descriptor.FieldDescriptorProto_TYPE_ENUM:
		typName := p.TypeName(p.ObjectNamed(field.GetTypeName()))
		p.P(`var `, varName, ` `, typName)
		p.decodeVarint(varName, typName)
	case descriptor.FieldDescriptorProto_TYPE_SFIXED32:
		p.P(`var `, varName, ` int32`)
		p.decodeFixed32(varName, "int32")
	case descriptor.FieldDescriptorProto_TYPE_SFIXED64:
		p.P(`var `, varName, ` int64`)
		p.decodeFixed64(varName, "int64")
	case descriptor.FieldDescriptorProto_TYPE_SINT32:
		p.P(`var `, varName, `temp int32`)
		p.decodeVarint(varName+"temp", "int32")
		p.P(varName, `temp = int32((uint32(`, varName, `temp) >> 1) ^ uint32(((`, varName, `temp&1)<<31)>>31))`)
		p.P(varName, ` := int32(`, varName, `temp)`)
	case descriptor.FieldDescriptorProto_TYPE_SINT64:
		p.P(`var `, varName, `temp uint64`)
		p.decodeVarint(varName+"temp", "uint64")
		p.P(varName, `temp = (`, varName, `temp >> 1) ^ uint64((int64(`, varName, `temp&1)<<63)>>63)`)
		p.P(varName, ` := int64(`, varName, `temp)`)
	}
}

func (p *unmarshal) noStarOrSliceType(msg *generator.Descriptor, field *descriptor.FieldDescriptorProto) string {
	typ, _ := p.GoType(msg, field)
	if typ[0] == '*' {
		return typ[1:]
	}
	if typ[0] == '[' && typ[1] == ']' {
		return typ[2:]
	}
	return typ
}

func (p *unmarshal) field(file *generator.FileDescriptor, msg *generator.Descriptor, field *descriptor.FieldDescriptorProto, fieldname string, proto3 bool) {
	repeated := field.IsRepeated()
	nullable := gogoproto.IsNullable(field)
	typ := p.noStarOrSliceType(msg, field)
	oneof := field.OneofIndex != nil
	switch *field.Type {
	case descriptor.FieldDescriptorProto_TYPE_DOUBLE:
		if !p.unsafe || gogoproto.IsCastType(field) {
			p.P(`var v uint64`)
			p.decodeFixed64("v", "uint64")
			if oneof {
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{`, typ, "(", p.mathPkg.Use(), `.Float64frombits(v))}`)
			} else if repeated {
				p.P(`v2 := `, typ, "(", p.mathPkg.Use(), `.Float64frombits(v))`)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v2)`)
			} else if proto3 || !nullable {
				p.P(`m.`, fieldname, ` = `, typ, "(", p.mathPkg.Use(), `.Float64frombits(v))`)
			} else {
				p.P(`v2 := `, typ, "(", p.mathPkg.Use(), `.Float64frombits(v))`)
				p.P(`m.`, fieldname, ` = &v2`)
			}
		} else {
			if oneof {
				p.P(`var v float64`)
				p.unsafeFixed64("v", "float64")
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v float64`)
				p.unsafeFixed64("v", "float64")
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.unsafeFixed64(`m.`+fieldname, "float64")
			} else {
				p.P(`var v float64`)
				p.unsafeFixed64("v", "float64")
				p.P(`m.`, fieldname, ` = &v`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_FLOAT:
		if !p.unsafe || gogoproto.IsCastType(field) {
			p.P(`var v uint32`)
			p.decodeFixed32("v", "uint32")
			if oneof {
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{`, typ, "(", p.mathPkg.Use(), `.Float32frombits(v))}`)
			} else if repeated {
				p.P(`v2 := `, typ, "(", p.mathPkg.Use(), `.Float32frombits(v))`)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v2)`)
			} else if proto3 || !nullable {
				p.P(`m.`, fieldname, ` = `, typ, "(", p.mathPkg.Use(), `.Float32frombits(v))`)
			} else {
				p.P(`v2 := `, typ, "(", p.mathPkg.Use(), `.Float32frombits(v))`)
				p.P(`m.`, fieldname, ` = &v2`)
			}
		} else {
			if oneof {
				p.P(`var v float32`)
				p.unsafeFixed32("v", "float32")
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v float32`)
				p.unsafeFixed32("v", "float32")
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.unsafeFixed32("m."+fieldname, "float32")
			} else {
				p.P(`var v float32`)
				p.unsafeFixed32("v", "float32")
				p.P(`m.`, fieldname, ` = &v`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_INT64:
		if oneof {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if repeated {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = 0`)
			p.decodeVarint("m."+fieldname, typ)
		} else {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &v`)
		}
	case descriptor.FieldDescriptorProto_TYPE_UINT64:
		if oneof {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if repeated {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = 0`)
			p.decodeVarint("m."+fieldname, typ)
		} else {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &v`)
		}
	case descriptor.FieldDescriptorProto_TYPE_INT32:
		if oneof {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if repeated {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = 0`)
			p.decodeVarint("m."+fieldname, typ)
		} else {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &v`)
		}
	case descriptor.FieldDescriptorProto_TYPE_FIXED64:
		if !p.unsafe || gogoproto.IsCastType(field) {
			if oneof {
				p.P(`var v `, typ)
				p.decodeFixed64("v", typ)
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v `, typ)
				p.decodeFixed64("v", typ)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.P(`m.`, fieldname, ` = 0`)
				p.decodeFixed64("m."+fieldname, typ)
			} else {
				p.P(`var v `, typ)
				p.decodeFixed64("v", typ)
				p.P(`m.`, fieldname, ` = &v`)
			}
		} else {
			if oneof {
				p.P(`var v uint64`)
				p.unsafeFixed64("v", "uint64")
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v uint64`)
				p.unsafeFixed64("v", "uint64")
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.unsafeFixed64("m."+fieldname, "uint64")
			} else {
				p.P(`var v uint64`)
				p.unsafeFixed64("v", "uint64")
				p.P(`m.`, fieldname, ` = &v`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_FIXED32:
		if !p.unsafe || gogoproto.IsCastType(field) {
			if oneof {
				p.P(`var v `, typ)
				p.decodeFixed32("v", typ)
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v `, typ)
				p.decodeFixed32("v", typ)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.P(`m.`, fieldname, ` = 0`)
				p.decodeFixed32("m."+fieldname, typ)
			} else {
				p.P(`var v `, typ)
				p.decodeFixed32("v", typ)
				p.P(`m.`, fieldname, ` = &v`)
			}
		} else {
			if oneof {
				p.P(`var v uint32`)
				p.unsafeFixed32("v", "uint32")
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v uint32`)
				p.unsafeFixed32("v", "uint32")
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.unsafeFixed32("m."+fieldname, "uint32")
			} else {
				p.P(`var v uint32`)
				p.unsafeFixed32("v", "uint32")
				p.P(`m.`, fieldname, ` = &v`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_BOOL:
		p.P(`var v int`)
		p.decodeVarint("v", "int")
		if oneof {
			p.P(`b := `, typ, `(v != 0)`)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{b}`)
		} else if repeated {
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, `, typ, `(v != 0))`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = `, typ, `(v != 0)`)
		} else {
			p.P(`b := `, typ, `(v != 0)`)
			p.P(`m.`, fieldname, ` = &b`)
		}
	case descriptor.FieldDescriptorProto_TYPE_STRING:
		p.P(`var stringLen uint64`)
		p.decodeVarint("stringLen", "uint64")
		p.P(`intStringLen := int(stringLen)`)
		p.P(`if intStringLen < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`postIndex := iNdEx + intStringLen`)
		p.P(`if postIndex > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		if oneof {
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{`, typ, `(data[iNdEx:postIndex])}`)
		} else if repeated {
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, `, typ, `(data[iNdEx:postIndex]))`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = `, typ, `(data[iNdEx:postIndex])`)
		} else {
			p.P(`s := `, typ, `(data[iNdEx:postIndex])`)
			p.P(`m.`, fieldname, ` = &s`)
		}
		p.P(`iNdEx = postIndex`)
	case descriptor.FieldDescriptorProto_TYPE_GROUP:
		panic(fmt.Errorf("unmarshaler does not support group %v", fieldname))
	case descriptor.FieldDescriptorProto_TYPE_MESSAGE:
		desc := p.ObjectNamed(field.GetTypeName())
		msgname := p.TypeName(desc)
		p.P(`var msglen int`)
		p.decodeVarint("msglen", "int")
		p.P(`if msglen < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`postIndex := iNdEx + msglen`)
		p.P(`if postIndex > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		if oneof {
			p.P(`v := &`, msgname, `{}`)
			p.P(`if err := v.Unmarshal(data[iNdEx:postIndex]); err != nil {`)
			p.In()
			p.P(`return err`)
			p.Out()
			p.P(`}`)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if p.IsMap(field) {
			m := p.GoMapType(nil, field)

			keygoTyp, _ := p.GoType(nil, m.KeyField)
			keygoAliasTyp, _ := p.GoType(nil, m.KeyAliasField)
			// keys may not be pointers
			keygoTyp = strings.Replace(keygoTyp, "*", "", 1)
			keygoAliasTyp = strings.Replace(keygoAliasTyp, "*", "", 1)

			valuegoTyp, _ := p.GoType(nil, m.ValueField)
			valuegoAliasTyp, _ := p.GoType(nil, m.ValueAliasField)

			// if the map type is an alias and key or values are aliases (type Foo map[Bar]Baz),
			// we need to explicitly record their use here.
			p.RecordTypeUse(m.KeyAliasField.GetTypeName())
			p.RecordTypeUse(m.ValueAliasField.GetTypeName())

			nullable, valuegoTyp, valuegoAliasTyp = generator.GoMapValueTypes(field, m.ValueField, valuegoTyp, valuegoAliasTyp)

			p.P(`var keykey uint64`)
			p.decodeVarint("keykey", "uint64")
			p.mapField("mapkey", m.KeyAliasField)
			p.P(`var valuekey uint64`)
			p.decodeVarint("valuekey", "uint64")
			p.mapField("mapvalue", m.ValueAliasField)
			p.P(`if m.`, fieldname, ` == nil {`)
			p.In()
			p.P(`m.`, fieldname, ` = make(`, m.GoType, `)`)
			p.Out()
			p.P(`}`)
			s := `m.` + fieldname
			if keygoTyp == keygoAliasTyp {
				s += `[mapkey]`
			} else {
				s += `[` + keygoAliasTyp + `(mapkey)]`
			}
			v := `mapvalue`
			if m.ValueField.IsMessage() && !nullable {
				v = `*` + v
			}
			if valuegoTyp != valuegoAliasTyp {
				v = `((` + valuegoAliasTyp + `)(` + v + `))`
			}
			p.P(s, ` = `, v)
		} else if repeated {
			if nullable {
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, &`, msgname, `{})`)
			} else {
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, `, msgname, `{})`)
			}
			p.P(`if err := m.`, fieldname, `[len(m.`, fieldname, `)-1].Unmarshal(data[iNdEx:postIndex]); err != nil {`)
			p.In()
			p.P(`return err`)
			p.Out()
			p.P(`}`)
		} else if nullable {
			p.P(`if m.`, fieldname, ` == nil {`)
			p.In()
			p.P(`m.`, fieldname, ` = &`, msgname, `{}`)
			p.Out()
			p.P(`}`)
			p.P(`if err := m.`, fieldname, `.Unmarshal(data[iNdEx:postIndex]); err != nil {`)
			p.In()
			p.P(`return err`)
			p.Out()
			p.P(`}`)
		} else {
			p.P(`if err := m.`, fieldname, `.Unmarshal(data[iNdEx:postIndex]); err != nil {`)
			p.In()
			p.P(`return err`)
			p.Out()
			p.P(`}`)
		}
		p.P(`iNdEx = postIndex`)
	case descriptor.FieldDescriptorProto_TYPE_BYTES:
		p.P(`var byteLen int`)
		p.decodeVarint("byteLen", "int")
		p.P(`if byteLen < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength` + p.localName)
		p.Out()
		p.P(`}`)
		p.P(`postIndex := iNdEx + byteLen`)
		p.P(`if postIndex > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		if !gogoproto.IsCustomType(field) {
			if oneof {
				p.P(`v := make([]byte, postIndex-iNdEx)`)
				p.P(`copy(v, data[iNdEx:postIndex])`)
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, make([]byte, postIndex-iNdEx))`)
				p.P(`copy(m.`, fieldname, `[len(m.`, fieldname, `)-1], data[iNdEx:postIndex])`)
			} else {
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `[:0] , data[iNdEx:postIndex]...)`)
				p.P(`if m.`, fieldname, ` == nil {`)
				p.In()
				p.P(`m.`, fieldname, ` = []byte{}`)
				p.Out()
				p.P(`}`)
			}
		} else {
			_, ctyp, err := generator.GetCustomType(field)
			if err != nil {
				panic(err)
			}
			if oneof {
				p.P(`var vv `, ctyp)
				p.P(`v := &vv`)
				p.P(`if err := v.Unmarshal(data[iNdEx:postIndex]); err != nil {`)
				p.In()
				p.P(`return err`)
				p.Out()
				p.P(`}`)
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{*v}`)
			} else if repeated {
				p.P(`var v `, ctyp)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
				p.P(`if err := m.`, fieldname, `[len(m.`, fieldname, `)-1].Unmarshal(data[iNdEx:postIndex]); err != nil {`)
				p.In()
				p.P(`return err`)
				p.Out()
				p.P(`}`)
			} else if nullable {
				p.P(`var v `, ctyp)
				p.P(`m.`, fieldname, ` = &v`)
				p.P(`if err := m.`, fieldname, `.Unmarshal(data[iNdEx:postIndex]); err != nil {`)
				p.In()
				p.P(`return err`)
				p.Out()
				p.P(`}`)
			} else {
				p.P(`if err := m.`, fieldname, `.Unmarshal(data[iNdEx:postIndex]); err != nil {`)
				p.In()
				p.P(`return err`)
				p.Out()
				p.P(`}`)
			}
		}
		p.P(`iNdEx = postIndex`)
	case descriptor.FieldDescriptorProto_TYPE_UINT32:
		if oneof {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if repeated {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = 0`)
			p.decodeVarint("m."+fieldname, typ)
		} else {
			p.P(`var v `, typ)
			p.decodeVarint("v", typ)
			p.P(`m.`, fieldname, ` = &v`)
		}
	case descriptor.FieldDescriptorProto_TYPE_ENUM:
		typName := p.TypeName(p.ObjectNamed(field.GetTypeName()))
		if oneof {
			p.P(`var v `, typName)
			p.decodeVarint("v", typName)
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if repeated {
			p.P(`var v `, typName)
			p.decodeVarint("v", typName)
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = 0`)
			p.decodeVarint("m."+fieldname, typName)
		} else {
			p.P(`var v `, typName)
			p.decodeVarint("v", typName)
			p.P(`m.`, fieldname, ` = &v`)
		}
	case descriptor.FieldDescriptorProto_TYPE_SFIXED32:
		if !p.unsafe || gogoproto.IsCastType(field) {
			if oneof {
				p.P(`var v `, typ)
				p.decodeFixed32("v", typ)
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v `, typ)
				p.decodeFixed32("v", typ)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.P(`m.`, fieldname, ` = 0`)
				p.decodeFixed32("m."+fieldname, typ)
			} else {
				p.P(`var v `, typ)
				p.decodeFixed32("v", typ)
				p.P(`m.`, fieldname, ` = &v`)
			}
		} else {
			if oneof {
				p.P(`var v int32`)
				p.unsafeFixed32("v", "int32")
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v int32`)
				p.unsafeFixed32("v", "int32")
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.unsafeFixed32("m."+fieldname, "int32")
			} else {
				p.P(`var v int32`)
				p.unsafeFixed32("v", "int32")
				p.P(`m.`, fieldname, ` = &v`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_SFIXED64:
		if !p.unsafe || gogoproto.IsCastType(field) {
			if oneof {
				p.P(`var v `, typ)
				p.decodeFixed64("v", typ)
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v `, typ)
				p.decodeFixed64("v", typ)
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.P(`m.`, fieldname, ` = 0`)
				p.decodeFixed64("m."+fieldname, typ)
			} else {
				p.P(`var v `, typ)
				p.decodeFixed64("v", typ)
				p.P(`m.`, fieldname, ` = &v`)
			}
		} else {
			if oneof {
				p.P(`var v int64`)
				p.unsafeFixed64("v", "int64")
				p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
			} else if repeated {
				p.P(`var v int64`)
				p.unsafeFixed64("v", "int64")
				p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
			} else if proto3 || !nullable {
				p.unsafeFixed64("m."+fieldname, "int64")
			} else {
				p.P(`var v int64`)
				p.unsafeFixed64("v", "int64")
				p.P(`m.`, fieldname, ` = &v`)
			}
		}
	case descriptor.FieldDescriptorProto_TYPE_SINT32:
		p.P(`var v `, typ)
		p.decodeVarint("v", typ)
		p.P(`v = `, typ, `((uint32(v) >> 1) ^ uint32(((v&1)<<31)>>31))`)
		if oneof {
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{v}`)
		} else if repeated {
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, v)`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = v`)
		} else {
			p.P(`m.`, fieldname, ` = &v`)
		}
	case descriptor.FieldDescriptorProto_TYPE_SINT64:
		p.P(`var v uint64`)
		p.decodeVarint("v", "uint64")
		p.P(`v = (v >> 1) ^ uint64((int64(v&1)<<63)>>63)`)
		if oneof {
			p.P(`m.`, fieldname, ` = &`, p.OneOfTypeName(msg, field), `{`, typ, `(v)}`)
		} else if repeated {
			p.P(`m.`, fieldname, ` = append(m.`, fieldname, `, `, typ, `(v))`)
		} else if proto3 || !nullable {
			p.P(`m.`, fieldname, ` = `, typ, `(v)`)
		} else {
			p.P(`v2 := `, typ, `(v)`)
			p.P(`m.`, fieldname, ` = &v2`)
		}
	default:
		panic("not implemented")
	}
}

func (p *unmarshal) Generate(file *generator.FileDescriptor) {
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.atleastOne = false
	p.localName = generator.FileName(file)
	if p.unsafe {
		p.localName += "Unsafe"
	}

	p.ioPkg = p.NewImport("io")
	p.mathPkg = p.NewImport("math")
	p.unsafePkg = p.NewImport("unsafe")
	fmtPkg := p.NewImport("fmt")
	protoPkg := p.NewImport("github.com/gogo/protobuf/proto")
	if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
		protoPkg = p.NewImport("github.com/golang/protobuf/proto")
	}

	for _, message := range file.Messages() {
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		if p.unsafe {
			if !gogoproto.IsUnsafeUnmarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				continue
			}
			if gogoproto.IsUnmarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				panic(fmt.Sprintf("unsafe_unmarshaler and unmarshaler enabled for %v", ccTypeName))
			}
		}
		if !p.unsafe {
			if !gogoproto.IsUnmarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				continue
			}
			if gogoproto.IsUnsafeUnmarshaler(file.FileDescriptorProto, message.DescriptorProto) {
				panic(fmt.Sprintf("unsafe_unmarshaler and unmarshaler enabled for %v", ccTypeName))
			}
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		p.atleastOne = true

		// build a map required field_id -> bitmask offset
		rfMap := make(map[int32]uint)
		rfNextId := uint(0)
		for _, field := range message.Field {
			if field.IsRequired() {
				rfMap[field.GetNumber()] = rfNextId
				rfNextId++
			}
		}
		rfCount := len(rfMap)

		p.P(`func (m *`, ccTypeName, `) Unmarshal(data []byte) error {`)
		p.In()
		if rfCount > 0 {
			p.P(`var hasFields [`, strconv.Itoa(1+(rfCount-1)/64), `]uint64`)
		}
		p.P(`l := len(data)`)
		p.P(`iNdEx := 0`)
		p.P(`for iNdEx < l {`)
		p.In()
		p.P(`preIndex := iNdEx`)
		p.P(`var wire uint64`)
		p.decodeVarint("wire", "uint64")
		p.P(`fieldNum := int32(wire >> 3)`)
		if len(message.Field) > 0 || !message.IsGroup() {
			p.P(`wireType := int(wire & 0x7)`)
		}
		if !message.IsGroup() {
			p.P(`if wireType == `, strconv.Itoa(proto.WireEndGroup), ` {`)
			p.In()
			p.P(`return `, fmtPkg.Use(), `.Errorf("proto: `+message.GetName()+`: wiretype end group for non-group")`)
			p.Out()
			p.P(`}`)
		}
		p.P(`if fieldNum <= 0 {`)
		p.In()
		p.P(`return `, fmtPkg.Use(), `.Errorf("proto: `+message.GetName()+`: illegal tag %d (wire type %d)", fieldNum, wire)`)
		p.Out()
		p.P(`}`)
		p.P(`switch fieldNum {`)
		p.In()
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			errFieldname := fieldname
			if field.OneofIndex != nil {
				errFieldname = p.GetOneOfFieldName(message, field)
			}
			packed := field.IsPacked()
			p.P(`case `, strconv.Itoa(int(field.GetNumber())), `:`)
			p.In()
			wireType := field.WireType()
			if packed {
				p.P(`if wireType == `, strconv.Itoa(proto.WireBytes), `{`)
				p.In()
				p.P(`var packedLen int`)
				p.decodeVarint("packedLen", "int")
				p.P(`if packedLen < 0 {`)
				p.In()
				p.P(`return ErrInvalidLength` + p.localName)
				p.Out()
				p.P(`}`)
				p.P(`postIndex := iNdEx + packedLen`)
				p.P(`if postIndex > l {`)
				p.In()
				p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
				p.Out()
				p.P(`}`)
				p.P(`for iNdEx < postIndex {`)
				p.In()
				p.field(file, message, field, fieldname, false)
				p.Out()
				p.P(`}`)
				p.Out()
				p.P(`} else if wireType == `, strconv.Itoa(wireType), `{`)
				p.In()
				p.field(file, message, field, fieldname, false)
				p.Out()
				p.P(`} else {`)
				p.In()
				p.P(`return ` + fmtPkg.Use() + `.Errorf("proto: wrong wireType = %d for field ` + errFieldname + `", wireType)`)
				p.Out()
				p.P(`}`)
			} else {
				p.P(`if wireType != `, strconv.Itoa(wireType), `{`)
				p.In()
				p.P(`return ` + fmtPkg.Use() + `.Errorf("proto: wrong wireType = %d for field ` + errFieldname + `", wireType)`)
				p.Out()
				p.P(`}`)
				p.field(file, message, field, fieldname, proto3)
			}

			if field.IsRequired() {
				fieldBit, ok := rfMap[field.GetNumber()]
				if !ok {
					panic("field is required, but no bit registered")
				}
				p.P(`hasFields[`, strconv.Itoa(int(fieldBit/64)), `] |= uint64(`, fmt.Sprintf("0x%08x", 1<<(fieldBit%64)), `)`)
			}
		}
		p.Out()
		p.P(`default:`)
		p.In()
		if message.DescriptorProto.HasExtension() {
			c := []string{}
			for _, erange := range message.GetExtensionRange() {
				c = append(c, `((fieldNum >= `+strconv.Itoa(int(erange.GetStart()))+") && (fieldNum<"+strconv.Itoa(int(erange.GetEnd()))+`))`)
			}
			p.P(`if `, strings.Join(c, "||"), `{`)
			p.In()
			p.P(`var sizeOfWire int`)
			p.P(`for {`)
			p.In()
			p.P(`sizeOfWire++`)
			p.P(`wire >>= 7`)
			p.P(`if wire == 0 {`)
			p.In()
			p.P(`break`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)
			p.P(`iNdEx-=sizeOfWire`)
			p.P(`skippy, err := skip`, p.localName+`(data[iNdEx:])`)
			p.P(`if err != nil {`)
			p.In()
			p.P(`return err`)
			p.Out()
			p.P(`}`)
			p.P(`if skippy < 0 {`)
			p.In()
			p.P(`return ErrInvalidLength`, p.localName)
			p.Out()
			p.P(`}`)
			p.P(`if (iNdEx + skippy) > l {`)
			p.In()
			p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
			p.Out()
			p.P(`}`)
			p.P(protoPkg.Use(), `.AppendExtension(m, int32(fieldNum), data[iNdEx:iNdEx+skippy])`)
			p.P(`iNdEx += skippy`)
			p.Out()
			p.P(`} else {`)
			p.In()
		}
		p.P(`iNdEx=preIndex`)
		p.P(`skippy, err := skip`, p.localName, `(data[iNdEx:])`)
		p.P(`if err != nil {`)
		p.In()
		p.P(`return err`)
		p.Out()
		p.P(`}`)
		p.P(`if skippy < 0 {`)
		p.In()
		p.P(`return ErrInvalidLength`, p.localName)
		p.Out()
		p.P(`}`)
		p.P(`if (iNdEx + skippy) > l {`)
		p.In()
		p.P(`return `, p.ioPkg.Use(), `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`m.XXX_unrecognized = append(m.XXX_unrecognized, data[iNdEx:iNdEx+skippy]...)`)
		}
		p.P(`iNdEx += skippy`)
		p.Out()
		if message.DescriptorProto.HasExtension() {
			p.Out()
			p.P(`}`)
		}
		p.Out()
		p.P(`}`)
		p.Out()
		p.P(`}`)

		for _, field := range message.Field {
			if !field.IsRequired() {
				continue
			}

			fieldBit, ok := rfMap[field.GetNumber()]
			if !ok {
				panic("field is required, but no bit registered")
			}

			p.P(`if hasFields[`, strconv.Itoa(int(fieldBit/64)), `] & uint64(`, fmt.Sprintf("0x%08x", 1<<(fieldBit%64)), `) == 0 {`)
			p.In()
			if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
				p.P(`return new(`, protoPkg.Use(), `.RequiredNotSetError)`)
			} else {
				p.P(`return `, protoPkg.Use(), `.NewRequiredNotSetError("`, field.GetName(), `")`)
			}
			p.Out()
			p.P(`}`)
		}
		p.P()
		p.P(`if iNdEx > l {`)
		p.In()
		p.P(`return ` + p.ioPkg.Use() + `.ErrUnexpectedEOF`)
		p.Out()
		p.P(`}`)
		p.P(`return nil`)
		p.Out()
		p.P(`}`)
	}
	if !p.atleastOne {
		return
	}

	p.P(`func skip` + p.localName + `(data []byte) (n int, err error) {
		l := len(data)
		iNdEx := 0
		for iNdEx < l {
			var wire uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflow` + p.localName + `
				}
				if iNdEx >= l {
					return 0, ` + p.ioPkg.Use() + `.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				wire |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			wireType := int(wire & 0x7)
			switch wireType {
			case 0:
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflow` + p.localName + `
					}
					if iNdEx >= l {
						return 0, ` + p.ioPkg.Use() + `.ErrUnexpectedEOF
					}
					iNdEx++
					if data[iNdEx-1] < 0x80 {
						break
					}
				}
				return iNdEx, nil
			case 1:
				iNdEx += 8
				return iNdEx, nil
			case 2:
				var length int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflow` + p.localName + `
					}
					if iNdEx >= l {
						return 0, ` + p.ioPkg.Use() + `.ErrUnexpectedEOF
					}
					b := data[iNdEx]
					iNdEx++
					length |= (int(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				iNdEx += length
				if length < 0 {
					return 0, ErrInvalidLength` + p.localName + `
				}
				return iNdEx, nil
			case 3:
				for {
					var innerWire uint64
					var start int = iNdEx
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return 0, ErrIntOverflow` + p.localName + `
						}
						if iNdEx >= l {
							return 0, ` + p.ioPkg.Use() + `.ErrUnexpectedEOF
						}
						b := data[iNdEx]
						iNdEx++
						innerWire |= (uint64(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					innerWireType := int(innerWire & 0x7)
					if innerWireType == 4 {
						break
					}
					next, err := skip` + p.localName + `(data[start:])
					if err != nil {
						return 0, err
					}
					iNdEx = start + next
				}
				return iNdEx, nil
			case 4:
				return iNdEx, nil
			case 5:
				iNdEx += 4
				return iNdEx, nil
			default:
				return 0, ` + fmtPkg.Use() + `.Errorf("proto: illegal wireType %d", wireType)
			}
		}
		panic("unreachable")
	}

	var (
		ErrInvalidLength` + p.localName + ` = ` + fmtPkg.Use() + `.Errorf("proto: negative length found during unmarshaling")
		ErrIntOverflow` + p.localName + ` = ` + fmtPkg.Use() + `.Errorf("proto: integer overflow")
	)
	`)
}

func init() {
	generator.RegisterPlugin(NewUnmarshal())
	generator.RegisterPlugin(NewUnsafeUnmarshal())
}
