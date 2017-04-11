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
The face plugin generates a function will be generated which can convert a structure which satisfies an interface (face) to the specified structure.
This interface contains getters for each of the fields in the struct.
The specified struct is also generated with the getters.
This means that getters should be turned off so as not to conflict with face getters.
This allows it to satisfy its own face.

It is enabled by the following extensions:

  - face
  - face_all

Turn off getters by using the following extensions:

  - getters
  - getters_all

The face plugin also generates a test given it is enabled using one of the following extensions:

  - testgen
  - testgen_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  message A {
	option (gogoproto.face) = true;
	option (gogoproto.goproto_getters) = false;
	optional string Description = 1 [(gogoproto.nullable) = false];
	optional int64 Number = 2 [(gogoproto.nullable) = false];
	optional bytes Id = 3 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uuid", (gogoproto.nullable) = false];
  }

given to the face plugin, will generate the following code:

	type AFace interface {
		Proto() github_com_gogo_protobuf_proto.Message
		GetDescription() string
		GetNumber() int64
		GetId() github_com_gogo_protobuf_test_custom.Uuid
	}

	func (this *A) Proto() github_com_gogo_protobuf_proto.Message {
		return this
	}

	func (this *A) TestProto() github_com_gogo_protobuf_proto.Message {
		return NewAFromFace(this)
	}

	func (this *A) GetDescription() string {
		return this.Description
	}

	func (this *A) GetNumber() int64 {
		return this.Number
	}

	func (this *A) GetId() github_com_gogo_protobuf_test_custom.Uuid {
		return this.Id
	}

	func NewAFromFace(that AFace) *A {
		this := &A{}
		this.Description = that.GetDescription()
		this.Number = that.GetNumber()
		this.Id = that.GetId()
		return this
	}

and the following test code:

	func TestAFace(t *testing7.T) {
		popr := math_rand7.New(math_rand7.NewSource(time7.Now().UnixNano()))
		p := NewPopulatedA(popr, true)
		msg := p.TestProto()
		if !p.Equal(msg) {
			t.Fatalf("%#v !Face Equal %#v", msg, p)
		}
	}

The struct A, representing the message, will also be generated just like always.
As you can see A satisfies its own Face, AFace.

Creating another struct which satisfies AFace is very easy.
Simply create all these methods specified in AFace.
Implementing The Proto method is done with the helper function NewAFromFace:

	func (this *MyStruct) Proto() proto.Message {
	  return NewAFromFace(this)
	}

just the like TestProto method which is used to test the NewAFromFace function.

*/
package face

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type plugin struct {
	*generator.Generator
	generator.PluginImports
}

func NewPlugin() *plugin {
	return &plugin{}
}

func (p *plugin) Name() string {
	return "face"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	p.PluginImports = generator.NewPluginImports(p.Generator)
	protoPkg := p.NewImport("github.com/gogo/protobuf/proto")
	if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
		protoPkg = p.NewImport("github.com/golang/protobuf/proto")
	}
	for _, message := range file.Messages() {
		if !gogoproto.IsFace(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		if message.DescriptorProto.HasExtension() {
			panic("face does not support message with extensions")
		}
		if gogoproto.HasGoGetters(file.FileDescriptorProto, message.DescriptorProto) {
			panic("face requires getters to be disabled please use gogoproto.getters or gogoproto.getters_all and set it to false")
		}
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		p.P(`type `, ccTypeName, `Face interface{`)
		p.In()
		p.P(`Proto() `, protoPkg.Use(), `.Message`)
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			goTyp, _ := p.GoType(message, field)
			if p.IsMap(field) {
				m := p.GoMapType(nil, field)
				goTyp = m.GoType
			}
			p.P(`Get`, fieldname, `() `, goTyp)
		}
		p.Out()
		p.P(`}`)
		p.P(``)
		p.P(`func (this *`, ccTypeName, `) Proto() `, protoPkg.Use(), `.Message {`)
		p.In()
		p.P(`return this`)
		p.Out()
		p.P(`}`)
		p.P(``)
		p.P(`func (this *`, ccTypeName, `) TestProto() `, protoPkg.Use(), `.Message {`)
		p.In()
		p.P(`return New`, ccTypeName, `FromFace(this)`)
		p.Out()
		p.P(`}`)
		p.P(``)
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			goTyp, _ := p.GoType(message, field)
			if p.IsMap(field) {
				m := p.GoMapType(nil, field)
				goTyp = m.GoType
			}
			p.P(`func (this *`, ccTypeName, `) Get`, fieldname, `() `, goTyp, `{`)
			p.In()
			p.P(` return this.`, fieldname)
			p.Out()
			p.P(`}`)
			p.P(``)
		}
		p.P(``)
		p.P(`func New`, ccTypeName, `FromFace(that `, ccTypeName, `Face) *`, ccTypeName, ` {`)
		p.In()
		p.P(`this := &`, ccTypeName, `{}`)
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			p.P(`this.`, fieldname, ` = that.Get`, fieldname, `()`)
		}
		p.P(`return this`)
		p.Out()
		p.P(`}`)
		p.P(``)
	}
}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
