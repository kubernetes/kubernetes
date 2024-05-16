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
The onlyone plugin generates code for the onlyone extension.
All fields must be nullable and only one of the fields may be set, like a union.
Two methods are generated

  GetValue() interface{}

and

  SetValue(v interface{}) (set bool)

These provide easier interaction with a onlyone.

The onlyone extension is not called union as this causes compile errors in the C++ generated code.
There can only be one ;)

It is enabled by the following extensions:

  - onlyone
  - onlyone_all

The onlyone plugin also generates a test given it is enabled using one of the following extensions:

  - testgen
  - testgen_all

Lets look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  message U {
	  option (gogoproto.onlyone) = true;
	  optional A A = 1;
	  optional B B = 2;
  }

given to the onlyone plugin, will generate code which looks a lot like this:

	func (this *U) GetValue() interface{} {
		if this.A != nil {
			return this.A
		}
		if this.B != nil {
			return this.B
		}
		return nil
	}

	func (this *U) SetValue(value interface{}) bool {
		switch vt := value.(type) {
		case *A:
			this.A = vt
		case *B:
			this.B = vt
		default:
			return false
		}
		return true
	}

and the following test code:

  func TestUUnion(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedU(popr)
	v := p.GetValue()
	msg := &U{}
	if !msg.SetValue(v) {
		t.Fatalf("Union: Could not set Value")
	}
	if !p.Equal(msg) {
		t.Fatalf("%#v !Union Equal %#v", msg, p)
	}
  }

*/
package union

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type union struct {
	*generator.Generator
	generator.PluginImports
}

func NewUnion() *union {
	return &union{}
}

func (p *union) Name() string {
	return "union"
}

func (p *union) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *union) Generate(file *generator.FileDescriptor) {
	p.PluginImports = generator.NewPluginImports(p.Generator)

	for _, message := range file.Messages() {
		if !gogoproto.IsUnion(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if message.DescriptorProto.HasExtension() {
			panic("onlyone does not currently support extensions")
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}

		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		p.P(`func (this *`, ccTypeName, `) GetValue() interface{} {`)
		p.In()
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			if fieldname == "Value" {
				panic("cannot have a onlyone message " + ccTypeName + " with a field named Value")
			}
			p.P(`if this.`, fieldname, ` != nil {`)
			p.In()
			p.P(`return this.`, fieldname)
			p.Out()
			p.P(`}`)
		}
		p.P(`return nil`)
		p.Out()
		p.P(`}`)
		p.P(``)
		p.P(`func (this *`, ccTypeName, `) SetValue(value interface{}) bool {`)
		p.In()
		p.P(`switch vt := value.(type) {`)
		p.In()
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			goTyp, _ := p.GoType(message, field)
			p.P(`case `, goTyp, `:`)
			p.In()
			p.P(`this.`, fieldname, ` = vt`)
			p.Out()
		}
		p.P(`default:`)
		p.In()
		for _, field := range message.Field {
			fieldname := p.GetFieldName(message, field)
			if field.IsMessage() {
				goTyp, _ := p.GoType(message, field)
				obj := p.ObjectNamed(field.GetTypeName()).(*generator.Descriptor)

				if gogoproto.IsUnion(obj.File().FileDescriptorProto, obj.DescriptorProto) {
					p.P(`this.`, fieldname, ` = new(`, generator.GoTypeToName(goTyp), `)`)
					p.P(`if set := this.`, fieldname, `.SetValue(value); set {`)
					p.In()
					p.P(`return true`)
					p.Out()
					p.P(`}`)
					p.P(`this.`, fieldname, ` = nil`)
				}
			}
		}
		p.P(`return false`)
		p.Out()
		p.P(`}`)
		p.P(`return true`)
		p.Out()
		p.P(`}`)
	}
}

func init() {
	generator.RegisterPlugin(NewUnion())
}
