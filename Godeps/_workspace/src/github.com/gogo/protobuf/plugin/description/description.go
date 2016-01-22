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
The description (experimental) plugin generates a Description method for each message.
The Description method returns a populated google_protobuf.FileDescriptorSet struct.
This contains the description of the files used to generate this message.

It is enabled by the following extensions:

  - description
  - description_all

The description plugin also generates a test given it is enabled using one of the following extensions:

  - testgen
  - testgen_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  message B {
	option (gogoproto.description) = true;
	optional A A = 1 [(gogoproto.nullable) = false, (gogoproto.embed) = true];
	repeated bytes G = 2 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uint128", (gogoproto.nullable) = false];
  }

given to the description plugin, will generate the following code:

  func (this *B) Description() (desc *google_protobuf.FileDescriptorSet) {
	return ExampleDescription()
  }

and the following test code:

  func TestDescription(t *testing9.T) {
	ExampleDescription()
  }

The hope is to use this struct in some way instead of reflect.
This package is subject to change, since a use has not been figured out yet.

*/
package description

import (
	"fmt"
	"github.com/gogo/protobuf/gogoproto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type plugin struct {
	*generator.Generator
	used bool
}

func NewPlugin() *plugin {
	return &plugin{}
}

func (p *plugin) Name() string {
	return "description"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	p.used = false
	localName := generator.FileName(file)
	for _, message := range file.Messages() {
		if !gogoproto.HasDescription(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		p.used = true
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		p.P(`func (this *`, ccTypeName, `) Description() (desc *descriptor.FileDescriptorSet) {`)
		p.In()
		p.P(`return `, localName, `Description()`)
		p.Out()
		p.P(`}`)
	}

	if p.used {

		p.P(`func `, localName, `Description() (desc *descriptor.FileDescriptorSet) {`)
		p.In()
		//Don't generate SourceCodeInfo, since it will create too much code.

		ss := make([]*descriptor.SourceCodeInfo, 0)
		for _, f := range p.Generator.AllFiles().GetFile() {
			ss = append(ss, f.SourceCodeInfo)
			f.SourceCodeInfo = nil
		}
		s := fmt.Sprintf("%#v", p.Generator.AllFiles())
		for i, f := range p.Generator.AllFiles().GetFile() {
			f.SourceCodeInfo = ss[i]
		}
		p.P(`return `, s)
		p.Out()
		p.P(`}`)
	}
}

func (this *plugin) GenerateImports(file *generator.FileDescriptor) {
	if this.used {
		this.P(`import "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"`)
	}
}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
