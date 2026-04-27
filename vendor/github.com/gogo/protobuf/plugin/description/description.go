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
	"bytes"
	"compress/gzip"
	"fmt"
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
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
	return "description"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	used := false
	localName := generator.FileName(file)

	p.PluginImports = generator.NewPluginImports(p.Generator)
	descriptorPkg := p.NewImport("github.com/gogo/protobuf/protoc-gen-gogo/descriptor")
	protoPkg := p.NewImport("github.com/gogo/protobuf/proto")
	gzipPkg := p.NewImport("compress/gzip")
	bytesPkg := p.NewImport("bytes")
	ioutilPkg := p.NewImport("io/ioutil")

	for _, message := range file.Messages() {
		if !gogoproto.HasDescription(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		used = true
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		p.P(`func (this *`, ccTypeName, `) Description() (desc *`, descriptorPkg.Use(), `.FileDescriptorSet) {`)
		p.In()
		p.P(`return `, localName, `Description()`)
		p.Out()
		p.P(`}`)
	}

	if used {

		p.P(`func `, localName, `Description() (desc *`, descriptorPkg.Use(), `.FileDescriptorSet) {`)
		p.In()
		//Don't generate SourceCodeInfo, since it will create too much code.

		ss := make([]*descriptor.SourceCodeInfo, 0)
		for _, f := range p.Generator.AllFiles().GetFile() {
			ss = append(ss, f.SourceCodeInfo)
			f.SourceCodeInfo = nil
		}
		b, err := proto.Marshal(p.Generator.AllFiles())
		if err != nil {
			panic(err)
		}
		for i, f := range p.Generator.AllFiles().GetFile() {
			f.SourceCodeInfo = ss[i]
		}
		p.P(`d := &`, descriptorPkg.Use(), `.FileDescriptorSet{}`)
		var buf bytes.Buffer
		w, _ := gzip.NewWriterLevel(&buf, gzip.BestCompression)
		w.Write(b)
		w.Close()
		b = buf.Bytes()
		p.P("var gzipped = []byte{")
		p.In()
		p.P("// ", len(b), " bytes of a gzipped FileDescriptorSet")
		for len(b) > 0 {
			n := 16
			if n > len(b) {
				n = len(b)
			}

			s := ""
			for _, c := range b[:n] {
				s += fmt.Sprintf("0x%02x,", c)
			}
			p.P(s)

			b = b[n:]
		}
		p.Out()
		p.P("}")
		p.P(`r := `, bytesPkg.Use(), `.NewReader(gzipped)`)
		p.P(`gzipr, err := `, gzipPkg.Use(), `.NewReader(r)`)
		p.P(`if err != nil {`)
		p.In()
		p.P(`panic(err)`)
		p.Out()
		p.P(`}`)
		p.P(`ungzipped, err := `, ioutilPkg.Use(), `.ReadAll(gzipr)`)
		p.P(`if err != nil {`)
		p.In()
		p.P(`panic(err)`)
		p.Out()
		p.P(`}`)
		p.P(`if err := `, protoPkg.Use(), `.Unmarshal(ungzipped, d); err != nil {`)
		p.In()
		p.P(`panic(err)`)
		p.Out()
		p.P(`}`)
		p.P(`return d`)
		p.Out()
		p.P(`}`)
	}
}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
