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
The enumstringer (experimental) plugin generates a String method for each enum.

It is enabled by the following extensions:

  - enum_stringer
  - enum_stringer_all

This package is subject to change.

*/
package enumstringer

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type enumstringer struct {
	*generator.Generator
	generator.PluginImports
	atleastOne bool
	localName  string
}

func NewEnumStringer() *enumstringer {
	return &enumstringer{}
}

func (p *enumstringer) Name() string {
	return "enumstringer"
}

func (p *enumstringer) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *enumstringer) Generate(file *generator.FileDescriptor) {
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.atleastOne = false

	p.localName = generator.FileName(file)

	strconvPkg := p.NewImport("strconv")

	for _, enum := range file.Enums() {
		if !gogoproto.IsEnumStringer(file.FileDescriptorProto, enum.EnumDescriptorProto) {
			continue
		}
		if gogoproto.IsGoEnumStringer(file.FileDescriptorProto, enum.EnumDescriptorProto) {
			panic("Go enum stringer conflicts with new enumstringer plugin: please use gogoproto.goproto_enum_stringer or gogoproto.goproto_enum_string_all and set it to false")
		}
		p.atleastOne = true
		ccTypeName := generator.CamelCaseSlice(enum.TypeName())
		p.P("func (x ", ccTypeName, ") String() string {")
		p.In()
		p.P(`s, ok := `, ccTypeName, `_name[int32(x)]`)
		p.P(`if ok {`)
		p.In()
		p.P(`return s`)
		p.Out()
		p.P(`}`)
		p.P(`return `, strconvPkg.Use(), `.Itoa(int(x))`)
		p.Out()
		p.P(`}`)
	}

	if !p.atleastOne {
		return
	}

}

func init() {
	generator.RegisterPlugin(NewEnumStringer())
}
