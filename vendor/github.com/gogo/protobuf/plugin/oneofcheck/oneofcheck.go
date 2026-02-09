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
The oneofcheck plugin is used to check whether oneof is not used incorrectly.
For instance:
An error is caused if a oneof field:
  - is used in a face
  - is an embedded field

*/
package oneofcheck

import (
	"fmt"
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
	"os"
)

type plugin struct {
	*generator.Generator
}

func NewPlugin() *plugin {
	return &plugin{}
}

func (p *plugin) Name() string {
	return "oneofcheck"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	for _, msg := range file.Messages() {
		face := gogoproto.IsFace(file.FileDescriptorProto, msg.DescriptorProto)
		for _, field := range msg.GetField() {
			if field.OneofIndex == nil {
				continue
			}
			if face {
				fmt.Fprintf(os.Stderr, "ERROR: field %v.%v cannot be in a face and oneof\n", generator.CamelCase(*msg.Name), generator.CamelCase(*field.Name))
				os.Exit(1)
			}
			if gogoproto.IsEmbed(field) {
				fmt.Fprintf(os.Stderr, "ERROR: field %v.%v cannot be in an oneof and an embedded field\n", generator.CamelCase(*msg.Name), generator.CamelCase(*field.Name))
				os.Exit(1)
			}
			if !gogoproto.IsNullable(field) {
				fmt.Fprintf(os.Stderr, "ERROR: field %v.%v cannot be in an oneof and a non-nullable field\n", generator.CamelCase(*msg.Name), generator.CamelCase(*field.Name))
				os.Exit(1)
			}
			if gogoproto.IsUnion(file.FileDescriptorProto, msg.DescriptorProto) {
				fmt.Fprintf(os.Stderr, "ERROR: field %v.%v cannot be in an oneof and in an union (deprecated)\n", generator.CamelCase(*msg.Name), generator.CamelCase(*field.Name))
				os.Exit(1)
			}
		}
	}
}

func (p *plugin) GenerateImports(*generator.FileDescriptor) {}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
