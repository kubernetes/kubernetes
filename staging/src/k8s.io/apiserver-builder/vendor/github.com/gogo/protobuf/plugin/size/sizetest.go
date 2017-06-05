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

package size

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/plugin/testgen"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type test struct {
	*generator.Generator
}

func NewTest(g *generator.Generator) testgen.TestPlugin {
	return &test{g}
}

func (p *test) Generate(imports generator.PluginImports, file *generator.FileDescriptor) bool {
	used := false
	randPkg := imports.NewImport("math/rand")
	timePkg := imports.NewImport("time")
	testingPkg := imports.NewImport("testing")
	protoPkg := imports.NewImport("github.com/gogo/protobuf/proto")
	if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
		protoPkg = imports.NewImport("github.com/golang/protobuf/proto")
	}
	for _, message := range file.Messages() {
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		sizeName := ""
		if gogoproto.IsSizer(file.FileDescriptorProto, message.DescriptorProto) {
			sizeName = "Size"
		} else if gogoproto.IsProtoSizer(file.FileDescriptorProto, message.DescriptorProto) {
			sizeName = "ProtoSize"
		} else {
			continue
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}

		if gogoproto.HasTestGen(file.FileDescriptorProto, message.DescriptorProto) {
			used = true
			p.P(`func Test`, ccTypeName, sizeName, `(t *`, testingPkg.Use(), `.T) {`)
			p.In()
			p.P(`seed := `, timePkg.Use(), `.Now().UnixNano()`)
			p.P(`popr := `, randPkg.Use(), `.New(`, randPkg.Use(), `.NewSource(seed))`)
			p.P(`p := NewPopulated`, ccTypeName, `(popr, true)`)
			p.P(`size2 := `, protoPkg.Use(), `.Size(p)`)
			p.P(`data, err := `, protoPkg.Use(), `.Marshal(p)`)
			p.P(`if err != nil {`)
			p.In()
			p.P(`t.Fatalf("seed = %d, err = %v", seed, err)`)
			p.Out()
			p.P(`}`)
			p.P(`size := p.`, sizeName, `()`)
			p.P(`if len(data) != size {`)
			p.In()
			p.P(`t.Errorf("seed = %d, size %v != marshalled size %v", seed, size, len(data))`)
			p.Out()
			p.P(`}`)
			p.P(`if size2 != size {`)
			p.In()
			p.P(`t.Errorf("seed = %d, size %v != before marshal proto.Size %v", seed, size, size2)`)
			p.Out()
			p.P(`}`)
			p.P(`size3 := `, protoPkg.Use(), `.Size(p)`)
			p.P(`if size3 != size {`)
			p.In()
			p.P(`t.Errorf("seed = %d, size %v != after marshal proto.Size %v", seed, size, size3)`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)
			p.P()
		}

		if gogoproto.HasBenchGen(file.FileDescriptorProto, message.DescriptorProto) {
			used = true
			p.P(`func Benchmark`, ccTypeName, sizeName, `(b *`, testingPkg.Use(), `.B) {`)
			p.In()
			p.P(`popr := `, randPkg.Use(), `.New(`, randPkg.Use(), `.NewSource(616))`)
			p.P(`total := 0`)
			p.P(`pops := make([]*`, ccTypeName, `, 1000)`)
			p.P(`for i := 0; i < 1000; i++ {`)
			p.In()
			p.P(`pops[i] = NewPopulated`, ccTypeName, `(popr, false)`)
			p.Out()
			p.P(`}`)
			p.P(`b.ResetTimer()`)
			p.P(`for i := 0; i < b.N; i++ {`)
			p.In()
			p.P(`total += pops[i%1000].`, sizeName, `()`)
			p.Out()
			p.P(`}`)
			p.P(`b.SetBytes(int64(total / b.N))`)
			p.Out()
			p.P(`}`)
			p.P()
		}

	}
	return used
}

func init() {
	testgen.RegisterTestPlugin(NewTest)
}
