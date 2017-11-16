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

package compare

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
	"github.com/gogo/protobuf/vanity"
)

type plugin struct {
	*generator.Generator
	generator.PluginImports
	fmtPkg      generator.Single
	bytesPkg    generator.Single
	sortkeysPkg generator.Single
	protoPkg    generator.Single
}

func NewPlugin() *plugin {
	return &plugin{}
}

func (p *plugin) Name() string {
	return "compare"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.fmtPkg = p.NewImport("fmt")
	p.bytesPkg = p.NewImport("bytes")
	p.sortkeysPkg = p.NewImport("github.com/gogo/protobuf/sortkeys")
	p.protoPkg = p.NewImport("github.com/gogo/protobuf/proto")

	for _, msg := range file.Messages() {
		if msg.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		if gogoproto.HasCompare(file.FileDescriptorProto, msg.DescriptorProto) {
			p.generateMessage(file, msg)
		}
	}
}

func (p *plugin) generateNullableField(fieldname string) {
	p.P(`if this.`, fieldname, ` != nil && that1.`, fieldname, ` != nil {`)
	p.In()
	p.P(`if *this.`, fieldname, ` != *that1.`, fieldname, `{`)
	p.In()
	p.P(`if *this.`, fieldname, ` < *that1.`, fieldname, `{`)
	p.In()
	p.P(`return -1`)
	p.Out()
	p.P(`}`)
	p.P(`return 1`)
	p.Out()
	p.P(`}`)
	p.Out()
	p.P(`} else if this.`, fieldname, ` != nil {`)
	p.In()
	p.P(`return 1`)
	p.Out()
	p.P(`} else if that1.`, fieldname, ` != nil {`)
	p.In()
	p.P(`return -1`)
	p.Out()
	p.P(`}`)
}

func (p *plugin) generateMsgNullAndTypeCheck(ccTypeName string) {
	p.P(`if that == nil {`)
	p.In()
	p.P(`if this == nil {`)
	p.In()
	p.P(`return 0`)
	p.Out()
	p.P(`}`)
	p.P(`return 1`)
	p.Out()
	p.P(`}`)
	p.P(``)
	p.P(`that1, ok := that.(*`, ccTypeName, `)`)
	p.P(`if !ok {`)
	p.In()
	p.P(`that2, ok := that.(`, ccTypeName, `)`)
	p.P(`if ok {`)
	p.In()
	p.P(`that1 = &that2`)
	p.Out()
	p.P(`} else {`)
	p.In()
	p.P(`return 1`)
	p.Out()
	p.P(`}`)
	p.Out()
	p.P(`}`)
	p.P(`if that1 == nil {`)
	p.In()
	p.P(`if this == nil {`)
	p.In()
	p.P(`return 0`)
	p.Out()
	p.P(`}`)
	p.P(`return 1`)
	p.Out()
	p.P(`} else if this == nil {`)
	p.In()
	p.P(`return -1`)
	p.Out()
	p.P(`}`)
}

func (p *plugin) generateField(file *generator.FileDescriptor, message *generator.Descriptor, field *descriptor.FieldDescriptorProto) {
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	fieldname := p.GetOneOfFieldName(message, field)
	repeated := field.IsRepeated()
	ctype := gogoproto.IsCustomType(field)
	nullable := gogoproto.IsNullable(field)
	// oneof := field.OneofIndex != nil
	if !repeated {
		if ctype {
			if nullable {
				p.P(`if that1.`, fieldname, ` == nil {`)
				p.In()
				p.P(`if this.`, fieldname, ` != nil {`)
				p.In()
				p.P(`return 1`)
				p.Out()
				p.P(`}`)
				p.Out()
				p.P(`} else if this.`, fieldname, ` == nil {`)
				p.In()
				p.P(`return -1`)
				p.Out()
				p.P(`} else if c := this.`, fieldname, `.Compare(*that1.`, fieldname, `); c != 0 {`)
			} else {
				p.P(`if c := this.`, fieldname, `.Compare(that1.`, fieldname, `); c != 0 {`)
			}
			p.In()
			p.P(`return c`)
			p.Out()
			p.P(`}`)
		} else {
			if field.IsMessage() || p.IsGroup(field) {
				if nullable {
					p.P(`if c := this.`, fieldname, `.Compare(that1.`, fieldname, `); c != 0 {`)
				} else {
					p.P(`if c := this.`, fieldname, `.Compare(&that1.`, fieldname, `); c != 0 {`)
				}
				p.In()
				p.P(`return c`)
				p.Out()
				p.P(`}`)
			} else if field.IsBytes() {
				p.P(`if c := `, p.bytesPkg.Use(), `.Compare(this.`, fieldname, `, that1.`, fieldname, `); c != 0 {`)
				p.In()
				p.P(`return c`)
				p.Out()
				p.P(`}`)
			} else if field.IsString() {
				if nullable && !proto3 {
					p.generateNullableField(fieldname)
				} else {
					p.P(`if this.`, fieldname, ` != that1.`, fieldname, `{`)
					p.In()
					p.P(`if this.`, fieldname, ` < that1.`, fieldname, `{`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
					p.P(`return 1`)
					p.Out()
					p.P(`}`)
				}
			} else if field.IsBool() {
				if nullable && !proto3 {
					p.P(`if this.`, fieldname, ` != nil && that1.`, fieldname, ` != nil {`)
					p.In()
					p.P(`if *this.`, fieldname, ` != *that1.`, fieldname, `{`)
					p.In()
					p.P(`if !*this.`, fieldname, ` {`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
					p.P(`return 1`)
					p.Out()
					p.P(`}`)
					p.Out()
					p.P(`} else if this.`, fieldname, ` != nil {`)
					p.In()
					p.P(`return 1`)
					p.Out()
					p.P(`} else if that1.`, fieldname, ` != nil {`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
				} else {
					p.P(`if this.`, fieldname, ` != that1.`, fieldname, `{`)
					p.In()
					p.P(`if !this.`, fieldname, ` {`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
					p.P(`return 1`)
					p.Out()
					p.P(`}`)
				}
			} else {
				if nullable && !proto3 {
					p.generateNullableField(fieldname)
				} else {
					p.P(`if this.`, fieldname, ` != that1.`, fieldname, `{`)
					p.In()
					p.P(`if this.`, fieldname, ` < that1.`, fieldname, `{`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
					p.P(`return 1`)
					p.Out()
					p.P(`}`)
				}
			}
		}
	} else {
		p.P(`if len(this.`, fieldname, `) != len(that1.`, fieldname, `) {`)
		p.In()
		p.P(`if len(this.`, fieldname, `) < len(that1.`, fieldname, `) {`)
		p.In()
		p.P(`return -1`)
		p.Out()
		p.P(`}`)
		p.P(`return 1`)
		p.Out()
		p.P(`}`)
		p.P(`for i := range this.`, fieldname, ` {`)
		p.In()
		if ctype {
			p.P(`if c := this.`, fieldname, `[i].Compare(that1.`, fieldname, `[i]); c != 0 {`)
			p.In()
			p.P(`return c`)
			p.Out()
			p.P(`}`)
		} else {
			if p.IsMap(field) {
				m := p.GoMapType(nil, field)
				valuegoTyp, _ := p.GoType(nil, m.ValueField)
				valuegoAliasTyp, _ := p.GoType(nil, m.ValueAliasField)
				nullable, valuegoTyp, valuegoAliasTyp = generator.GoMapValueTypes(field, m.ValueField, valuegoTyp, valuegoAliasTyp)

				mapValue := m.ValueAliasField
				if mapValue.IsMessage() || p.IsGroup(mapValue) {
					if nullable && valuegoTyp == valuegoAliasTyp {
						p.P(`if c := this.`, fieldname, `[i].Compare(that1.`, fieldname, `[i]); c != 0 {`)
					} else {
						// Compare() has a pointer receiver, but map value is a value type
						a := `this.` + fieldname + `[i]`
						b := `that1.` + fieldname + `[i]`
						if valuegoTyp != valuegoAliasTyp {
							// cast back to the type that has the generated methods on it
							a = `(` + valuegoTyp + `)(` + a + `)`
							b = `(` + valuegoTyp + `)(` + b + `)`
						}
						p.P(`a := `, a)
						p.P(`b := `, b)
						if nullable {
							p.P(`if c := a.Compare(b); c != 0 {`)
						} else {
							p.P(`if c := (&a).Compare(&b); c != 0 {`)
						}
					}
					p.In()
					p.P(`return c`)
					p.Out()
					p.P(`}`)
				} else if mapValue.IsBytes() {
					p.P(`if c := `, p.bytesPkg.Use(), `.Compare(this.`, fieldname, `[i], that1.`, fieldname, `[i]); c != 0 {`)
					p.In()
					p.P(`return c`)
					p.Out()
					p.P(`}`)
				} else if mapValue.IsString() {
					p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
					p.In()
					p.P(`if this.`, fieldname, `[i] < that1.`, fieldname, `[i] {`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
					p.P(`return 1`)
					p.Out()
					p.P(`}`)
				} else {
					p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
					p.In()
					p.P(`if this.`, fieldname, `[i] < that1.`, fieldname, `[i] {`)
					p.In()
					p.P(`return -1`)
					p.Out()
					p.P(`}`)
					p.P(`return 1`)
					p.Out()
					p.P(`}`)
				}
			} else if field.IsMessage() || p.IsGroup(field) {
				if nullable {
					p.P(`if c := this.`, fieldname, `[i].Compare(that1.`, fieldname, `[i]); c != 0 {`)
					p.In()
					p.P(`return c`)
					p.Out()
					p.P(`}`)
				} else {
					p.P(`if c := this.`, fieldname, `[i].Compare(&that1.`, fieldname, `[i]); c != 0 {`)
					p.In()
					p.P(`return c`)
					p.Out()
					p.P(`}`)
				}
			} else if field.IsBytes() {
				p.P(`if c := `, p.bytesPkg.Use(), `.Compare(this.`, fieldname, `[i], that1.`, fieldname, `[i]); c != 0 {`)
				p.In()
				p.P(`return c`)
				p.Out()
				p.P(`}`)
			} else if field.IsString() {
				p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
				p.In()
				p.P(`if this.`, fieldname, `[i] < that1.`, fieldname, `[i] {`)
				p.In()
				p.P(`return -1`)
				p.Out()
				p.P(`}`)
				p.P(`return 1`)
				p.Out()
				p.P(`}`)
			} else if field.IsBool() {
				p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
				p.In()
				p.P(`if !this.`, fieldname, `[i] {`)
				p.In()
				p.P(`return -1`)
				p.Out()
				p.P(`}`)
				p.P(`return 1`)
				p.Out()
				p.P(`}`)
			} else {
				p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
				p.In()
				p.P(`if this.`, fieldname, `[i] < that1.`, fieldname, `[i] {`)
				p.In()
				p.P(`return -1`)
				p.Out()
				p.P(`}`)
				p.P(`return 1`)
				p.Out()
				p.P(`}`)
			}
		}
		p.Out()
		p.P(`}`)
	}
}

func (p *plugin) generateMessage(file *generator.FileDescriptor, message *generator.Descriptor) {
	ccTypeName := generator.CamelCaseSlice(message.TypeName())
	p.P(`func (this *`, ccTypeName, `) Compare(that interface{}) int {`)
	p.In()
	p.generateMsgNullAndTypeCheck(ccTypeName)
	oneofs := make(map[string]struct{})

	for _, field := range message.Field {
		oneof := field.OneofIndex != nil
		if oneof {
			fieldname := p.GetFieldName(message, field)
			if _, ok := oneofs[fieldname]; ok {
				continue
			} else {
				oneofs[fieldname] = struct{}{}
			}
			p.P(`if that1.`, fieldname, ` == nil {`)
			p.In()
			p.P(`if this.`, fieldname, ` != nil {`)
			p.In()
			p.P(`return 1`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`} else if this.`, fieldname, ` == nil {`)
			p.In()
			p.P(`return -1`)
			p.Out()
			p.P(`} else if c := this.`, fieldname, `.Compare(that1.`, fieldname, `); c != 0 {`)
			p.In()
			p.P(`return c`)
			p.Out()
			p.P(`}`)
		} else {
			p.generateField(file, message, field)
		}
	}
	if message.DescriptorProto.HasExtension() {
		if gogoproto.HasExtensionsMap(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`thismap := `, p.protoPkg.Use(), `.GetUnsafeExtensionsMap(this)`)
			p.P(`thatmap := `, p.protoPkg.Use(), `.GetUnsafeExtensionsMap(that1)`)
			p.P(`extkeys := make([]int32, 0, len(thismap)+len(thatmap))`)
			p.P(`for k, _ := range thismap {`)
			p.In()
			p.P(`extkeys = append(extkeys, k)`)
			p.Out()
			p.P(`}`)
			p.P(`for k, _ := range thatmap {`)
			p.In()
			p.P(`if _, ok := thismap[k]; !ok {`)
			p.In()
			p.P(`extkeys = append(extkeys, k)`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)
			p.P(p.sortkeysPkg.Use(), `.Int32s(extkeys)`)
			p.P(`for _, k := range extkeys {`)
			p.In()
			p.P(`if v, ok := thismap[k]; ok {`)
			p.In()
			p.P(`if v2, ok := thatmap[k]; ok {`)
			p.In()
			p.P(`if c := v.Compare(&v2); c != 0 {`)
			p.In()
			p.P(`return c`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`} else  {`)
			p.In()
			p.P(`return 1`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`} else {`)
			p.In()
			p.P(`return -1`)
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)
		} else {
			fieldname := "XXX_extensions"
			p.P(`if c := `, p.bytesPkg.Use(), `.Compare(this.`, fieldname, `, that1.`, fieldname, `); c != 0 {`)
			p.In()
			p.P(`return c`)
			p.Out()
			p.P(`}`)
		}
	}
	if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
		fieldname := "XXX_unrecognized"
		p.P(`if c := `, p.bytesPkg.Use(), `.Compare(this.`, fieldname, `, that1.`, fieldname, `); c != 0 {`)
		p.In()
		p.P(`return c`)
		p.Out()
		p.P(`}`)
	}
	p.P(`return 0`)
	p.Out()
	p.P(`}`)

	//Generate Compare methods for oneof fields
	m := proto.Clone(message.DescriptorProto).(*descriptor.DescriptorProto)
	for _, field := range m.Field {
		oneof := field.OneofIndex != nil
		if !oneof {
			continue
		}
		ccTypeName := p.OneOfTypeName(message, field)
		p.P(`func (this *`, ccTypeName, `) Compare(that interface{}) int {`)
		p.In()

		p.generateMsgNullAndTypeCheck(ccTypeName)
		vanity.TurnOffNullableForNativeTypesWithoutDefaultsOnly(field)
		p.generateField(file, message, field)

		p.P(`return 0`)
		p.Out()
		p.P(`}`)
	}
}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
