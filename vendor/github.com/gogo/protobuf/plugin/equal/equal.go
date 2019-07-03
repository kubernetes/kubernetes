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
The equal plugin generates an Equal and a VerboseEqual method for each message.
These equal methods are quite obvious.
The only difference is that VerboseEqual returns a non nil error if it is not equal.
This error contains more detail on exactly which part of the message was not equal to the other message.
The idea is that this is useful for debugging.

Equal is enabled using the following extensions:

  - equal
  - equal_all

While VerboseEqual is enable dusing the following extensions:

  - verbose_equal
  - verbose_equal_all

The equal plugin also generates a test given it is enabled using one of the following extensions:

  - testgen
  - testgen_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  option (gogoproto.equal_all) = true;
  option (gogoproto.verbose_equal_all) = true;

  message B {
	optional A A = 1 [(gogoproto.nullable) = false, (gogoproto.embed) = true];
	repeated bytes G = 2 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uint128", (gogoproto.nullable) = false];
  }

given to the equal plugin, will generate the following code:

	func (this *B) VerboseEqual(that interface{}) error {
		if that == nil {
			if this == nil {
				return nil
			}
			return fmt2.Errorf("that == nil && this != nil")
		}

		that1, ok := that.(*B)
		if !ok {
			return fmt2.Errorf("that is not of type *B")
		}
		if that1 == nil {
			if this == nil {
				return nil
			}
			return fmt2.Errorf("that is type *B but is nil && this != nil")
		} else if this == nil {
			return fmt2.Errorf("that is type *B but is not nil && this == nil")
		}
		if !this.A.Equal(&that1.A) {
			return fmt2.Errorf("A this(%v) Not Equal that(%v)", this.A, that1.A)
		}
		if len(this.G) != len(that1.G) {
			return fmt2.Errorf("G this(%v) Not Equal that(%v)", len(this.G), len(that1.G))
		}
		for i := range this.G {
			if !this.G[i].Equal(that1.G[i]) {
				return fmt2.Errorf("G this[%v](%v) Not Equal that[%v](%v)", i, this.G[i], i, that1.G[i])
			}
		}
		if !bytes.Equal(this.XXX_unrecognized, that1.XXX_unrecognized) {
			return fmt2.Errorf("XXX_unrecognized this(%v) Not Equal that(%v)", this.XXX_unrecognized, that1.XXX_unrecognized)
		}
		return nil
	}

	func (this *B) Equal(that interface{}) bool {
		if that == nil {
			return this == nil
		}

		that1, ok := that.(*B)
		if !ok {
			return false
		}
		if that1 == nil {
			return this == nil
		} else if this == nil {
			return false
		}
		if !this.A.Equal(&that1.A) {
			return false
		}
		if len(this.G) != len(that1.G) {
			return false
		}
		for i := range this.G {
			if !this.G[i].Equal(that1.G[i]) {
				return false
			}
		}
		if !bytes.Equal(this.XXX_unrecognized, that1.XXX_unrecognized) {
			return false
		}
		return true
	}

and the following test code:

	func TestBVerboseEqual(t *testing8.T) {
		popr := math_rand8.New(math_rand8.NewSource(time8.Now().UnixNano()))
		p := NewPopulatedB(popr, false)
		dAtA, err := github_com_gogo_protobuf_proto2.Marshal(p)
		if err != nil {
			panic(err)
		}
		msg := &B{}
		if err := github_com_gogo_protobuf_proto2.Unmarshal(dAtA, msg); err != nil {
			panic(err)
		}
		if err := p.VerboseEqual(msg); err != nil {
			t.Fatalf("%#v !VerboseEqual %#v, since %v", msg, p, err)
	}

*/
package equal

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
	fmtPkg   generator.Single
	bytesPkg generator.Single
	protoPkg generator.Single
}

func NewPlugin() *plugin {
	return &plugin{}
}

func (p *plugin) Name() string {
	return "equal"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.fmtPkg = p.NewImport("fmt")
	p.bytesPkg = p.NewImport("bytes")
	p.protoPkg = p.NewImport("github.com/gogo/protobuf/proto")

	for _, msg := range file.Messages() {
		if msg.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		if gogoproto.HasVerboseEqual(file.FileDescriptorProto, msg.DescriptorProto) {
			p.generateMessage(file, msg, true)
		}
		if gogoproto.HasEqual(file.FileDescriptorProto, msg.DescriptorProto) {
			p.generateMessage(file, msg, false)
		}
	}
}

func (p *plugin) generateNullableField(fieldname string, verbose bool) {
	p.P(`if this.`, fieldname, ` != nil && that1.`, fieldname, ` != nil {`)
	p.In()
	p.P(`if *this.`, fieldname, ` != *that1.`, fieldname, `{`)
	p.In()
	if verbose {
		p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", *this.`, fieldname, `, *that1.`, fieldname, `)`)
	} else {
		p.P(`return false`)
	}
	p.Out()
	p.P(`}`)
	p.Out()
	p.P(`} else if this.`, fieldname, ` != nil {`)
	p.In()
	if verbose {
		p.P(`return `, p.fmtPkg.Use(), `.Errorf("this.`, fieldname, ` == nil && that.`, fieldname, ` != nil")`)
	} else {
		p.P(`return false`)
	}
	p.Out()
	p.P(`} else if that1.`, fieldname, ` != nil {`)
}

func (p *plugin) generateMsgNullAndTypeCheck(ccTypeName string, verbose bool) {
	p.P(`if that == nil {`)
	p.In()
	if verbose {
		p.P(`if this == nil {`)
		p.In()
		p.P(`return nil`)
		p.Out()
		p.P(`}`)
		p.P(`return `, p.fmtPkg.Use(), `.Errorf("that == nil && this != nil")`)
	} else {
		p.P(`return this == nil`)
	}
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
	if verbose {
		p.P(`return `, p.fmtPkg.Use(), `.Errorf("that is not of type *`, ccTypeName, `")`)
	} else {
		p.P(`return false`)
	}
	p.Out()
	p.P(`}`)
	p.Out()
	p.P(`}`)
	p.P(`if that1 == nil {`)
	p.In()
	if verbose {
		p.P(`if this == nil {`)
		p.In()
		p.P(`return nil`)
		p.Out()
		p.P(`}`)
		p.P(`return `, p.fmtPkg.Use(), `.Errorf("that is type *`, ccTypeName, ` but is nil && this != nil")`)
	} else {
		p.P(`return this == nil`)
	}
	p.Out()
	p.P(`} else if this == nil {`)
	p.In()
	if verbose {
		p.P(`return `, p.fmtPkg.Use(), `.Errorf("that is type *`, ccTypeName, ` but is not nil && this == nil")`)
	} else {
		p.P(`return false`)
	}
	p.Out()
	p.P(`}`)
}

func (p *plugin) generateField(file *generator.FileDescriptor, message *generator.Descriptor, field *descriptor.FieldDescriptorProto, verbose bool) {
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	fieldname := p.GetOneOfFieldName(message, field)
	repeated := field.IsRepeated()
	ctype := gogoproto.IsCustomType(field)
	nullable := gogoproto.IsNullable(field)
	isDuration := gogoproto.IsStdDuration(field)
	isTimestamp := gogoproto.IsStdTime(field)
	// oneof := field.OneofIndex != nil
	if !repeated {
		if ctype || isTimestamp {
			if nullable {
				p.P(`if that1.`, fieldname, ` == nil {`)
				p.In()
				p.P(`if this.`, fieldname, ` != nil {`)
				p.In()
				if verbose {
					p.P(`return `, p.fmtPkg.Use(), `.Errorf("this.`, fieldname, ` != nil && that1.`, fieldname, ` == nil")`)
				} else {
					p.P(`return false`)
				}
				p.Out()
				p.P(`}`)
				p.Out()
				p.P(`} else if !this.`, fieldname, `.Equal(*that1.`, fieldname, `) {`)
			} else {
				p.P(`if !this.`, fieldname, `.Equal(that1.`, fieldname, `) {`)
			}
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", this.`, fieldname, `, that1.`, fieldname, `)`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
		} else if isDuration {
			if nullable {
				p.generateNullableField(fieldname, verbose)
			} else {
				p.P(`if this.`, fieldname, ` != that1.`, fieldname, `{`)
			}
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", this.`, fieldname, `, that1.`, fieldname, `)`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
		} else {
			if field.IsMessage() || p.IsGroup(field) {
				if nullable {
					p.P(`if !this.`, fieldname, `.Equal(that1.`, fieldname, `) {`)
				} else {
					p.P(`if !this.`, fieldname, `.Equal(&that1.`, fieldname, `) {`)
				}
			} else if field.IsBytes() {
				p.P(`if !`, p.bytesPkg.Use(), `.Equal(this.`, fieldname, `, that1.`, fieldname, `) {`)
			} else if field.IsString() {
				if nullable && !proto3 {
					p.generateNullableField(fieldname, verbose)
				} else {
					p.P(`if this.`, fieldname, ` != that1.`, fieldname, `{`)
				}
			} else {
				if nullable && !proto3 {
					p.generateNullableField(fieldname, verbose)
				} else {
					p.P(`if this.`, fieldname, ` != that1.`, fieldname, `{`)
				}
			}
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", this.`, fieldname, `, that1.`, fieldname, `)`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
		}
	} else {
		p.P(`if len(this.`, fieldname, `) != len(that1.`, fieldname, `) {`)
		p.In()
		if verbose {
			p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", len(this.`, fieldname, `), len(that1.`, fieldname, `))`)
		} else {
			p.P(`return false`)
		}
		p.Out()
		p.P(`}`)
		p.P(`for i := range this.`, fieldname, ` {`)
		p.In()
		if ctype && !p.IsMap(field) {
			p.P(`if !this.`, fieldname, `[i].Equal(that1.`, fieldname, `[i]) {`)
		} else if isTimestamp {
			if nullable {
				p.P(`if !this.`, fieldname, `[i].Equal(*that1.`, fieldname, `[i]) {`)
			} else {
				p.P(`if !this.`, fieldname, `[i].Equal(that1.`, fieldname, `[i]) {`)
			}
		} else if isDuration {
			if nullable {
				p.P(`if dthis, dthat := this.`, fieldname, `[i], that1.`, fieldname, `[i]; (dthis != nil && dthat != nil && *dthis != *dthat) || (dthis != nil && dthat == nil) || (dthis == nil && dthat != nil)  {`)
			} else {
				p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
			}
		} else {
			if p.IsMap(field) {
				m := p.GoMapType(nil, field)
				valuegoTyp, _ := p.GoType(nil, m.ValueField)
				valuegoAliasTyp, _ := p.GoType(nil, m.ValueAliasField)
				nullable, valuegoTyp, valuegoAliasTyp = generator.GoMapValueTypes(field, m.ValueField, valuegoTyp, valuegoAliasTyp)

				mapValue := m.ValueAliasField
				if mapValue.IsMessage() || p.IsGroup(mapValue) {
					if nullable && valuegoTyp == valuegoAliasTyp {
						p.P(`if !this.`, fieldname, `[i].Equal(that1.`, fieldname, `[i]) {`)
					} else {
						// Equal() has a pointer receiver, but map value is a value type
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
							p.P(`if !a.Equal(b) {`)
						} else {
							p.P(`if !(&a).Equal(&b) {`)
						}
					}
				} else if mapValue.IsBytes() {
					if ctype {
						if nullable {
							p.P(`if !this.`, fieldname, `[i].Equal(*that1.`, fieldname, `[i]) { //nullable`)
						} else {
							p.P(`if !this.`, fieldname, `[i].Equal(that1.`, fieldname, `[i]) { //not nullable`)
						}
					} else {
						p.P(`if !`, p.bytesPkg.Use(), `.Equal(this.`, fieldname, `[i], that1.`, fieldname, `[i]) {`)
					}
				} else if mapValue.IsString() {
					p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
				} else {
					p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
				}
			} else if field.IsMessage() || p.IsGroup(field) {
				if nullable {
					p.P(`if !this.`, fieldname, `[i].Equal(that1.`, fieldname, `[i]) {`)
				} else {
					p.P(`if !this.`, fieldname, `[i].Equal(&that1.`, fieldname, `[i]) {`)
				}
			} else if field.IsBytes() {
				p.P(`if !`, p.bytesPkg.Use(), `.Equal(this.`, fieldname, `[i], that1.`, fieldname, `[i]) {`)
			} else if field.IsString() {
				p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
			} else {
				p.P(`if this.`, fieldname, `[i] != that1.`, fieldname, `[i] {`)
			}
		}
		p.In()
		if verbose {
			p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this[%v](%v) Not Equal that[%v](%v)", i, this.`, fieldname, `[i], i, that1.`, fieldname, `[i])`)
		} else {
			p.P(`return false`)
		}
		p.Out()
		p.P(`}`)
		p.Out()
		p.P(`}`)
	}
}

func (p *plugin) generateMessage(file *generator.FileDescriptor, message *generator.Descriptor, verbose bool) {
	ccTypeName := generator.CamelCaseSlice(message.TypeName())
	if verbose {
		p.P(`func (this *`, ccTypeName, `) VerboseEqual(that interface{}) error {`)
	} else {
		p.P(`func (this *`, ccTypeName, `) Equal(that interface{}) bool {`)
	}
	p.In()
	p.generateMsgNullAndTypeCheck(ccTypeName, verbose)
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
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("this.`, fieldname, ` != nil && that1.`, fieldname, ` == nil")`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`} else if this.`, fieldname, ` == nil {`)
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("this.`, fieldname, ` == nil && that1.`, fieldname, ` != nil")`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			if verbose {
				p.P(`} else if err := this.`, fieldname, `.VerboseEqual(that1.`, fieldname, `); err != nil {`)
			} else {
				p.P(`} else if !this.`, fieldname, `.Equal(that1.`, fieldname, `) {`)
			}
			p.In()
			if verbose {
				p.P(`return err`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
		} else {
			p.generateField(file, message, field, verbose)
		}
	}
	if message.DescriptorProto.HasExtension() {
		if gogoproto.HasExtensionsMap(file.FileDescriptorProto, message.DescriptorProto) {
			fieldname := "XXX_InternalExtensions"
			p.P(`thismap := `, p.protoPkg.Use(), `.GetUnsafeExtensionsMap(this)`)
			p.P(`thatmap := `, p.protoPkg.Use(), `.GetUnsafeExtensionsMap(that1)`)
			p.P(`for k, v := range thismap {`)
			p.In()
			p.P(`if v2, ok := thatmap[k]; ok {`)
			p.In()
			p.P(`if !v.Equal(&v2) {`)
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this[%v](%v) Not Equal that[%v](%v)", k, thismap[k], k, thatmap[k])`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`} else  {`)
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, `[%v] Not In that", k)`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)

			p.P(`for k, _ := range thatmap {`)
			p.In()
			p.P(`if _, ok := thismap[k]; !ok {`)
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, `[%v] Not In this", k)`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
			p.Out()
			p.P(`}`)
		} else {
			fieldname := "XXX_extensions"
			p.P(`if !`, p.bytesPkg.Use(), `.Equal(this.`, fieldname, `, that1.`, fieldname, `) {`)
			p.In()
			if verbose {
				p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", this.`, fieldname, `, that1.`, fieldname, `)`)
			} else {
				p.P(`return false`)
			}
			p.Out()
			p.P(`}`)
		}
	}
	if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
		fieldname := "XXX_unrecognized"
		p.P(`if !`, p.bytesPkg.Use(), `.Equal(this.`, fieldname, `, that1.`, fieldname, `) {`)
		p.In()
		if verbose {
			p.P(`return `, p.fmtPkg.Use(), `.Errorf("`, fieldname, ` this(%v) Not Equal that(%v)", this.`, fieldname, `, that1.`, fieldname, `)`)
		} else {
			p.P(`return false`)
		}
		p.Out()
		p.P(`}`)
	}
	if verbose {
		p.P(`return nil`)
	} else {
		p.P(`return true`)
	}
	p.Out()
	p.P(`}`)

	//Generate Equal methods for oneof fields
	m := proto.Clone(message.DescriptorProto).(*descriptor.DescriptorProto)
	for _, field := range m.Field {
		oneof := field.OneofIndex != nil
		if !oneof {
			continue
		}
		ccTypeName := p.OneOfTypeName(message, field)
		if verbose {
			p.P(`func (this *`, ccTypeName, `) VerboseEqual(that interface{}) error {`)
		} else {
			p.P(`func (this *`, ccTypeName, `) Equal(that interface{}) bool {`)
		}
		p.In()

		p.generateMsgNullAndTypeCheck(ccTypeName, verbose)
		vanity.TurnOffNullableForNativeTypes(field)
		p.generateField(file, message, field, verbose)

		if verbose {
			p.P(`return nil`)
		} else {
			p.P(`return true`)
		}
		p.Out()
		p.P(`}`)
	}
}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
