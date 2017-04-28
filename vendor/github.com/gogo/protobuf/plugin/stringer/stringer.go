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
The stringer plugin generates a String method for each message.

It is enabled by the following extensions:

  - stringer
  - stringer_all

The stringer plugin also generates a test given it is enabled using one of the following extensions:

  - testgen
  - testgen_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  option (gogoproto.goproto_stringer_all) = false;
  option (gogoproto.stringer_all) =  true;

  message A {
	optional string Description = 1 [(gogoproto.nullable) = false];
	optional int64 Number = 2 [(gogoproto.nullable) = false];
	optional bytes Id = 3 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uuid", (gogoproto.nullable) = false];
  }

given to the stringer stringer, will generate the following code:

  func (this *A) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&A{`,
		`Description:` + fmt.Sprintf("%v", this.Description) + `,`,
		`Number:` + fmt.Sprintf("%v", this.Number) + `,`,
		`Id:` + fmt.Sprintf("%v", this.Id) + `,`,
		`XXX_unrecognized:` + fmt.Sprintf("%v", this.XXX_unrecognized) + `,`,
		`}`,
	}, "")
	return s
  }

and the following test code:

	func TestAStringer(t *testing4.T) {
		popr := math_rand4.New(math_rand4.NewSource(time4.Now().UnixNano()))
		p := NewPopulatedA(popr, false)
		s1 := p.String()
		s2 := fmt1.Sprintf("%v", p)
		if s1 != s2 {
			t.Fatalf("String want %v got %v", s1, s2)
		}
	}

Typically fmt.Printf("%v") will stop to print when it reaches a pointer and
not print their values, while the generated String method will always print all values, recursively.

*/
package stringer

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
	"strings"
)

type stringer struct {
	*generator.Generator
	generator.PluginImports
	atleastOne bool
	localName  string
}

func NewStringer() *stringer {
	return &stringer{}
}

func (p *stringer) Name() string {
	return "stringer"
}

func (p *stringer) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *stringer) Generate(file *generator.FileDescriptor) {
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.atleastOne = false

	p.localName = generator.FileName(file)

	fmtPkg := p.NewImport("fmt")
	stringsPkg := p.NewImport("strings")
	reflectPkg := p.NewImport("reflect")
	sortKeysPkg := p.NewImport("github.com/gogo/protobuf/sortkeys")
	protoPkg := p.NewImport("github.com/gogo/protobuf/proto")
	for _, message := range file.Messages() {
		if !gogoproto.IsStringer(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if gogoproto.EnabledGoStringer(file.FileDescriptorProto, message.DescriptorProto) {
			panic("old string method needs to be disabled, please use gogoproto.goproto_stringer or gogoproto.goproto_stringer_all and set it to false")
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		p.atleastOne = true
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		p.P(`func (this *`, ccTypeName, `) String() string {`)
		p.In()
		p.P(`if this == nil {`)
		p.In()
		p.P(`return "nil"`)
		p.Out()
		p.P(`}`)
		for _, field := range message.Field {
			if !p.IsMap(field) {
				continue
			}
			fieldname := p.GetFieldName(message, field)

			m := p.GoMapType(nil, field)
			mapgoTyp, keyField, keyAliasField := m.GoType, m.KeyField, m.KeyAliasField
			keysName := `keysFor` + fieldname
			keygoTyp, _ := p.GoType(nil, keyField)
			keygoTyp = strings.Replace(keygoTyp, "*", "", 1)
			keygoAliasTyp, _ := p.GoType(nil, keyAliasField)
			keygoAliasTyp = strings.Replace(keygoAliasTyp, "*", "", 1)
			keyCapTyp := generator.CamelCase(keygoTyp)
			p.P(keysName, ` := make([]`, keygoTyp, `, 0, len(this.`, fieldname, `))`)
			p.P(`for k, _ := range this.`, fieldname, ` {`)
			p.In()
			if keygoAliasTyp == keygoTyp {
				p.P(keysName, ` = append(`, keysName, `, k)`)
			} else {
				p.P(keysName, ` = append(`, keysName, `, `, keygoTyp, `(k))`)
			}
			p.Out()
			p.P(`}`)
			p.P(sortKeysPkg.Use(), `.`, keyCapTyp, `s(`, keysName, `)`)
			mapName := `mapStringFor` + fieldname
			p.P(mapName, ` := "`, mapgoTyp, `{"`)
			p.P(`for _, k := range `, keysName, ` {`)
			p.In()
			if keygoAliasTyp == keygoTyp {
				p.P(mapName, ` += fmt.Sprintf("%v: %v,", k, this.`, fieldname, `[k])`)
			} else {
				p.P(mapName, ` += fmt.Sprintf("%v: %v,", k, this.`, fieldname, `[`, keygoAliasTyp, `(k)])`)
			}
			p.Out()
			p.P(`}`)
			p.P(mapName, ` += "}"`)
		}
		p.P("s := ", stringsPkg.Use(), ".Join([]string{`&", ccTypeName, "{`,")
		oneofs := make(map[string]struct{})
		for _, field := range message.Field {
			nullable := gogoproto.IsNullable(field)
			repeated := field.IsRepeated()
			fieldname := p.GetFieldName(message, field)
			oneof := field.OneofIndex != nil
			if oneof {
				if _, ok := oneofs[fieldname]; ok {
					continue
				} else {
					oneofs[fieldname] = struct{}{}
				}
				p.P("`", fieldname, ":`", ` + `, fmtPkg.Use(), `.Sprintf("%v", this.`, fieldname, ") + `,", "`,")
			} else if p.IsMap(field) {
				mapName := `mapStringFor` + fieldname
				p.P("`", fieldname, ":`", ` + `, mapName, " + `,", "`,")
			} else if (field.IsMessage() && !gogoproto.IsCustomType(field)) || p.IsGroup(field) {
				desc := p.ObjectNamed(field.GetTypeName())
				msgname := p.TypeName(desc)
				msgnames := strings.Split(msgname, ".")
				typeName := msgnames[len(msgnames)-1]
				if nullable {
					p.P("`", fieldname, ":`", ` + `, stringsPkg.Use(), `.Replace(`, fmtPkg.Use(), `.Sprintf("%v", this.`, fieldname, `), "`, typeName, `","`, msgname, `"`, ", 1) + `,", "`,")
				} else if repeated {
					p.P("`", fieldname, ":`", ` + `, stringsPkg.Use(), `.Replace(`, stringsPkg.Use(), `.Replace(`, fmtPkg.Use(), `.Sprintf("%v", this.`, fieldname, `), "`, typeName, `","`, msgname, `"`, ", 1),`&`,``,1) + `,", "`,")
				} else {
					p.P("`", fieldname, ":`", ` + `, stringsPkg.Use(), `.Replace(`, stringsPkg.Use(), `.Replace(this.`, fieldname, `.String(), "`, typeName, `","`, msgname, `"`, ", 1),`&`,``,1) + `,", "`,")
				}
			} else {
				if nullable && !repeated && !proto3 {
					p.P("`", fieldname, ":`", ` + valueToString`, p.localName, `(this.`, fieldname, ") + `,", "`,")
				} else {
					p.P("`", fieldname, ":`", ` + `, fmtPkg.Use(), `.Sprintf("%v", this.`, fieldname, ") + `,", "`,")
				}
			}
		}
		if message.DescriptorProto.HasExtension() {
			if gogoproto.HasExtensionsMap(file.FileDescriptorProto, message.DescriptorProto) {
				p.P("`XXX_InternalExtensions:` + ", protoPkg.Use(), ".StringFromInternalExtension(this) + `,`,")
			} else {
				p.P("`XXX_extensions:` + ", protoPkg.Use(), ".StringFromExtensionsBytes(this.XXX_extensions) + `,`,")
			}
		}
		if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
			p.P("`XXX_unrecognized:` + ", fmtPkg.Use(), `.Sprintf("%v", this.XXX_unrecognized) + `, "`,`,")
		}
		p.P("`}`,")
		p.P(`}`, `,""`, ")")
		p.P(`return s`)
		p.Out()
		p.P(`}`)

		//Generate String methods for oneof fields
		for _, field := range message.Field {
			oneof := field.OneofIndex != nil
			if !oneof {
				continue
			}
			ccTypeName := p.OneOfTypeName(message, field)
			p.P(`func (this *`, ccTypeName, `) String() string {`)
			p.In()
			p.P(`if this == nil {`)
			p.In()
			p.P(`return "nil"`)
			p.Out()
			p.P(`}`)
			p.P("s := ", stringsPkg.Use(), ".Join([]string{`&", ccTypeName, "{`,")
			fieldname := p.GetOneOfFieldName(message, field)
			if field.IsMessage() || p.IsGroup(field) {
				desc := p.ObjectNamed(field.GetTypeName())
				msgname := p.TypeName(desc)
				msgnames := strings.Split(msgname, ".")
				typeName := msgnames[len(msgnames)-1]
				p.P("`", fieldname, ":`", ` + `, stringsPkg.Use(), `.Replace(`, fmtPkg.Use(), `.Sprintf("%v", this.`, fieldname, `), "`, typeName, `","`, msgname, `"`, ", 1) + `,", "`,")
			} else {
				p.P("`", fieldname, ":`", ` + `, fmtPkg.Use(), `.Sprintf("%v", this.`, fieldname, ") + `,", "`,")
			}
			p.P("`}`,")
			p.P(`}`, `,""`, ")")
			p.P(`return s`)
			p.Out()
			p.P(`}`)
		}
	}

	if !p.atleastOne {
		return
	}

	p.P(`func valueToString`, p.localName, `(v interface{}) string {`)
	p.In()
	p.P(`rv := `, reflectPkg.Use(), `.ValueOf(v)`)
	p.P(`if rv.IsNil() {`)
	p.In()
	p.P(`return "nil"`)
	p.Out()
	p.P(`}`)
	p.P(`pv := `, reflectPkg.Use(), `.Indirect(rv).Interface()`)
	p.P(`return `, fmtPkg.Use(), `.Sprintf("*%v", pv)`)
	p.Out()
	p.P(`}`)

}

func init() {
	generator.RegisterPlugin(NewStringer())
}
