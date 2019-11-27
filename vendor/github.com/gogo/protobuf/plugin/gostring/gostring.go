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
The gostring plugin generates a GoString method for each message.
The GoString method is called whenever you use a fmt.Printf as such:

  fmt.Printf("%#v", mymessage)

or whenever you actually call GoString()
The output produced by the GoString method can be copied from the output into code and used to set a variable.
It is totally valid Go Code and is populated exactly as the struct that was printed out.

It is enabled by the following extensions:

  - gostring
  - gostring_all

The gostring plugin also generates a test given it is enabled using one of the following extensions:

  - testgen
  - testgen_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  option (gogoproto.gostring_all) = true;

  message A {
	optional string Description = 1 [(gogoproto.nullable) = false];
	optional int64 Number = 2 [(gogoproto.nullable) = false];
	optional bytes Id = 3 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uuid", (gogoproto.nullable) = false];
  }

given to the gostring plugin, will generate the following code:

  func (this *A) GoString() string {
	if this == nil {
		return "nil"
	}
	s := strings1.Join([]string{`&test.A{` + `Description:` + fmt1.Sprintf("%#v", this.Description), `Number:` + fmt1.Sprintf("%#v", this.Number), `Id:` + fmt1.Sprintf("%#v", this.Id), `XXX_unrecognized:` + fmt1.Sprintf("%#v", this.XXX_unrecognized) + `}`}, ", ")
	return s
  }

and the following test code:

	func TestAGoString(t *testing6.T) {
		popr := math_rand6.New(math_rand6.NewSource(time6.Now().UnixNano()))
		p := NewPopulatedA(popr, false)
		s1 := p.GoString()
		s2 := fmt2.Sprintf("%#v", p)
		if s1 != s2 {
			t.Fatalf("GoString want %v got %v", s1, s2)
		}
		_, err := go_parser.ParseExpr(s1)
		if err != nil {
			panic(err)
		}
	}

Typically fmt.Printf("%#v") will stop to print when it reaches a pointer and
not print their values, while the generated GoString method will always print all values, recursively.

*/
package gostring

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type gostring struct {
	*generator.Generator
	generator.PluginImports
	atleastOne bool
	localName  string
	overwrite  bool
}

func NewGoString() *gostring {
	return &gostring{}
}

func (p *gostring) Name() string {
	return "gostring"
}

func (p *gostring) Overwrite() {
	p.overwrite = true
}

func (p *gostring) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *gostring) Generate(file *generator.FileDescriptor) {
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.atleastOne = false

	p.localName = generator.FileName(file)

	fmtPkg := p.NewImport("fmt")
	stringsPkg := p.NewImport("strings")
	protoPkg := p.NewImport("github.com/gogo/protobuf/proto")
	if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
		protoPkg = p.NewImport("github.com/golang/protobuf/proto")
	}
	sortPkg := p.NewImport("sort")
	strconvPkg := p.NewImport("strconv")
	reflectPkg := p.NewImport("reflect")
	sortKeysPkg := p.NewImport("github.com/gogo/protobuf/sortkeys")

	extensionToGoStringUsed := false
	for _, message := range file.Messages() {
		if !p.overwrite && !gogoproto.HasGoString(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		p.atleastOne = true
		packageName := file.GoPackageName()

		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		p.P(`func (this *`, ccTypeName, `) GoString() string {`)
		p.In()
		p.P(`if this == nil {`)
		p.In()
		p.P(`return "nil"`)
		p.Out()
		p.P(`}`)

		p.P(`s := make([]string, 0, `, strconv.Itoa(len(message.Field)+4), `)`)
		p.P(`s = append(s, "&`, packageName, ".", ccTypeName, `{")`)

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
				p.P(`if this.`, fieldname, ` != nil {`)
				p.In()
				p.P(`s = append(s, "`, fieldname, `: " + `, fmtPkg.Use(), `.Sprintf("%#v", this.`, fieldname, `) + ",\n")`)
				p.Out()
				p.P(`}`)
			} else if p.IsMap(field) {
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
					p.P(mapName, ` += fmt.Sprintf("%#v: %#v,", k, this.`, fieldname, `[k])`)
				} else {
					p.P(mapName, ` += fmt.Sprintf("%#v: %#v,", k, this.`, fieldname, `[`, keygoAliasTyp, `(k)])`)
				}
				p.Out()
				p.P(`}`)
				p.P(mapName, ` += "}"`)
				p.P(`if this.`, fieldname, ` != nil {`)
				p.In()
				p.P(`s = append(s, "`, fieldname, `: " + `, mapName, `+ ",\n")`)
				p.Out()
				p.P(`}`)
			} else if (field.IsMessage() && !gogoproto.IsCustomType(field) && !gogoproto.IsStdType(field)) || p.IsGroup(field) {
				if nullable || repeated {
					p.P(`if this.`, fieldname, ` != nil {`)
					p.In()
				}
				if nullable {
					p.P(`s = append(s, "`, fieldname, `: " + `, fmtPkg.Use(), `.Sprintf("%#v", this.`, fieldname, `) + ",\n")`)
				} else if repeated {
					if nullable {
						p.P(`s = append(s, "`, fieldname, `: " + `, fmtPkg.Use(), `.Sprintf("%#v", this.`, fieldname, `) + ",\n")`)
					} else {
						goTyp, _ := p.GoType(message, field)
						goTyp = strings.Replace(goTyp, "[]", "", 1)
						p.P("vs := make([]*", goTyp, ", len(this.", fieldname, "))")
						p.P("for i := range vs {")
						p.In()
						p.P("vs[i] = &this.", fieldname, "[i]")
						p.Out()
						p.P("}")
						p.P(`s = append(s, "`, fieldname, `: " + `, fmtPkg.Use(), `.Sprintf("%#v", vs) + ",\n")`)
					}
				} else {
					p.P(`s = append(s, "`, fieldname, `: " + `, stringsPkg.Use(), `.Replace(this.`, fieldname, `.GoString()`, ",`&`,``,1)", ` + ",\n")`)
				}
				if nullable || repeated {
					p.Out()
					p.P(`}`)
				}
			} else {
				if !proto3 && (nullable || repeated) {
					p.P(`if this.`, fieldname, ` != nil {`)
					p.In()
				}
				if field.IsEnum() {
					if nullable && !repeated && !proto3 {
						goTyp, _ := p.GoType(message, field)
						p.P(`s = append(s, "`, fieldname, `: " + valueToGoString`, p.localName, `(this.`, fieldname, `,"`, generator.GoTypeToName(goTyp), `"`, `) + ",\n")`)
					} else {
						p.P(`s = append(s, "`, fieldname, `: " + `, fmtPkg.Use(), `.Sprintf("%#v", this.`, fieldname, `) + ",\n")`)
					}
				} else {
					if nullable && !repeated && !proto3 {
						goTyp, _ := p.GoType(message, field)
						p.P(`s = append(s, "`, fieldname, `: " + valueToGoString`, p.localName, `(this.`, fieldname, `,"`, generator.GoTypeToName(goTyp), `"`, `) + ",\n")`)
					} else {
						p.P(`s = append(s, "`, fieldname, `: " + `, fmtPkg.Use(), `.Sprintf("%#v", this.`, fieldname, `) + ",\n")`)
					}
				}
				if !proto3 && (nullable || repeated) {
					p.Out()
					p.P(`}`)
				}
			}
		}
		if message.DescriptorProto.HasExtension() {
			if gogoproto.HasExtensionsMap(file.FileDescriptorProto, message.DescriptorProto) {
				p.P(`s = append(s, "XXX_InternalExtensions: " + extensionToGoString`, p.localName, `(this) + ",\n")`)
				extensionToGoStringUsed = true
			} else {
				p.P(`if this.XXX_extensions != nil {`)
				p.In()
				p.P(`s = append(s, "XXX_extensions: " + `, fmtPkg.Use(), `.Sprintf("%#v", this.XXX_extensions) + ",\n")`)
				p.Out()
				p.P(`}`)
			}
		}
		if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
			p.P(`if this.XXX_unrecognized != nil {`)
			p.In()
			p.P(`s = append(s, "XXX_unrecognized:" + `, fmtPkg.Use(), `.Sprintf("%#v", this.XXX_unrecognized) + ",\n")`)
			p.Out()
			p.P(`}`)
		}

		p.P(`s = append(s, "}")`)
		p.P(`return `, stringsPkg.Use(), `.Join(s, "")`)
		p.Out()
		p.P(`}`)

		//Generate GoString methods for oneof fields
		for _, field := range message.Field {
			oneof := field.OneofIndex != nil
			if !oneof {
				continue
			}
			ccTypeName := p.OneOfTypeName(message, field)
			p.P(`func (this *`, ccTypeName, `) GoString() string {`)
			p.In()
			p.P(`if this == nil {`)
			p.In()
			p.P(`return "nil"`)
			p.Out()
			p.P(`}`)
			fieldname := p.GetOneOfFieldName(message, field)
			outStr := strings.Join([]string{
				"s := ",
				stringsPkg.Use(), ".Join([]string{`&", packageName, ".", ccTypeName, "{` + \n",
				"`", fieldname, ":` + ", fmtPkg.Use(), `.Sprintf("%#v", this.`, fieldname, `)`,
				" + `}`",
				`}`,
				`,", "`,
				`)`}, "")
			p.P(outStr)
			p.P(`return s`)
			p.Out()
			p.P(`}`)
		}
	}

	if !p.atleastOne {
		return
	}

	p.P(`func valueToGoString`, p.localName, `(v interface{}, typ string) string {`)
	p.In()
	p.P(`rv := `, reflectPkg.Use(), `.ValueOf(v)`)
	p.P(`if rv.IsNil() {`)
	p.In()
	p.P(`return "nil"`)
	p.Out()
	p.P(`}`)
	p.P(`pv := `, reflectPkg.Use(), `.Indirect(rv).Interface()`)
	p.P(`return `, fmtPkg.Use(), `.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)`)
	p.Out()
	p.P(`}`)

	if extensionToGoStringUsed {
		if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
			fmt.Fprintf(os.Stderr, "The GoString plugin for messages with extensions requires importing gogoprotobuf. Please see file %s", file.GetName())
			os.Exit(1)
		}
		p.P(`func extensionToGoString`, p.localName, `(m `, protoPkg.Use(), `.Message) string {`)
		p.In()
		p.P(`e := `, protoPkg.Use(), `.GetUnsafeExtensionsMap(m)`)
		p.P(`if e == nil { return "nil" }`)
		p.P(`s := "proto.NewUnsafeXXX_InternalExtensions(map[int32]proto.Extension{"`)
		p.P(`keys := make([]int, 0, len(e))`)
		p.P(`for k := range e {`)
		p.In()
		p.P(`keys = append(keys, int(k))`)
		p.Out()
		p.P(`}`)
		p.P(sortPkg.Use(), `.Ints(keys)`)
		p.P(`ss := []string{}`)
		p.P(`for _, k := range keys {`)
		p.In()
		p.P(`ss = append(ss, `, strconvPkg.Use(), `.Itoa(k) + ": " + e[int32(k)].GoString())`)
		p.Out()
		p.P(`}`)
		p.P(`s+=`, stringsPkg.Use(), `.Join(ss, ",") + "})"`)
		p.P(`return s`)
		p.Out()
		p.P(`}`)
	}
}

func init() {
	generator.RegisterPlugin(NewGoString())
}
