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
The populate plugin generates a NewPopulated function.
This function returns a newly populated structure.

It is enabled by the following extensions:

  - populate
  - populate_all

Let us look at:

  github.com/gogo/protobuf/test/example/example.proto

Btw all the output can be seen at:

  github.com/gogo/protobuf/test/example/*

The following message:

  option (gogoproto.populate_all) = true;

  message B {
	optional A A = 1 [(gogoproto.nullable) = false, (gogoproto.embed) = true];
	repeated bytes G = 2 [(gogoproto.customtype) = "github.com/gogo/protobuf/test/custom.Uint128", (gogoproto.nullable) = false];
  }

given to the populate plugin, will generate code the following code:

  func NewPopulatedB(r randyExample, easy bool) *B {
	this := &B{}
	v2 := NewPopulatedA(r, easy)
	this.A = *v2
	if r.Intn(10) != 0 {
		v3 := r.Intn(10)
		this.G = make([]github_com_gogo_protobuf_test_custom.Uint128, v3)
		for i := 0; i < v3; i++ {
			v4 := github_com_gogo_protobuf_test_custom.NewPopulatedUint128(r)
			this.G[i] = *v4
		}
	}
	if !easy && r.Intn(10) != 0 {
		this.XXX_unrecognized = randUnrecognizedExample(r, 3)
	}
	return this
  }

The idea that is useful for testing.
Most of the other plugins' generated test code uses it.
You will still be able to use the generated test code of other packages
if you turn off the popluate plugin and write your own custom NewPopulated function.

If the easy flag is not set the XXX_unrecognized and XXX_extensions fields are also populated.
These have caused problems with JSON marshalling and unmarshalling tests.

*/
package populate

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
	"github.com/gogo/protobuf/vanity"
)

type VarGen interface {
	Next() string
	Current() string
}

type varGen struct {
	index int64
}

func NewVarGen() VarGen {
	return &varGen{0}
}

func (this *varGen) Next() string {
	this.index++
	return fmt.Sprintf("v%d", this.index)
}

func (this *varGen) Current() string {
	return fmt.Sprintf("v%d", this.index)
}

type plugin struct {
	*generator.Generator
	generator.PluginImports
	varGen     VarGen
	atleastOne bool
	localName  string
	typesPkg   generator.Single
}

func NewPlugin() *plugin {
	return &plugin{}
}

func (p *plugin) Name() string {
	return "populate"
}

func (p *plugin) Init(g *generator.Generator) {
	p.Generator = g
}

func value(typeName string, fieldType descriptor.FieldDescriptorProto_Type) string {
	switch fieldType {
	case descriptor.FieldDescriptorProto_TYPE_DOUBLE:
		return typeName + "(r.Float64())"
	case descriptor.FieldDescriptorProto_TYPE_FLOAT:
		return typeName + "(r.Float32())"
	case descriptor.FieldDescriptorProto_TYPE_INT64,
		descriptor.FieldDescriptorProto_TYPE_SFIXED64,
		descriptor.FieldDescriptorProto_TYPE_SINT64:
		return typeName + "(r.Int63())"
	case descriptor.FieldDescriptorProto_TYPE_UINT64,
		descriptor.FieldDescriptorProto_TYPE_FIXED64:
		return typeName + "(uint64(r.Uint32()))"
	case descriptor.FieldDescriptorProto_TYPE_INT32,
		descriptor.FieldDescriptorProto_TYPE_SINT32,
		descriptor.FieldDescriptorProto_TYPE_SFIXED32,
		descriptor.FieldDescriptorProto_TYPE_ENUM:
		return typeName + "(r.Int31())"
	case descriptor.FieldDescriptorProto_TYPE_UINT32,
		descriptor.FieldDescriptorProto_TYPE_FIXED32:
		return typeName + "(r.Uint32())"
	case descriptor.FieldDescriptorProto_TYPE_BOOL:
		return typeName + `(bool(r.Intn(2) == 0))`
	case descriptor.FieldDescriptorProto_TYPE_STRING,
		descriptor.FieldDescriptorProto_TYPE_GROUP,
		descriptor.FieldDescriptorProto_TYPE_MESSAGE,
		descriptor.FieldDescriptorProto_TYPE_BYTES:
	}
	panic(fmt.Errorf("unexpected type %v", typeName))
}

func negative(fieldType descriptor.FieldDescriptorProto_Type) bool {
	switch fieldType {
	case descriptor.FieldDescriptorProto_TYPE_UINT64,
		descriptor.FieldDescriptorProto_TYPE_FIXED64,
		descriptor.FieldDescriptorProto_TYPE_UINT32,
		descriptor.FieldDescriptorProto_TYPE_FIXED32,
		descriptor.FieldDescriptorProto_TYPE_BOOL:
		return false
	}
	return true
}

func (p *plugin) getFuncName(goTypName string, field *descriptor.FieldDescriptorProto) string {
	funcName := "NewPopulated" + goTypName
	goTypNames := strings.Split(goTypName, ".")
	if len(goTypNames) == 2 {
		funcName = goTypNames[0] + ".NewPopulated" + goTypNames[1]
	} else if len(goTypNames) != 1 {
		panic(fmt.Errorf("unreachable: too many dots in %v", goTypName))
	}
	if field != nil {
		switch {
		case gogoproto.IsStdTime(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdTime"
		case gogoproto.IsStdDuration(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdDuration"
		case gogoproto.IsStdDouble(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdDouble"
		case gogoproto.IsStdFloat(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdFloat"
		case gogoproto.IsStdInt64(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdInt64"
		case gogoproto.IsStdUInt64(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdUInt64"
		case gogoproto.IsStdInt32(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdInt32"
		case gogoproto.IsStdUInt32(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdUInt32"
		case gogoproto.IsStdBool(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdBool"
		case gogoproto.IsStdString(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdString"
		case gogoproto.IsStdBytes(field):
			funcName = p.typesPkg.Use() + ".NewPopulatedStdBytes"
		}
	}
	return funcName
}

func (p *plugin) getFuncCall(goTypName string, field *descriptor.FieldDescriptorProto) string {
	funcName := p.getFuncName(goTypName, field)
	funcCall := funcName + "(r, easy)"
	return funcCall
}

func (p *plugin) getCustomFuncCall(goTypName string) string {
	funcName := p.getFuncName(goTypName, nil)
	funcCall := funcName + "(r)"
	return funcCall
}

func (p *plugin) getEnumVal(field *descriptor.FieldDescriptorProto, goTyp string) string {
	enum := p.ObjectNamed(field.GetTypeName()).(*generator.EnumDescriptor)
	l := len(enum.Value)
	values := make([]string, l)
	for i := range enum.Value {
		values[i] = strconv.Itoa(int(*enum.Value[i].Number))
	}
	arr := "[]int32{" + strings.Join(values, ",") + "}"
	val := strings.Join([]string{generator.GoTypeToName(goTyp), `(`, arr, `[r.Intn(`, fmt.Sprintf("%d", l), `)])`}, "")
	return val
}

func (p *plugin) GenerateField(file *generator.FileDescriptor, message *generator.Descriptor, field *descriptor.FieldDescriptorProto) {
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	goTyp, _ := p.GoType(message, field)
	fieldname := p.GetOneOfFieldName(message, field)
	goTypName := generator.GoTypeToName(goTyp)
	if p.IsMap(field) {
		m := p.GoMapType(nil, field)
		keygoTyp, _ := p.GoType(nil, m.KeyField)
		keygoTyp = strings.Replace(keygoTyp, "*", "", 1)
		keygoAliasTyp, _ := p.GoType(nil, m.KeyAliasField)
		keygoAliasTyp = strings.Replace(keygoAliasTyp, "*", "", 1)

		valuegoTyp, _ := p.GoType(nil, m.ValueField)
		valuegoAliasTyp, _ := p.GoType(nil, m.ValueAliasField)
		keytypName := generator.GoTypeToName(keygoTyp)
		keygoAliasTyp = generator.GoTypeToName(keygoAliasTyp)
		valuetypAliasName := generator.GoTypeToName(valuegoAliasTyp)

		nullable, valuegoTyp, valuegoAliasTyp := generator.GoMapValueTypes(field, m.ValueField, valuegoTyp, valuegoAliasTyp)

		p.P(p.varGen.Next(), ` := r.Intn(10)`)
		p.P(`this.`, fieldname, ` = make(`, m.GoType, `)`)
		p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
		p.In()
		keyval := ""
		if m.KeyField.IsString() {
			keyval = fmt.Sprintf("randString%v(r)", p.localName)
		} else {
			keyval = value(keytypName, m.KeyField.GetType())
		}
		if keygoAliasTyp != keygoTyp {
			keyval = keygoAliasTyp + `(` + keyval + `)`
		}
		if m.ValueField.IsMessage() || p.IsGroup(field) ||
			(m.ValueField.IsBytes() && gogoproto.IsCustomType(field)) {
			s := `this.` + fieldname + `[` + keyval + `] = `
			if gogoproto.IsStdType(field) {
				valuegoTyp = valuegoAliasTyp
			}
			funcCall := p.getCustomFuncCall(goTypName)
			if !gogoproto.IsCustomType(field) {
				goTypName = generator.GoTypeToName(valuegoTyp)
				funcCall = p.getFuncCall(goTypName, m.ValueAliasField)
			}
			if !nullable {
				funcCall = `*` + funcCall
			}
			if valuegoTyp != valuegoAliasTyp {
				funcCall = `(` + valuegoAliasTyp + `)(` + funcCall + `)`
			}
			s += funcCall
			p.P(s)
		} else if m.ValueField.IsEnum() {
			s := `this.` + fieldname + `[` + keyval + `]` + ` = ` + p.getEnumVal(m.ValueField, valuegoTyp)
			p.P(s)
		} else if m.ValueField.IsBytes() {
			count := p.varGen.Next()
			p.P(count, ` := r.Intn(100)`)
			p.P(p.varGen.Next(), ` := `, keyval)
			p.P(`this.`, fieldname, `[`, p.varGen.Current(), `] = make(`, valuegoTyp, `, `, count, `)`)
			p.P(`for i := 0; i < `, count, `; i++ {`)
			p.In()
			p.P(`this.`, fieldname, `[`, p.varGen.Current(), `][i] = byte(r.Intn(256))`)
			p.Out()
			p.P(`}`)
		} else if m.ValueField.IsString() {
			s := `this.` + fieldname + `[` + keyval + `]` + ` = ` + fmt.Sprintf("randString%v(r)", p.localName)
			p.P(s)
		} else {
			p.P(p.varGen.Next(), ` := `, keyval)
			p.P(`this.`, fieldname, `[`, p.varGen.Current(), `] = `, value(valuetypAliasName, m.ValueField.GetType()))
			if negative(m.ValueField.GetType()) {
				p.P(`if r.Intn(2) == 0 {`)
				p.In()
				p.P(`this.`, fieldname, `[`, p.varGen.Current(), `] *= -1`)
				p.Out()
				p.P(`}`)
			}
		}
		p.Out()
		p.P(`}`)
	} else if gogoproto.IsCustomType(field) {
		funcCall := p.getCustomFuncCall(goTypName)
		if field.IsRepeated() {
			p.P(p.varGen.Next(), ` := r.Intn(10)`)
			p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
			p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
			p.In()
			p.P(p.varGen.Next(), `:= `, funcCall)
			p.P(`this.`, fieldname, `[i] = *`, p.varGen.Current())
			p.Out()
			p.P(`}`)
		} else if gogoproto.IsNullable(field) {
			p.P(`this.`, fieldname, ` = `, funcCall)
		} else {
			p.P(p.varGen.Next(), `:= `, funcCall)
			p.P(`this.`, fieldname, ` = *`, p.varGen.Current())
		}
	} else if field.IsMessage() || p.IsGroup(field) {
		funcCall := p.getFuncCall(goTypName, field)
		if field.IsRepeated() {
			p.P(p.varGen.Next(), ` := r.Intn(5)`)
			p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
			p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
			p.In()
			if gogoproto.IsNullable(field) {
				p.P(`this.`, fieldname, `[i] = `, funcCall)
			} else {
				p.P(p.varGen.Next(), `:= `, funcCall)
				p.P(`this.`, fieldname, `[i] = *`, p.varGen.Current())
			}
			p.Out()
			p.P(`}`)
		} else {
			if gogoproto.IsNullable(field) {
				p.P(`this.`, fieldname, ` = `, funcCall)
			} else {
				p.P(p.varGen.Next(), `:= `, funcCall)
				p.P(`this.`, fieldname, ` = *`, p.varGen.Current())
			}
		}
	} else {
		if field.IsEnum() {
			val := p.getEnumVal(field, goTyp)
			if field.IsRepeated() {
				p.P(p.varGen.Next(), ` := r.Intn(10)`)
				p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
				p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
				p.In()
				p.P(`this.`, fieldname, `[i] = `, val)
				p.Out()
				p.P(`}`)
			} else if !gogoproto.IsNullable(field) || proto3 {
				p.P(`this.`, fieldname, ` = `, val)
			} else {
				p.P(p.varGen.Next(), ` := `, val)
				p.P(`this.`, fieldname, ` = &`, p.varGen.Current())
			}
		} else if field.IsBytes() {
			if field.IsRepeated() {
				p.P(p.varGen.Next(), ` := r.Intn(10)`)
				p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
				p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
				p.In()
				p.P(p.varGen.Next(), ` := r.Intn(100)`)
				p.P(`this.`, fieldname, `[i] = make([]byte,`, p.varGen.Current(), `)`)
				p.P(`for j := 0; j < `, p.varGen.Current(), `; j++ {`)
				p.In()
				p.P(`this.`, fieldname, `[i][j] = byte(r.Intn(256))`)
				p.Out()
				p.P(`}`)
				p.Out()
				p.P(`}`)
			} else {
				p.P(p.varGen.Next(), ` := r.Intn(100)`)
				p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
				p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
				p.In()
				p.P(`this.`, fieldname, `[i] = byte(r.Intn(256))`)
				p.Out()
				p.P(`}`)
			}
		} else if field.IsString() {
			typName := generator.GoTypeToName(goTyp)
			val := fmt.Sprintf("%s(randString%v(r))", typName, p.localName)
			if field.IsRepeated() {
				p.P(p.varGen.Next(), ` := r.Intn(10)`)
				p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
				p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
				p.In()
				p.P(`this.`, fieldname, `[i] = `, val)
				p.Out()
				p.P(`}`)
			} else if !gogoproto.IsNullable(field) || proto3 {
				p.P(`this.`, fieldname, ` = `, val)
			} else {
				p.P(p.varGen.Next(), `:= `, val)
				p.P(`this.`, fieldname, ` = &`, p.varGen.Current())
			}
		} else {
			typName := generator.GoTypeToName(goTyp)
			if field.IsRepeated() {
				p.P(p.varGen.Next(), ` := r.Intn(10)`)
				p.P(`this.`, fieldname, ` = make(`, goTyp, `, `, p.varGen.Current(), `)`)
				p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
				p.In()
				p.P(`this.`, fieldname, `[i] = `, value(typName, field.GetType()))
				if negative(field.GetType()) {
					p.P(`if r.Intn(2) == 0 {`)
					p.In()
					p.P(`this.`, fieldname, `[i] *= -1`)
					p.Out()
					p.P(`}`)
				}
				p.Out()
				p.P(`}`)
			} else if !gogoproto.IsNullable(field) || proto3 {
				p.P(`this.`, fieldname, ` = `, value(typName, field.GetType()))
				if negative(field.GetType()) {
					p.P(`if r.Intn(2) == 0 {`)
					p.In()
					p.P(`this.`, fieldname, ` *= -1`)
					p.Out()
					p.P(`}`)
				}
			} else {
				p.P(p.varGen.Next(), ` := `, value(typName, field.GetType()))
				if negative(field.GetType()) {
					p.P(`if r.Intn(2) == 0 {`)
					p.In()
					p.P(p.varGen.Current(), ` *= -1`)
					p.Out()
					p.P(`}`)
				}
				p.P(`this.`, fieldname, ` = &`, p.varGen.Current())
			}
		}
	}
}

func (p *plugin) hasLoop(pkg string, field *descriptor.FieldDescriptorProto, visited []*generator.Descriptor, excludes []*generator.Descriptor) *generator.Descriptor {
	if field.IsMessage() || p.IsGroup(field) || p.IsMap(field) {
		var fieldMessage *generator.Descriptor
		if p.IsMap(field) {
			m := p.GoMapType(nil, field)
			if !m.ValueField.IsMessage() {
				return nil
			}
			fieldMessage = p.ObjectNamed(m.ValueField.GetTypeName()).(*generator.Descriptor)
		} else {
			fieldMessage = p.ObjectNamed(field.GetTypeName()).(*generator.Descriptor)
		}
		fieldTypeName := generator.CamelCaseSlice(fieldMessage.TypeName())
		for _, message := range visited {
			messageTypeName := generator.CamelCaseSlice(message.TypeName())
			if fieldTypeName == messageTypeName {
				for _, e := range excludes {
					if fieldTypeName == generator.CamelCaseSlice(e.TypeName()) {
						return nil
					}
				}
				return fieldMessage
			}
		}

		for _, f := range fieldMessage.Field {
			if strings.HasPrefix(f.GetTypeName(), "."+pkg) {
				visited = append(visited, fieldMessage)
				loopTo := p.hasLoop(pkg, f, visited, excludes)
				if loopTo != nil {
					return loopTo
				}
			}
		}
	}
	return nil
}

func (p *plugin) loops(pkg string, field *descriptor.FieldDescriptorProto, message *generator.Descriptor) int {
	//fmt.Fprintf(os.Stderr, "loops %v %v\n", field.GetTypeName(), generator.CamelCaseSlice(message.TypeName()))
	excludes := []*generator.Descriptor{}
	loops := 0
	for {
		visited := []*generator.Descriptor{}
		loopTo := p.hasLoop(pkg, field, visited, excludes)
		if loopTo == nil {
			break
		}
		//fmt.Fprintf(os.Stderr, "loopTo %v\n", generator.CamelCaseSlice(loopTo.TypeName()))
		excludes = append(excludes, loopTo)
		loops++
	}
	return loops
}

func (p *plugin) Generate(file *generator.FileDescriptor) {
	p.atleastOne = false
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.varGen = NewVarGen()
	proto3 := gogoproto.IsProto3(file.FileDescriptorProto)
	p.typesPkg = p.NewImport("github.com/gogo/protobuf/types")
	p.localName = generator.FileName(file)
	protoPkg := p.NewImport("github.com/gogo/protobuf/proto")
	if !gogoproto.ImportsGoGoProto(file.FileDescriptorProto) {
		protoPkg = p.NewImport("github.com/golang/protobuf/proto")
	}

	for _, message := range file.Messages() {
		if !gogoproto.HasPopulate(file.FileDescriptorProto, message.DescriptorProto) {
			continue
		}
		if message.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		p.atleastOne = true
		ccTypeName := generator.CamelCaseSlice(message.TypeName())
		loopLevels := make([]int, len(message.Field))
		maxLoopLevel := 0
		for i, field := range message.Field {
			loopLevels[i] = p.loops(file.GetPackage(), field, message)
			if loopLevels[i] > maxLoopLevel {
				maxLoopLevel = loopLevels[i]
			}
		}
		ranTotal := 0
		for i := range loopLevels {
			ranTotal += int(math.Pow10(maxLoopLevel - loopLevels[i]))
		}
		p.P(`func NewPopulated`, ccTypeName, `(r randy`, p.localName, `, easy bool) *`, ccTypeName, ` {`)
		p.In()
		p.P(`this := &`, ccTypeName, `{}`)
		if gogoproto.IsUnion(message.File().FileDescriptorProto, message.DescriptorProto) && len(message.Field) > 0 {
			p.P(`fieldNum := r.Intn(`, fmt.Sprintf("%d", ranTotal), `)`)
			p.P(`switch fieldNum {`)
			k := 0
			for i, field := range message.Field {
				is := []string{}
				ran := int(math.Pow10(maxLoopLevel - loopLevels[i]))
				for j := 0; j < ran; j++ {
					is = append(is, fmt.Sprintf("%d", j+k))
				}
				k += ran
				p.P(`case `, strings.Join(is, ","), `:`)
				p.In()
				p.GenerateField(file, message, field)
				p.Out()
			}
			p.P(`}`)
		} else {
			var maxFieldNumber int32
			oneofs := make(map[string]struct{})
			for fieldIndex, field := range message.Field {
				if field.GetNumber() > maxFieldNumber {
					maxFieldNumber = field.GetNumber()
				}
				oneof := field.OneofIndex != nil
				if !oneof {
					if field.IsRequired() || (!gogoproto.IsNullable(field) && !field.IsRepeated()) || (proto3 && !field.IsMessage()) {
						p.GenerateField(file, message, field)
					} else {
						if loopLevels[fieldIndex] > 0 {
							p.P(`if r.Intn(10) == 0 {`)
						} else {
							p.P(`if r.Intn(10) != 0 {`)
						}
						p.In()
						p.GenerateField(file, message, field)
						p.Out()
						p.P(`}`)
					}
				} else {
					fieldname := p.GetFieldName(message, field)
					if _, ok := oneofs[fieldname]; ok {
						continue
					} else {
						oneofs[fieldname] = struct{}{}
					}
					fieldNumbers := []int32{}
					for _, f := range message.Field {
						fname := p.GetFieldName(message, f)
						if fname == fieldname {
							fieldNumbers = append(fieldNumbers, f.GetNumber())
						}
					}

					p.P(`oneofNumber_`, fieldname, ` := `, fmt.Sprintf("%#v", fieldNumbers), `[r.Intn(`, strconv.Itoa(len(fieldNumbers)), `)]`)
					p.P(`switch oneofNumber_`, fieldname, ` {`)
					for _, f := range message.Field {
						fname := p.GetFieldName(message, f)
						if fname != fieldname {
							continue
						}
						p.P(`case `, strconv.Itoa(int(f.GetNumber())), `:`)
						p.In()
						ccTypeName := p.OneOfTypeName(message, f)
						p.P(`this.`, fname, ` = NewPopulated`, ccTypeName, `(r, easy)`)
						p.Out()
					}
					p.P(`}`)
				}
			}
			if message.DescriptorProto.HasExtension() {
				p.P(`if !easy && r.Intn(10) != 0 {`)
				p.In()
				p.P(`l := r.Intn(5)`)
				p.P(`for i := 0; i < l; i++ {`)
				p.In()
				if len(message.DescriptorProto.GetExtensionRange()) > 1 {
					p.P(`eIndex := r.Intn(`, strconv.Itoa(len(message.DescriptorProto.GetExtensionRange())), `)`)
					p.P(`fieldNumber := 0`)
					p.P(`switch eIndex {`)
					for i, e := range message.DescriptorProto.GetExtensionRange() {
						p.P(`case `, strconv.Itoa(i), `:`)
						p.In()
						p.P(`fieldNumber = r.Intn(`, strconv.Itoa(int(e.GetEnd()-e.GetStart())), `) + `, strconv.Itoa(int(e.GetStart())))
						p.Out()
						if e.GetEnd() > maxFieldNumber {
							maxFieldNumber = e.GetEnd()
						}
					}
					p.P(`}`)
				} else {
					e := message.DescriptorProto.GetExtensionRange()[0]
					p.P(`fieldNumber := r.Intn(`, strconv.Itoa(int(e.GetEnd()-e.GetStart())), `) + `, strconv.Itoa(int(e.GetStart())))
					if e.GetEnd() > maxFieldNumber {
						maxFieldNumber = e.GetEnd()
					}
				}
				p.P(`wire := r.Intn(4)`)
				p.P(`if wire == 3 { wire = 5 }`)
				p.P(`dAtA := randField`, p.localName, `(nil, r, fieldNumber, wire)`)
				p.P(protoPkg.Use(), `.SetRawExtension(this, int32(fieldNumber), dAtA)`)
				p.Out()
				p.P(`}`)
				p.Out()
				p.P(`}`)
			}

			if maxFieldNumber < (1 << 10) {
				p.P(`if !easy && r.Intn(10) != 0 {`)
				p.In()
				if gogoproto.HasUnrecognized(file.FileDescriptorProto, message.DescriptorProto) {
					p.P(`this.XXX_unrecognized = randUnrecognized`, p.localName, `(r, `, strconv.Itoa(int(maxFieldNumber+1)), `)`)
				}
				p.Out()
				p.P(`}`)
			}
		}
		p.P(`return this`)
		p.Out()
		p.P(`}`)
		p.P(``)

		//Generate NewPopulated functions for oneof fields
		m := proto.Clone(message.DescriptorProto).(*descriptor.DescriptorProto)
		for _, f := range m.Field {
			oneof := f.OneofIndex != nil
			if !oneof {
				continue
			}
			ccTypeName := p.OneOfTypeName(message, f)
			p.P(`func NewPopulated`, ccTypeName, `(r randy`, p.localName, `, easy bool) *`, ccTypeName, ` {`)
			p.In()
			p.P(`this := &`, ccTypeName, `{}`)
			vanity.TurnOffNullableForNativeTypes(f)
			p.GenerateField(file, message, f)
			p.P(`return this`)
			p.Out()
			p.P(`}`)
		}
	}

	if !p.atleastOne {
		return
	}

	p.P(`type randy`, p.localName, ` interface {`)
	p.In()
	p.P(`Float32() float32`)
	p.P(`Float64() float64`)
	p.P(`Int63() int64`)
	p.P(`Int31() int32`)
	p.P(`Uint32() uint32`)
	p.P(`Intn(n int) int`)
	p.Out()
	p.P(`}`)

	p.P(`func randUTF8Rune`, p.localName, `(r randy`, p.localName, `) rune {`)
	p.In()
	p.P(`ru := r.Intn(62)`)
	p.P(`if ru < 10 {`)
	p.In()
	p.P(`return rune(ru+48)`)
	p.Out()
	p.P(`} else if ru < 36 {`)
	p.In()
	p.P(`return rune(ru+55)`)
	p.Out()
	p.P(`}`)
	p.P(`return rune(ru+61)`)
	p.Out()
	p.P(`}`)

	p.P(`func randString`, p.localName, `(r randy`, p.localName, `) string {`)
	p.In()
	p.P(p.varGen.Next(), ` := r.Intn(100)`)
	p.P(`tmps := make([]rune, `, p.varGen.Current(), `)`)
	p.P(`for i := 0; i < `, p.varGen.Current(), `; i++ {`)
	p.In()
	p.P(`tmps[i] = randUTF8Rune`, p.localName, `(r)`)
	p.Out()
	p.P(`}`)
	p.P(`return string(tmps)`)
	p.Out()
	p.P(`}`)

	p.P(`func randUnrecognized`, p.localName, `(r randy`, p.localName, `, maxFieldNumber int) (dAtA []byte) {`)
	p.In()
	p.P(`l := r.Intn(5)`)
	p.P(`for i := 0; i < l; i++ {`)
	p.In()
	p.P(`wire := r.Intn(4)`)
	p.P(`if wire == 3 { wire = 5 }`)
	p.P(`fieldNumber := maxFieldNumber + r.Intn(100)`)
	p.P(`dAtA = randField`, p.localName, `(dAtA, r, fieldNumber, wire)`)
	p.Out()
	p.P(`}`)
	p.P(`return dAtA`)
	p.Out()
	p.P(`}`)

	p.P(`func randField`, p.localName, `(dAtA []byte, r randy`, p.localName, `, fieldNumber int, wire int) []byte {`)
	p.In()
	p.P(`key := uint32(fieldNumber)<<3 | uint32(wire)`)
	p.P(`switch wire {`)
	p.P(`case 0:`)
	p.In()
	p.P(`dAtA = encodeVarintPopulate`, p.localName, `(dAtA, uint64(key))`)
	p.P(p.varGen.Next(), ` := r.Int63()`)
	p.P(`if r.Intn(2) == 0 {`)
	p.In()
	p.P(p.varGen.Current(), ` *= -1`)
	p.Out()
	p.P(`}`)
	p.P(`dAtA = encodeVarintPopulate`, p.localName, `(dAtA, uint64(`, p.varGen.Current(), `))`)
	p.Out()
	p.P(`case 1:`)
	p.In()
	p.P(`dAtA = encodeVarintPopulate`, p.localName, `(dAtA, uint64(key))`)
	p.P(`dAtA = append(dAtA, byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)))`)
	p.Out()
	p.P(`case 2:`)
	p.In()
	p.P(`dAtA = encodeVarintPopulate`, p.localName, `(dAtA, uint64(key))`)
	p.P(`ll := r.Intn(100)`)
	p.P(`dAtA = encodeVarintPopulate`, p.localName, `(dAtA, uint64(ll))`)
	p.P(`for j := 0; j < ll; j++ {`)
	p.In()
	p.P(`dAtA = append(dAtA, byte(r.Intn(256)))`)
	p.Out()
	p.P(`}`)
	p.Out()
	p.P(`default:`)
	p.In()
	p.P(`dAtA = encodeVarintPopulate`, p.localName, `(dAtA, uint64(key))`)
	p.P(`dAtA = append(dAtA, byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)), byte(r.Intn(256)))`)
	p.Out()
	p.P(`}`)
	p.P(`return dAtA`)
	p.Out()
	p.P(`}`)

	p.P(`func encodeVarintPopulate`, p.localName, `(dAtA []byte, v uint64) []byte {`)
	p.In()
	p.P(`for v >= 1<<7 {`)
	p.In()
	p.P(`dAtA = append(dAtA, uint8(uint64(v)&0x7f|0x80))`)
	p.P(`v >>= 7`)
	p.Out()
	p.P(`}`)
	p.P(`dAtA = append(dAtA, uint8(v))`)
	p.P(`return dAtA`)
	p.Out()
	p.P(`}`)

}

func init() {
	generator.RegisterPlugin(NewPlugin())
}
