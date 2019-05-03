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

package generator

import (
	"bytes"
	"go/parser"
	"go/printer"
	"go/token"
	"path"
	"strings"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	plugin "github.com/gogo/protobuf/protoc-gen-gogo/plugin"
)

func (d *FileDescriptor) Messages() []*Descriptor {
	return d.desc
}

func (d *FileDescriptor) Enums() []*EnumDescriptor {
	return d.enum
}

func (d *Descriptor) IsGroup() bool {
	return d.group
}

func (g *Generator) IsGroup(field *descriptor.FieldDescriptorProto) bool {
	if d, ok := g.typeNameToObject[field.GetTypeName()].(*Descriptor); ok {
		return d.IsGroup()
	}
	return false
}

func (g *Generator) TypeNameByObject(typeName string) Object {
	o, ok := g.typeNameToObject[typeName]
	if !ok {
		g.Fail("can't find object with type", typeName)
	}
	return o
}

func (g *Generator) OneOfTypeName(message *Descriptor, field *descriptor.FieldDescriptorProto) string {
	typeName := message.TypeName()
	ccTypeName := CamelCaseSlice(typeName)
	fieldName := g.GetOneOfFieldName(message, field)
	tname := ccTypeName + "_" + fieldName
	// It is possible for this to collide with a message or enum
	// nested in this message. Check for collisions.
	ok := true
	for _, desc := range message.nested {
		if strings.Join(desc.TypeName(), "_") == tname {
			ok = false
			break
		}
	}
	for _, enum := range message.enums {
		if strings.Join(enum.TypeName(), "_") == tname {
			ok = false
			break
		}
	}
	if !ok {
		tname += "_"
	}
	return tname
}

type PluginImports interface {
	NewImport(pkg string) Single
	GenerateImports(file *FileDescriptor)
}

type pluginImports struct {
	generator *Generator
	singles   []Single
}

func NewPluginImports(generator *Generator) *pluginImports {
	return &pluginImports{generator, make([]Single, 0)}
}

func (this *pluginImports) NewImport(pkg string) Single {
	imp := newImportedPackage(this.generator.ImportPrefix, pkg)
	this.singles = append(this.singles, imp)
	return imp
}

func (this *pluginImports) GenerateImports(file *FileDescriptor) {
	for _, s := range this.singles {
		if s.IsUsed() {
			this.generator.PrintImport(GoPackageName(s.Name()), GoImportPath(s.Location()))
		}
	}
}

type Single interface {
	Use() string
	IsUsed() bool
	Name() string
	Location() string
}

type importedPackage struct {
	used         bool
	pkg          string
	name         string
	importPrefix string
}

func newImportedPackage(importPrefix string, pkg string) *importedPackage {
	return &importedPackage{
		pkg:          pkg,
		importPrefix: importPrefix,
	}
}

func (this *importedPackage) Use() string {
	if !this.used {
		this.name = string(cleanPackageName(this.pkg))
		this.used = true
	}
	return this.name
}

func (this *importedPackage) IsUsed() bool {
	return this.used
}

func (this *importedPackage) Name() string {
	return this.name
}

func (this *importedPackage) Location() string {
	return this.importPrefix + this.pkg
}

func (g *Generator) GetFieldName(message *Descriptor, field *descriptor.FieldDescriptorProto) string {
	goTyp, _ := g.GoType(message, field)
	fieldname := CamelCase(*field.Name)
	if gogoproto.IsCustomName(field) {
		fieldname = gogoproto.GetCustomName(field)
	}
	if gogoproto.IsEmbed(field) {
		fieldname = EmbedFieldName(goTyp)
	}
	if field.OneofIndex != nil {
		fieldname = message.OneofDecl[int(*field.OneofIndex)].GetName()
		fieldname = CamelCase(fieldname)
	}
	for _, f := range methodNames {
		if f == fieldname {
			return fieldname + "_"
		}
	}
	if !gogoproto.IsProtoSizer(message.file.FileDescriptorProto, message.DescriptorProto) {
		if fieldname == "Size" {
			return fieldname + "_"
		}
	}
	return fieldname
}

func (g *Generator) GetOneOfFieldName(message *Descriptor, field *descriptor.FieldDescriptorProto) string {
	goTyp, _ := g.GoType(message, field)
	fieldname := CamelCase(*field.Name)
	if gogoproto.IsCustomName(field) {
		fieldname = gogoproto.GetCustomName(field)
	}
	if gogoproto.IsEmbed(field) {
		fieldname = EmbedFieldName(goTyp)
	}
	for _, f := range methodNames {
		if f == fieldname {
			return fieldname + "_"
		}
	}
	if !gogoproto.IsProtoSizer(message.file.FileDescriptorProto, message.DescriptorProto) {
		if fieldname == "Size" {
			return fieldname + "_"
		}
	}
	return fieldname
}

func (g *Generator) IsMap(field *descriptor.FieldDescriptorProto) bool {
	if !field.IsMessage() {
		return false
	}
	byName := g.ObjectNamed(field.GetTypeName())
	desc, ok := byName.(*Descriptor)
	if byName == nil || !ok || !desc.GetOptions().GetMapEntry() {
		return false
	}
	return true
}

func (g *Generator) GetMapKeyField(field, keyField *descriptor.FieldDescriptorProto) *descriptor.FieldDescriptorProto {
	if !gogoproto.IsCastKey(field) {
		return keyField
	}
	keyField = proto.Clone(keyField).(*descriptor.FieldDescriptorProto)
	if keyField.Options == nil {
		keyField.Options = &descriptor.FieldOptions{}
	}
	keyType := gogoproto.GetCastKey(field)
	if err := proto.SetExtension(keyField.Options, gogoproto.E_Casttype, &keyType); err != nil {
		g.Fail(err.Error())
	}
	return keyField
}

func (g *Generator) GetMapValueField(field, valField *descriptor.FieldDescriptorProto) *descriptor.FieldDescriptorProto {
	if gogoproto.IsCustomType(field) && gogoproto.IsCastValue(field) {
		g.Fail("cannot have a customtype and casttype: ", field.String())
	}
	valField = proto.Clone(valField).(*descriptor.FieldDescriptorProto)
	if valField.Options == nil {
		valField.Options = &descriptor.FieldOptions{}
	}

	stdtime := gogoproto.IsStdTime(field)
	if stdtime {
		if err := proto.SetExtension(valField.Options, gogoproto.E_Stdtime, &stdtime); err != nil {
			g.Fail(err.Error())
		}
	}

	stddur := gogoproto.IsStdDuration(field)
	if stddur {
		if err := proto.SetExtension(valField.Options, gogoproto.E_Stdduration, &stddur); err != nil {
			g.Fail(err.Error())
		}
	}

	wktptr := gogoproto.IsWktPtr(field)
	if wktptr {
		if err := proto.SetExtension(valField.Options, gogoproto.E_Wktpointer, &wktptr); err != nil {
			g.Fail(err.Error())
		}
	}

	if valType := gogoproto.GetCastValue(field); len(valType) > 0 {
		if err := proto.SetExtension(valField.Options, gogoproto.E_Casttype, &valType); err != nil {
			g.Fail(err.Error())
		}
	}
	if valType := gogoproto.GetCustomType(field); len(valType) > 0 {
		if err := proto.SetExtension(valField.Options, gogoproto.E_Customtype, &valType); err != nil {
			g.Fail(err.Error())
		}
	}

	nullable := gogoproto.IsNullable(field)
	if err := proto.SetExtension(valField.Options, gogoproto.E_Nullable, &nullable); err != nil {
		g.Fail(err.Error())
	}
	return valField
}

// GoMapValueTypes returns the map value Go type and the alias map value Go type (for casting), taking into
// account whether the map is nullable or the value is a message.
func GoMapValueTypes(mapField, valueField *descriptor.FieldDescriptorProto, goValueType, goValueAliasType string) (nullable bool, outGoType string, outGoAliasType string) {
	nullable = gogoproto.IsNullable(mapField) && (valueField.IsMessage() || gogoproto.IsCustomType(mapField))
	if nullable {
		// ensure the non-aliased Go value type is a pointer for consistency
		if strings.HasPrefix(goValueType, "*") {
			outGoType = goValueType
		} else {
			outGoType = "*" + goValueType
		}
		outGoAliasType = goValueAliasType
	} else {
		outGoType = strings.Replace(goValueType, "*", "", 1)
		outGoAliasType = strings.Replace(goValueAliasType, "*", "", 1)
	}
	return
}

func GoTypeToName(goTyp string) string {
	return strings.Replace(strings.Replace(goTyp, "*", "", -1), "[]", "", -1)
}

func EmbedFieldName(goTyp string) string {
	goTyp = GoTypeToName(goTyp)
	goTyps := strings.Split(goTyp, ".")
	if len(goTyps) == 1 {
		return goTyp
	}
	if len(goTyps) == 2 {
		return goTyps[1]
	}
	panic("unreachable")
}

func (g *Generator) GeneratePlugin(p Plugin) {
	plugins = []Plugin{p}
	p.Init(g)
	// Generate the output. The generator runs for every file, even the files
	// that we don't generate output for, so that we can collate the full list
	// of exported symbols to support public imports.
	genFileMap := make(map[*FileDescriptor]bool, len(g.genFiles))
	for _, file := range g.genFiles {
		genFileMap[file] = true
	}
	for _, file := range g.allFiles {
		g.Reset()
		g.writeOutput = genFileMap[file]
		g.generatePlugin(file, p)
		if !g.writeOutput {
			continue
		}
		g.Response.File = append(g.Response.File, &plugin.CodeGeneratorResponse_File{
			Name:    proto.String(file.goFileName(g.pathType)),
			Content: proto.String(g.String()),
		})
	}
}

func (g *Generator) SetFile(filename string) {
	g.file = g.fileByName(filename)
}

func (g *Generator) generatePlugin(file *FileDescriptor, p Plugin) {
	g.writtenImports = make(map[string]bool)
	g.usedPackages = make(map[GoImportPath]bool)
	g.packageNames = make(map[GoImportPath]GoPackageName)
	g.usedPackageNames = make(map[GoPackageName]bool)
	g.file = file

	// Run the plugins before the imports so we know which imports are necessary.
	p.Generate(file)

	// Generate header and imports last, though they appear first in the output.
	rem := g.Buffer
	g.Buffer = new(bytes.Buffer)
	g.generateHeader()
	// p.GenerateImports(g.file)
	g.generateImports()
	if !g.writeOutput {
		return
	}
	g.Write(rem.Bytes())

	// Reformat generated code.
	contents := string(g.Buffer.Bytes())
	fset := token.NewFileSet()
	ast, err := parser.ParseFile(fset, "", g, parser.ParseComments)
	if err != nil {
		g.Fail("bad Go source code was generated:", contents, err.Error())
		return
	}
	g.Reset()
	err = (&printer.Config{Mode: printer.TabIndent | printer.UseSpaces, Tabwidth: 8}).Fprint(g, fset, ast)
	if err != nil {
		g.Fail("generated Go source code could not be reformatted:", err.Error())
	}
}

func GetCustomType(field *descriptor.FieldDescriptorProto) (packageName string, typ string, err error) {
	return getCustomType(field)
}

func getCustomType(field *descriptor.FieldDescriptorProto) (packageName string, typ string, err error) {
	if field.Options != nil {
		var v interface{}
		v, err = proto.GetExtension(field.Options, gogoproto.E_Customtype)
		if err == nil && v.(*string) != nil {
			ctype := *(v.(*string))
			packageName, typ = splitCPackageType(ctype)
			return packageName, typ, nil
		}
	}
	return "", "", err
}

func splitCPackageType(ctype string) (packageName string, typ string) {
	ss := strings.Split(ctype, ".")
	if len(ss) == 1 {
		return "", ctype
	}
	packageName = strings.Join(ss[0:len(ss)-1], ".")
	typeName := ss[len(ss)-1]
	importStr := strings.Map(badToUnderscore, packageName)
	typ = importStr + "." + typeName
	return packageName, typ
}

func getCastType(field *descriptor.FieldDescriptorProto) (packageName string, typ string, err error) {
	if field.Options != nil {
		var v interface{}
		v, err = proto.GetExtension(field.Options, gogoproto.E_Casttype)
		if err == nil && v.(*string) != nil {
			ctype := *(v.(*string))
			packageName, typ = splitCPackageType(ctype)
			return packageName, typ, nil
		}
	}
	return "", "", err
}

func FileName(file *FileDescriptor) string {
	fname := path.Base(file.FileDescriptorProto.GetName())
	fname = strings.Replace(fname, ".proto", "", -1)
	fname = strings.Replace(fname, "-", "_", -1)
	fname = strings.Replace(fname, ".", "_", -1)
	return CamelCase(fname)
}

func (g *Generator) AllFiles() *descriptor.FileDescriptorSet {
	set := &descriptor.FileDescriptorSet{}
	set.File = make([]*descriptor.FileDescriptorProto, len(g.allFiles))
	for i := range g.allFiles {
		set.File[i] = g.allFiles[i].FileDescriptorProto
	}
	return set
}

func (d *Descriptor) Path() string {
	return d.path
}

func (g *Generator) useTypes() string {
	pkg := strings.Map(badToUnderscore, "github.com/gogo/protobuf/types")
	g.customImports = append(g.customImports, "github.com/gogo/protobuf/types")
	return pkg
}

func (d *FileDescriptor) GoPackageName() string {
	return string(d.packageName)
}
