/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package runtime

import (
	"fmt"
	"io"
	"path"
	"reflect"
	"sort"
	"strings"

	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/util/sets"
)

// TODO(wojtek-t): As suggested in #8320, we should consider the strategy
// to first do the shallow copy and then recurse into things that need a
// deep copy (maps, pointers, slices). That sort of copy function would
// need one parameter - a pointer to the thing it's supposed to expand,
// and it would involve a lot less memory copying.
type DeepCopyGenerator interface {
	// Adds a type to a generator.
	// If the type is non-struct, it will return an error, otherwise deep-copy
	// functions for this type and all nested types will be generated.
	AddType(inType reflect.Type) error

	// ReplaceType registers a type that should be used instead of the type
	// with the provided pkgPath and name.
	ReplaceType(pkgPath, name string, in interface{})

	// AddImport registers a package name with the generator and returns its
	// short name.
	AddImport(pkgPath string) string

	// RepackImports creates a stable ordering of import short names
	RepackImports()

	// Writes all imports that are necessary for deep-copy function and
	// their registration.
	WriteImports(w io.Writer) error

	// Writes deel-copy functions for all types added via AddType() method
	// and their nested types.
	WriteDeepCopyFunctions(w io.Writer) error

	// Writes an init() function that registers all the generated deep-copy
	// functions.
	RegisterDeepCopyFunctions(w io.Writer, pkg string) error

	// When generating code, all references to "pkg" package name will be
	// replaced with "overwrite". It is used mainly to replace references
	// to name of the package in which the code will be created with empty
	// string.
	OverwritePackage(pkg, overwrite string)
}

func NewDeepCopyGenerator(scheme *conversion.Scheme, targetPkg string, include sets.String) DeepCopyGenerator {
	g := &deepCopyGenerator{
		scheme:        scheme,
		targetPkg:     targetPkg,
		copyables:     make(map[reflect.Type]bool),
		imports:       make(map[string]string),
		shortImports:  make(map[string]string),
		pkgOverwrites: make(map[string]string),
		replace:       make(map[pkgPathNamePair]reflect.Type),
		include:       include,
	}
	g.targetPackage(targetPkg)
	g.AddImport("k8s.io/kubernetes/pkg/conversion")
	return g
}

type pkgPathNamePair struct {
	PkgPath string
	Name    string
}

type deepCopyGenerator struct {
	scheme    *conversion.Scheme
	targetPkg string
	copyables map[reflect.Type]bool
	// map of package names to shortname
	imports map[string]string
	// map of short names to package names
	shortImports  map[string]string
	pkgOverwrites map[string]string
	replace       map[pkgPathNamePair]reflect.Type
	include       sets.String
}

func (g *deepCopyGenerator) addImportByPath(pkg string) string {
	if name, ok := g.imports[pkg]; ok {
		return name
	}
	name := path.Base(pkg)
	if _, ok := g.shortImports[name]; !ok {
		g.imports[pkg] = name
		g.shortImports[name] = pkg
		return name
	}
	if dirname := path.Base(path.Dir(pkg)); len(dirname) > 0 {
		name = dirname + name
		if _, ok := g.shortImports[name]; !ok {
			g.imports[pkg] = name
			g.shortImports[name] = pkg
			return name
		}
		if subdirname := path.Base(path.Dir(path.Dir(pkg))); len(subdirname) > 0 {
			name = subdirname + name
			if _, ok := g.shortImports[name]; !ok {
				g.imports[pkg] = name
				g.shortImports[name] = pkg
				return name
			}
		}
	}
	for i := 2; i < 100; i++ {
		generatedName := fmt.Sprintf("%s%d", name, i)
		if _, ok := g.shortImports[generatedName]; !ok {
			g.imports[pkg] = generatedName
			g.shortImports[generatedName] = pkg
			return generatedName
		}
	}
	panic(fmt.Sprintf("unable to find a unique name for the package path %q: %v", pkg, g.shortImports))
}

func (g *deepCopyGenerator) targetPackage(pkg string) {
	g.imports[pkg] = ""
	g.shortImports[""] = pkg
}

func (g *deepCopyGenerator) addAllRecursiveTypes(inType reflect.Type) error {
	if _, found := g.copyables[inType]; found {
		return nil
	}
	switch inType.Kind() {
	case reflect.Map:
		if err := g.addAllRecursiveTypes(inType.Key()); err != nil {
			return err
		}
		if err := g.addAllRecursiveTypes(inType.Elem()); err != nil {
			return err
		}
	case reflect.Slice, reflect.Ptr:
		if err := g.addAllRecursiveTypes(inType.Elem()); err != nil {
			return err
		}
	case reflect.Interface:
		g.addImportByPath(inType.PkgPath())
		return nil
	case reflect.Struct:
		g.addImportByPath(inType.PkgPath())
		found := false
		for s := range g.include {
			if strings.HasPrefix(inType.PkgPath(), s) {
				found = true
				break
			}
		}
		if !found {
			return nil
		}
		for i := 0; i < inType.NumField(); i++ {
			inField := inType.Field(i)
			if err := g.addAllRecursiveTypes(inField.Type); err != nil {
				return err
			}
		}
		g.copyables[inType] = true
	default:
		// Simple types should be copied automatically.
	}
	return nil
}

func (g *deepCopyGenerator) AddImport(pkg string) string {
	return g.addImportByPath(pkg)
}

// ReplaceType registers a replacement type to be used instead of the named type
func (g *deepCopyGenerator) ReplaceType(pkgPath, name string, t interface{}) {
	g.replace[pkgPathNamePair{pkgPath, name}] = reflect.TypeOf(t)
}

func (g *deepCopyGenerator) AddType(inType reflect.Type) error {
	if inType.Kind() != reflect.Struct {
		return fmt.Errorf("non-struct copies are not supported")
	}
	return g.addAllRecursiveTypes(inType)
}

func (g *deepCopyGenerator) RepackImports() {
	var packages []string
	for key := range g.imports {
		packages = append(packages, key)
	}
	sort.Strings(packages)
	g.imports = make(map[string]string)
	g.shortImports = make(map[string]string)

	g.targetPackage(g.targetPkg)
	for _, pkg := range packages {
		g.addImportByPath(pkg)
	}
}

func (g *deepCopyGenerator) WriteImports(w io.Writer) error {
	var packages []string
	for key := range g.imports {
		packages = append(packages, key)
	}
	sort.Strings(packages)

	buffer := newBuffer()
	indent := 0
	buffer.addLine("import (\n", indent)
	for _, importPkg := range packages {
		if len(importPkg) == 0 {
			continue
		}
		if len(g.imports[importPkg]) == 0 {
			continue
		}
		buffer.addLine(fmt.Sprintf("%s \"%s\"\n", g.imports[importPkg], importPkg), indent+1)
	}
	buffer.addLine(")\n", indent)
	buffer.addLine("\n", indent)
	if err := buffer.flushLines(w); err != nil {
		return err
	}
	return nil
}

type byPkgAndName []reflect.Type

func (s byPkgAndName) Len() int {
	return len(s)
}

func (s byPkgAndName) Less(i, j int) bool {
	fullNameI := s[i].PkgPath() + "/" + s[i].Name()
	fullNameJ := s[j].PkgPath() + "/" + s[j].Name()
	return fullNameI < fullNameJ
}

func (s byPkgAndName) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (g *deepCopyGenerator) nameForType(inType reflect.Type) string {
	switch inType.Kind() {
	case reflect.Slice:
		return fmt.Sprintf("[]%s", g.typeName(inType.Elem()))
	case reflect.Ptr:
		return fmt.Sprintf("*%s", g.typeName(inType.Elem()))
	case reflect.Map:
		if len(inType.Name()) == 0 {
			return fmt.Sprintf("map[%s]%s", g.typeName(inType.Key()), g.typeName(inType.Elem()))
		}
		fallthrough
	default:
		pkg, name := inType.PkgPath(), inType.Name()
		if len(name) == 0 && inType.Kind() == reflect.Struct {
			return "struct{}"
		}
		if len(pkg) == 0 {
			// Default package.
			return name
		}
		if val, found := g.pkgOverwrites[pkg]; found {
			pkg = val
		}
		if len(pkg) == 0 {
			return name
		}
		short := g.addImportByPath(pkg)
		if len(short) > 0 {
			return fmt.Sprintf("%s.%s", short, name)
		}
		return name
	}
}

func (g *deepCopyGenerator) typeName(inType reflect.Type) string {
	if t, ok := g.replace[pkgPathNamePair{inType.PkgPath(), inType.Name()}]; ok {
		return g.nameForType(t)
	}
	return g.nameForType(inType)
}

func (g *deepCopyGenerator) deepCopyFunctionName(inType reflect.Type) string {
	funcNameFormat := "deepCopy_%s_%s"
	inPkg := packageForName(inType)
	funcName := fmt.Sprintf(funcNameFormat, inPkg, inType.Name())
	return funcName
}

func (g *deepCopyGenerator) writeHeader(b *buffer, inType reflect.Type, indent int) {
	format := "func %s(in %s, out *%s, c *conversion.Cloner) error {\n"
	stmt := fmt.Sprintf(format, g.deepCopyFunctionName(inType), g.typeName(inType), g.typeName(inType))
	b.addLine(stmt, indent)
}

func (g *deepCopyGenerator) writeFooter(b *buffer, indent int) {
	b.addLine("return nil\n", indent+1)
	b.addLine("}\n", indent)
}

func (g *deepCopyGenerator) WriteDeepCopyFunctions(w io.Writer) error {
	var keys []reflect.Type
	for key := range g.copyables {
		keys = append(keys, key)
	}
	sort.Sort(byPkgAndName(keys))

	buffer := newBuffer()
	indent := 0
	for _, inType := range keys {
		if err := g.writeDeepCopyForType(buffer, inType, indent); err != nil {
			return err
		}
		buffer.addLine("\n", 0)
	}
	if err := buffer.flushLines(w); err != nil {
		return err
	}
	return nil
}

func (g *deepCopyGenerator) writeDeepCopyForMap(b *buffer, inField reflect.StructField, indent int) error {
	ifFormat := "if in.%s != nil {\n"
	ifStmt := fmt.Sprintf(ifFormat, inField.Name)
	b.addLine(ifStmt, indent)
	newFormat := "out.%s = make(%s)\n"
	newStmt := fmt.Sprintf(newFormat, inField.Name, g.typeName(inField.Type))
	b.addLine(newStmt, indent+1)
	forFormat := "for key, val := range in.%s {\n"
	forStmt := fmt.Sprintf(forFormat, inField.Name)
	b.addLine(forStmt, indent+1)

	switch inField.Type.Key().Kind() {
	case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
		return fmt.Errorf("not supported")
	default:
		switch inField.Type.Elem().Kind() {
		case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
			if _, found := g.copyables[inField.Type.Elem()]; found {
				newFormat := "newVal := new(%s)\n"
				newStmt := fmt.Sprintf(newFormat, g.typeName(inField.Type.Elem()))
				b.addLine(newStmt, indent+2)
				assignFormat := "if err := %s(val, newVal, c); err != nil {\n"
				funcName := g.deepCopyFunctionName(inField.Type.Elem())
				assignStmt := fmt.Sprintf(assignFormat, funcName)
				b.addLine(assignStmt, indent+2)
				b.addLine("return err\n", indent+3)
				b.addLine("}\n", indent+2)
				setFormat := "out.%s[key] = *newVal\n"
				setStmt := fmt.Sprintf(setFormat, inField.Name)
				b.addLine(setStmt, indent+2)
			} else {
				ifStmt := "if newVal, err := c.DeepCopy(val); err != nil {\n"
				b.addLine(ifStmt, indent+2)
				b.addLine("return err\n", indent+3)
				b.addLine("} else {\n", indent+2)
				assignFormat := "out.%s[key] = newVal.(%s)\n"
				assignStmt := fmt.Sprintf(assignFormat, inField.Name, g.typeName(inField.Type.Elem()))
				b.addLine(assignStmt, indent+3)
				b.addLine("}\n", indent+2)
			}
		default:
			assignFormat := "out.%s[key] = val\n"
			assignStmt := fmt.Sprintf(assignFormat, inField.Name)
			b.addLine(assignStmt, indent+2)
		}
	}
	b.addLine("}\n", indent+1)
	b.addLine("} else {\n", indent)
	elseFormat := "out.%s = nil\n"
	elseStmt := fmt.Sprintf(elseFormat, inField.Name)
	b.addLine(elseStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func (g *deepCopyGenerator) writeDeepCopyForPtr(b *buffer, inField reflect.StructField, indent int) error {
	ifFormat := "if in.%s != nil {\n"
	ifStmt := fmt.Sprintf(ifFormat, inField.Name)
	b.addLine(ifStmt, indent)

	kind := inField.Type.Elem().Kind()
	switch kind {
	case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
		if _, found := g.copyables[inField.Type.Elem()]; found {
			newFormat := "out.%s = new(%s)\n"
			newStmt := fmt.Sprintf(newFormat, inField.Name, g.typeName(inField.Type.Elem()))
			b.addLine(newStmt, indent+1)
			assignFormat := "if err := %s(*in.%s, out.%s, c); err != nil {\n"
			funcName := g.deepCopyFunctionName(inField.Type.Elem())
			assignStmt := fmt.Sprintf(assignFormat, funcName, inField.Name, inField.Name)
			b.addLine(assignStmt, indent+1)
			b.addLine("return err\n", indent+2)
			b.addLine("}\n", indent+1)
		} else {
			ifFormat := "if newVal, err := c.DeepCopy(in.%s); err != nil {\n"
			ifStmt := fmt.Sprintf(ifFormat, inField.Name)
			b.addLine(ifStmt, indent+1)
			b.addLine("return err\n", indent+2)
			if kind != reflect.Struct {
				b.addLine("} else if newVal == nil {\n", indent+1)
				b.addLine(fmt.Sprintf("out.%s = nil\n", inField.Name), indent+2)
			}
			b.addLine("} else {\n", indent+1)
			assignFormat := "out.%s = newVal.(%s)\n"
			assignStmt := fmt.Sprintf(assignFormat, inField.Name, g.typeName(inField.Type))
			b.addLine(assignStmt, indent+2)
			b.addLine("}\n", indent+1)
		}
	default:
		newFormat := "out.%s = new(%s)\n"
		newStmt := fmt.Sprintf(newFormat, inField.Name, g.typeName(inField.Type.Elem()))
		b.addLine(newStmt, indent+1)
		assignFormat := "*out.%s = *in.%s\n"
		assignStmt := fmt.Sprintf(assignFormat, inField.Name, inField.Name)
		b.addLine(assignStmt, indent+1)
	}
	b.addLine("} else {\n", indent)
	elseFormat := "out.%s = nil\n"
	elseStmt := fmt.Sprintf(elseFormat, inField.Name)
	b.addLine(elseStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func (g *deepCopyGenerator) writeDeepCopyForSlice(b *buffer, inField reflect.StructField, indent int) error {
	ifFormat := "if in.%s != nil {\n"
	ifStmt := fmt.Sprintf(ifFormat, inField.Name)
	b.addLine(ifStmt, indent)
	newFormat := "out.%s = make(%s, len(in.%s))\n"
	newStmt := fmt.Sprintf(newFormat, inField.Name, g.typeName(inField.Type), inField.Name)
	b.addLine(newStmt, indent+1)
	forFormat := "for i := range in.%s {\n"
	forStmt := fmt.Sprintf(forFormat, inField.Name)
	b.addLine(forStmt, indent+1)

	kind := inField.Type.Elem().Kind()
	switch kind {
	case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
		if _, found := g.copyables[inField.Type.Elem()]; found {
			assignFormat := "if err := %s(in.%s[i], &out.%s[i], c); err != nil {\n"
			funcName := g.deepCopyFunctionName(inField.Type.Elem())
			assignStmt := fmt.Sprintf(assignFormat, funcName, inField.Name, inField.Name)
			b.addLine(assignStmt, indent+2)
			b.addLine("return err\n", indent+3)
			b.addLine("}\n", indent+2)
		} else {
			ifFormat := "if newVal, err := c.DeepCopy(in.%s[i]); err != nil {\n"
			ifStmt := fmt.Sprintf(ifFormat, inField.Name)
			b.addLine(ifStmt, indent+2)
			b.addLine("return err\n", indent+3)
			if kind != reflect.Struct {
				b.addLine("} else if newVal == nil {\n", indent+2)
				b.addLine(fmt.Sprintf("out.%s[i] = nil\n", inField.Name), indent+3)
			}
			b.addLine("} else {\n", indent+2)
			assignFormat := "out.%s[i] = newVal.(%s)\n"
			assignStmt := fmt.Sprintf(assignFormat, inField.Name, g.typeName(inField.Type.Elem()))
			b.addLine(assignStmt, indent+3)
			b.addLine("}\n", indent+2)
		}
	default:
		assignFormat := "out.%s[i] = in.%s[i]\n"
		assignStmt := fmt.Sprintf(assignFormat, inField.Name, inField.Name)
		b.addLine(assignStmt, indent+2)
	}
	b.addLine("}\n", indent+1)
	b.addLine("} else {\n", indent)
	elseFormat := "out.%s = nil\n"
	elseStmt := fmt.Sprintf(elseFormat, inField.Name)
	b.addLine(elseStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func (g *deepCopyGenerator) writeDeepCopyForStruct(b *buffer, inType reflect.Type, indent int) error {
	for i := 0; i < inType.NumField(); i++ {
		inField := inType.Field(i)
		switch inField.Type.Kind() {
		case reflect.Map:
			if err := g.writeDeepCopyForMap(b, inField, indent); err != nil {
				return err
			}
		case reflect.Ptr:
			if err := g.writeDeepCopyForPtr(b, inField, indent); err != nil {
				return err
			}
		case reflect.Slice:
			if err := g.writeDeepCopyForSlice(b, inField, indent); err != nil {
				return err
			}
		case reflect.Interface:
			ifFormat := "if newVal, err := c.DeepCopy(in.%s); err != nil {\n"
			ifStmt := fmt.Sprintf(ifFormat, inField.Name)
			b.addLine(ifStmt, indent)
			b.addLine("return err\n", indent+1)
			b.addLine("} else if newVal == nil {\n", indent)
			b.addLine(fmt.Sprintf("out.%s = nil\n", inField.Name), indent+1)
			b.addLine("} else {\n", indent)
			copyFormat := "out.%s = newVal.(%s)\n"
			copyStmt := fmt.Sprintf(copyFormat, inField.Name, g.typeName(inField.Type))
			b.addLine(copyStmt, indent+1)
			b.addLine("}\n", indent)
		case reflect.Struct:
			if _, found := g.copyables[inField.Type]; found {
				ifFormat := "if err := %s(in.%s, &out.%s, c); err != nil {\n"
				funcName := g.deepCopyFunctionName(inField.Type)
				ifStmt := fmt.Sprintf(ifFormat, funcName, inField.Name, inField.Name)
				b.addLine(ifStmt, indent)
				b.addLine("return err\n", indent+1)
				b.addLine("}\n", indent)
			} else {
				ifFormat := "if newVal, err := c.DeepCopy(in.%s); err != nil {\n"
				ifStmt := fmt.Sprintf(ifFormat, inField.Name)
				b.addLine(ifStmt, indent)
				b.addLine("return err\n", indent+1)
				b.addLine("} else {\n", indent)
				assignFormat := "out.%s = newVal.(%s)\n"
				assignStmt := fmt.Sprintf(assignFormat, inField.Name, g.typeName(inField.Type))
				b.addLine(assignStmt, indent+1)
				b.addLine("}\n", indent)
			}
		default:
			// This should handle all simple types.
			assignFormat := "out.%s = in.%s\n"
			assignStmt := fmt.Sprintf(assignFormat, inField.Name, inField.Name)
			b.addLine(assignStmt, indent)
		}
	}
	return nil
}

func (g *deepCopyGenerator) writeDeepCopyForType(b *buffer, inType reflect.Type, indent int) error {
	g.writeHeader(b, inType, indent)
	switch inType.Kind() {
	case reflect.Struct:
		if err := g.writeDeepCopyForStruct(b, inType, indent+1); err != nil {
			return err
		}
	default:
		return fmt.Errorf("type not supported: %v", inType)
	}
	g.writeFooter(b, indent)
	return nil
}

func (g *deepCopyGenerator) writeRegisterHeader(b *buffer, pkg string, indent int) {
	b.addLine("func init() {\n", indent)
	registerFormat := "err := %s.AddGeneratedDeepCopyFuncs(\n"
	b.addLine(fmt.Sprintf(registerFormat, pkg), indent+1)
}

func (g *deepCopyGenerator) writeRegisterFooter(b *buffer, indent int) {
	b.addLine(")\n", indent+1)
	b.addLine("if err != nil {\n", indent+1)
	b.addLine("// if one of the deep copy functions is malformed, detect it immediately.\n", indent+2)
	b.addLine("panic(err)\n", indent+2)
	b.addLine("}\n", indent+1)
	b.addLine("}\n", indent)
	b.addLine("\n", indent)
}

func (g *deepCopyGenerator) RegisterDeepCopyFunctions(w io.Writer, pkg string) error {
	var keys []reflect.Type
	for key := range g.copyables {
		keys = append(keys, key)
	}
	sort.Sort(byPkgAndName(keys))

	buffer := newBuffer()
	indent := 0
	g.writeRegisterHeader(buffer, pkg, indent)
	for _, inType := range keys {
		funcStmt := fmt.Sprintf("%s,\n", g.deepCopyFunctionName(inType))
		buffer.addLine(funcStmt, indent+2)
	}
	g.writeRegisterFooter(buffer, indent)
	if err := buffer.flushLines(w); err != nil {
		return err
	}
	return nil
}

func (g *deepCopyGenerator) OverwritePackage(pkg, overwrite string) {
	g.pkgOverwrites[pkg] = overwrite
}
