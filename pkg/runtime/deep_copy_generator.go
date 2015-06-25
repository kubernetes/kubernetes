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
	"reflect"
	"sort"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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

	// Writes all imports that are necessary for deep-copy function and
	// their registration.
	WriteImports(w io.Writer, pkg string) error

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

func NewDeepCopyGenerator(scheme *conversion.Scheme) DeepCopyGenerator {
	return &deepCopyGenerator{
		scheme:        scheme,
		copyables:     make(map[reflect.Type]bool),
		imports:       util.StringSet{},
		pkgOverwrites: make(map[string]string),
	}
}

type deepCopyGenerator struct {
	scheme        *conversion.Scheme
	copyables     map[reflect.Type]bool
	imports       util.StringSet
	pkgOverwrites map[string]string
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
		g.imports.Insert(inType.PkgPath())
		return nil
	case reflect.Struct:
		g.imports.Insert(inType.PkgPath())
		if !strings.HasPrefix(inType.PkgPath(), "github.com/GoogleCloudPlatform/kubernetes") {
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

func (g *deepCopyGenerator) AddType(inType reflect.Type) error {
	if inType.Kind() != reflect.Struct {
		return fmt.Errorf("non-struct copies are not supported")
	}
	return g.addAllRecursiveTypes(inType)
}

func (g *deepCopyGenerator) WriteImports(w io.Writer, pkg string) error {
	var packages []string
	packages = append(packages, "github.com/GoogleCloudPlatform/kubernetes/pkg/api")
	packages = append(packages, "github.com/GoogleCloudPlatform/kubernetes/pkg/conversion")
	for key := range g.imports {
		packages = append(packages, key)
	}
	sort.Strings(packages)

	buffer := newBuffer()
	indent := 0
	buffer.addLine("import (\n", indent)
	for _, importPkg := range packages {
		if strings.HasSuffix(importPkg, pkg) {
			continue
		}
		buffer.addLine(fmt.Sprintf("\"%s\"\n", importPkg), indent+1)
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

func (g *deepCopyGenerator) typeName(inType reflect.Type) string {
	switch inType.Kind() {
	case reflect.Map:
		return fmt.Sprintf("map[%s]%s", g.typeName(inType.Key()), g.typeName(inType.Elem()))
	case reflect.Slice:
		return fmt.Sprintf("[]%s", g.typeName(inType.Elem()))
	case reflect.Ptr:
		return fmt.Sprintf("*%s", g.typeName(inType.Elem()))
	default:
		typeWithPkg := fmt.Sprintf("%s", inType)
		slices := strings.Split(typeWithPkg, ".")
		if len(slices) == 1 {
			// Default package.
			return slices[0]
		}
		if len(slices) == 2 {
			pkg := slices[0]
			if val, found := g.pkgOverwrites[pkg]; found {
				pkg = val
			}
			if pkg != "" {
				pkg = pkg + "."
			}
			return pkg + slices[1]
		}
		panic("Incorrect type name: " + typeWithPkg)
	}
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

	switch inField.Type.Elem().Kind() {
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

	switch inField.Type.Elem().Kind() {
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
	registerFormat := "err := %sScheme.AddGeneratedDeepCopyFuncs(\n"
	if pkg == "api" {
		b.addLine(fmt.Sprintf(registerFormat, ""), indent+1)
	} else {
		b.addLine(fmt.Sprintf(registerFormat, "api."), indent+1)
	}
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
