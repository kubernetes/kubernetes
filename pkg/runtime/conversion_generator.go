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
)

type ConversionGenerator interface {
	GenerateConversionsForType(version string, reflection reflect.Type) error
	WriteConversionFunctions(w io.Writer) error
	RegisterConversionFunctions(w io.Writer) error
	OverwritePackage(pkg, overwrite string)
}

func NewConversionGenerator(scheme *conversion.Scheme) ConversionGenerator {
	return &conversionGenerator{
		scheme:        scheme,
		convertibles:  make(map[reflect.Type]reflect.Type),
		pkgOverwrites: make(map[string]string),
	}
}

var complexTypes []reflect.Kind = []reflect.Kind{reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct}

type conversionGenerator struct {
	scheme       *conversion.Scheme
	convertibles map[reflect.Type]reflect.Type
	// If pkgOverwrites is set for a given package name, that package name
	// will be replaced while writing conversion function. If empty, package
	// name will be omitted.
	pkgOverwrites map[string]string

	// A buffer that is used for storing lines that needs to be written.
	linesToPrint []string
}

func (g *conversionGenerator) GenerateConversionsForType(version string, reflection reflect.Type) error {
	kind := reflection.Name()
	internalObj, err := g.scheme.NewObject(g.scheme.InternalVersion, kind)
	if err != nil {
		return fmt.Errorf("cannot create an object of type %v in internal version", kind)
	}
	internalObjType := reflect.TypeOf(internalObj)
	if internalObjType.Kind() != reflect.Ptr {
		return fmt.Errorf("created object should be of type Ptr: ", internalObjType.Kind())
	}
	inErr := g.generateConversionsBetween(reflection, internalObjType.Elem())
	outErr := g.generateConversionsBetween(internalObjType.Elem(), reflection)
	if inErr != nil || outErr != nil {
		return fmt.Errorf("errors: %v, %v", inErr, outErr)
	}
	return nil
}

func (g *conversionGenerator) generateConversionsBetween(inType, outType reflect.Type) error {
	existingConversion := g.scheme.Converter().HasConversionFunc(inType, outType) && g.scheme.Converter().HasConversionFunc(outType, inType)

	// Avoid processing the same type multiple times.
	if value, found := g.convertibles[inType]; found {
		if value != outType {
			return fmt.Errorf("multiple possible convertibles for %v", inType)
		}
		return nil
	}
	if inType == outType {
		// Don't generate conversion methods for the same type.
		return nil
	}

	if inType.Kind() != outType.Kind() {
		if existingConversion {
			return nil
		}
		return fmt.Errorf("cannot convert types of different kinds: %v %v", inType, outType)
	}
	// We should be able to generate conversions both sides.
	switch inType.Kind() {
	case reflect.Map:
		inErr := g.generateConversionsForMap(inType, outType)
		outErr := g.generateConversionsForMap(outType, inType)
		if !existingConversion && (inErr != nil || outErr != nil) {
			return inErr
		}
		// We don't add it to g.convertibles - maps should be handled correctly
		// inside appropriate conversion functions.
		return nil
	case reflect.Ptr:
		inErr := g.generateConversionsBetween(inType.Elem(), outType.Elem())
		outErr := g.generateConversionsBetween(outType.Elem(), inType.Elem())
		if !existingConversion && (inErr != nil || outErr != nil) {
			return inErr
		}
		// We don't add it to g.convertibles - maps should be handled correctly
		// inside appropriate conversion functions.
		return nil
	case reflect.Slice:
		inErr := g.generateConversionsForSlice(inType, outType)
		outErr := g.generateConversionsForSlice(outType, inType)
		if !existingConversion && (inErr != nil || outErr != nil) {
			return inErr
		}
		// We don't add it to g.convertibles - slices should be handled correctly
		// inside appropriate conversion functions.
		return nil
	case reflect.Interface:
		// TODO(wojtek-t): Currently we don't support converting interfaces.
		return fmt.Errorf("interfaces are not supported")
	case reflect.Struct:
		inErr := g.generateConversionsForStruct(inType, outType)
		outErr := g.generateConversionsForStruct(outType, inType)
		if !existingConversion && (inErr != nil || outErr != nil) {
			return inErr
		}
		if !existingConversion {
			g.convertibles[inType] = outType
		}
		return nil
	default:
		// All simple types should be handled correctly with default conversion.
		return nil
	}
	panic("This should never happen")
}

func isComplexType(reflection reflect.Type) bool {
	for _, complexType := range complexTypes {
		if complexType == reflection.Kind() {
			return true
		}
	}
	return false
}

func (g *conversionGenerator) generateConversionsForMap(inType, outType reflect.Type) error {
	inKey := inType.Key()
	outKey := outType.Key()
	if err := g.generateConversionsBetween(inKey, outKey); err != nil {
		return err
	}
	inValue := inType.Elem()
	outValue := outType.Elem()
	if err := g.generateConversionsBetween(inValue, outValue); err != nil {
		return err
	}
	return nil
}

func (g *conversionGenerator) generateConversionsForSlice(inType, outType reflect.Type) error {
	inElem := inType.Elem()
	outElem := outType.Elem()
	if err := g.generateConversionsBetween(inElem, outElem); err != nil {
		return err
	}
	return nil
}

func (g *conversionGenerator) generateConversionsForStruct(inType, outType reflect.Type) error {
	for i := 0; i < inType.NumField(); i++ {
		inField := inType.Field(i)
		outField, found := outType.FieldByName(inField.Name)
		if !found {
			return fmt.Errorf("couldn't find a corresponding field %v in %v", inField.Name, outType)
		}
		if isComplexType(inField.Type) {
			if err := g.generateConversionsBetween(inField.Type, outField.Type); err != nil {
				return err
			}
		}
	}
	return nil
}

// A buffer of lines that will be written.
type bufferedLine struct {
	line        string
	indentation int
}

type buffer struct {
	lines []bufferedLine
}

func newBuffer() *buffer {
	return &buffer{
		lines: make([]bufferedLine, 0),
	}
}

func (b *buffer) addLine(line string, indent int) {
	b.lines = append(b.lines, bufferedLine{line, indent})
}

func (b *buffer) flushLines(w io.Writer) error {
	for _, line := range b.lines {
		indentation := strings.Repeat("\t", line.indentation)
		fullLine := fmt.Sprintf("%s%s", indentation, line.line)
		if _, err := io.WriteString(w, fullLine); err != nil {
			return err
		}
	}
	return nil
}

type byName []reflect.Type

func (s byName) Len() int {
	return len(s)
}

func (s byName) Less(i, j int) bool {
	fullNameI := s[i].PkgPath() + "/" + s[i].Name()
	fullNameJ := s[j].PkgPath() + "/" + s[j].Name()
	return fullNameI < fullNameJ
}

func (s byName) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (g *conversionGenerator) WriteConversionFunctions(w io.Writer) error {
	// It's desired to print conversion functions always in the same order
	// (e.g. for better tracking of what has really been added).
	var keys []reflect.Type
	for key := range g.convertibles {
		keys = append(keys, key)
	}
	sort.Sort(byName(keys))

	buffer := newBuffer()
	indent := 0
	for _, inType := range keys {
		outType := g.convertibles[inType]
		// All types in g.convertibles are structs.
		if inType.Kind() != reflect.Struct {
			return fmt.Errorf("non-struct conversions are not-supported")
		}
		if err := g.writeConversionForType(buffer, inType, outType, indent); err != nil {
			return err
		}
	}
	if err := buffer.flushLines(w); err != nil {
		return err
	}
	return nil
}

func (g *conversionGenerator) writeRegisterHeader(b *buffer, indent int) {
	b.addLine("func init() {\n", indent)
	b.addLine("err := api.Scheme.AddGeneratedConversionFuncs(\n", indent+1)
}

func (g *conversionGenerator) writeRegisterFooter(b *buffer, indent int) {
	b.addLine(")\n", indent+1)
	b.addLine("if err != nil {\n", indent+1)
	b.addLine("// If one of the conversion functions is malformed, detect it immediately.\n", indent+2)
	b.addLine("panic(err)\n", indent+2)
	b.addLine("}\n", indent+1)
	b.addLine("}\n", indent)
	b.addLine("\n", indent)
}

func (g *conversionGenerator) RegisterConversionFunctions(w io.Writer) error {
	// Write conversion function names alphabetically ordered.
	var names []string
	for inType, outType := range g.convertibles {
		names = append(names, g.conversionFunctionName(inType, outType))
	}
	sort.Strings(names)

	buffer := newBuffer()
	indent := 0
	g.writeRegisterHeader(buffer, indent)
	for _, name := range names {
		buffer.addLine(fmt.Sprintf("%s,\n", name), indent+2)
	}
	g.writeRegisterFooter(buffer, indent)
	if err := buffer.flushLines(w); err != nil {
		return err
	}
	return nil
}

func (g *conversionGenerator) typeName(inType reflect.Type) string {
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

func (g *conversionGenerator) writeDefaultingFunc(b *buffer, inType reflect.Type, indent int) error {
	getStmt := "if defaulting, found := s.DefaultingInterface(reflect.TypeOf(*in)); found {\n"
	b.addLine(getStmt, indent)
	callFormat := "defaulting.(func(*%s))(in)\n"
	callStmt := fmt.Sprintf(callFormat, g.typeName(inType))
	b.addLine(callStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func packageForName(inType reflect.Type) string {
	if inType.PkgPath() == "" {
		return ""
	}
	slices := strings.Split(inType.PkgPath(), "/")
	return slices[len(slices)-1]
}

func (g *conversionGenerator) conversionFunctionName(inType, outType reflect.Type) string {
	funcNameFormat := "convert_%s_%s_To_%s_%s"
	inPkg := packageForName(inType)
	outPkg := packageForName(outType)
	funcName := fmt.Sprintf(funcNameFormat, inPkg, inType.Name(), outPkg, outType.Name())
	return funcName
}

func (g *conversionGenerator) writeHeader(b *buffer, name, inType, outType string, indent int) {
	format := "func %s(in *%s, out *%s, s conversion.Scope) error {\n"
	stmt := fmt.Sprintf(format, name, inType, outType)
	b.addLine(stmt, indent)
}

func (g *conversionGenerator) writeFooter(b *buffer, indent int) {
	b.addLine("return nil\n", indent+1)
	b.addLine("}\n", indent)
}

func (g *conversionGenerator) writeConversionForMap(b *buffer, inField, outField reflect.StructField, indent int) error {
	ifFormat := "if in.%s != nil {\n"
	ifStmt := fmt.Sprintf(ifFormat, inField.Name)
	b.addLine(ifStmt, indent)
	makeFormat := "out.%s = make(%s)\n"
	makeStmt := fmt.Sprintf(makeFormat, outField.Name, g.typeName(outField.Type))
	b.addLine(makeStmt, indent+1)
	forFormat := "for key, val := range in.%s {\n"
	forStmt := fmt.Sprintf(forFormat, inField.Name)
	b.addLine(forStmt, indent+1)

	// Whether we need to explicitly create a new value.
	newValue := false
	if isComplexType(inField.Type.Elem()) || !inField.Type.Elem().ConvertibleTo(outField.Type.Elem()) {
		newValue = true
		newFormat := "newVal := %s{}\n"
		newStmt := fmt.Sprintf(newFormat, g.typeName(outField.Type.Elem()))
		b.addLine(newStmt, indent+2)
		convertStmt := "if err := s.Convert(&val, &newVal, 0); err != nil {\n"
		b.addLine(convertStmt, indent+2)
		b.addLine("return err\n", indent+3)
		b.addLine("}\n", indent+2)
	}
	if inField.Type.Key().ConvertibleTo(outField.Type.Key()) {
		value := "val"
		if newValue {
			value = "newVal"
		}
		assignStmt := ""
		if inField.Type.Key().AssignableTo(outField.Type.Key()) {
			assignStmt = fmt.Sprintf("out.%s[key] = %s\n", outField.Name, value)
		} else {
			assignStmt = fmt.Sprintf("out.%s[%s(key)] = %s\n", outField.Name, g.typeName(outField.Type.Key()), value)
		}
		b.addLine(assignStmt, indent+2)
	} else {
		// TODO(wojtek-t): Support maps with keys that are non-convertible to each other.
		return fmt.Errorf("conversions between unconvertible keys in map are not supported.")
	}
	b.addLine("}\n", indent+1)
	b.addLine("} else {\n", indent)
	nilFormat := "out.%s = nil\n"
	nilStmt := fmt.Sprintf(nilFormat, outField.Name)
	b.addLine(nilStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func (g *conversionGenerator) writeConversionForSlice(b *buffer, inField, outField reflect.StructField, indent int) error {
	ifFormat := "if in.%s != nil {\n"
	ifStmt := fmt.Sprintf(ifFormat, inField.Name)
	b.addLine(ifStmt, indent)
	makeFormat := "out.%s = make(%s, len(in.%s))\n"
	makeStmt := fmt.Sprintf(makeFormat, outField.Name, g.typeName(outField.Type), inField.Name)
	b.addLine(makeStmt, indent+1)
	forFormat := "for i := range in.%s {\n"
	forStmt := fmt.Sprintf(forFormat, inField.Name)
	b.addLine(forStmt, indent+1)

	assigned := false
	switch inField.Type.Elem().Kind() {
	case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
		// Don't copy these via assignment/conversion!
	default:
		// This should handle all simple types.
		if inField.Type.Elem().AssignableTo(outField.Type.Elem()) {
			assignFormat := "out.%s[i] = in.%s[i]\n"
			assignStmt := fmt.Sprintf(assignFormat, outField.Name, inField.Name)
			b.addLine(assignStmt, indent+2)
			assigned = true
		} else if inField.Type.Elem().ConvertibleTo(outField.Type.Elem()) {
			assignFormat := "out.%s[i] = %s(in.%s[i])\n"
			assignStmt := fmt.Sprintf(assignFormat, outField.Name, g.typeName(outField.Type.Elem()), inField.Name)
			b.addLine(assignStmt, indent+2)
			assigned = true
		}
	}
	if !assigned {
		assignStmt := ""
		if g.existsDedicatedConversionFunction(inField.Type.Elem(), outField.Type.Elem()) {
			assignFormat := "if err := %s(&in.%s[i], &out.%s[i], s); err != nil {\n"
			funcName := g.conversionFunctionName(inField.Type.Elem(), outField.Type.Elem())
			assignStmt = fmt.Sprintf(assignFormat, funcName, inField.Name, outField.Name)
		} else {
			assignFormat := "if err := s.Convert(&in.%s[i], &out.%s[i], 0); err != nil {\n"
			assignStmt = fmt.Sprintf(assignFormat, inField.Name, outField.Name)
		}
		b.addLine(assignStmt, indent+2)
		b.addLine("return err\n", indent+3)
		b.addLine("}\n", indent+2)
	}
	b.addLine("}\n", indent+1)
	b.addLine("} else {\n", indent)
	nilFormat := "out.%s = nil\n"
	nilStmt := fmt.Sprintf(nilFormat, outField.Name)
	b.addLine(nilStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func (g *conversionGenerator) writeConversionForPtr(b *buffer, inField, outField reflect.StructField, indent int) error {
	switch inField.Type.Elem().Kind() {
	case reflect.Map, reflect.Ptr, reflect.Slice, reflect.Interface, reflect.Struct:
		// Don't copy these via assignment/conversion!
	default:
		// This should handle pointers to all simple types.
		assignable := inField.Type.Elem().AssignableTo(outField.Type.Elem())
		convertible := inField.Type.Elem().ConvertibleTo(outField.Type.Elem())
		if assignable || convertible {
			ifFormat := "if in.%s != nil {\n"
			ifStmt := fmt.Sprintf(ifFormat, inField.Name)
			b.addLine(ifStmt, indent)
			newFormat := "out.%s = new(%s)\n"
			newStmt := fmt.Sprintf(newFormat, outField.Name, g.typeName(outField.Type.Elem()))
			b.addLine(newStmt, indent+1)
		}
		if assignable {
			assignFormat := "*out.%s = *in.%s\n"
			assignStmt := fmt.Sprintf(assignFormat, outField.Name, inField.Name)
			b.addLine(assignStmt, indent+1)
		} else if convertible {
			assignFormat := "*out.%s = %s(*in.%s)\n"
			assignStmt := fmt.Sprintf(assignFormat, outField.Name, g.typeName(outField.Type.Elem()), inField.Name)
			b.addLine(assignStmt, indent+1)
		}
		if assignable || convertible {
			b.addLine("} else {\n", indent)
			nilFormat := "out.%s = nil\n"
			nilStmt := fmt.Sprintf(nilFormat, outField.Name)
			b.addLine(nilStmt, indent+1)
			b.addLine("}\n", indent)
			return nil
		}
	}

	ifFormat := "if in.%s != nil {\n"
	ifStmt := fmt.Sprintf(ifFormat, inField.Name)
	b.addLine(ifStmt, indent)
	assignStmt := ""
	if g.existsDedicatedConversionFunction(inField.Type.Elem(), outField.Type.Elem()) {
		newFormat := "out.%s = new(%s)\n"
		newStmt := fmt.Sprintf(newFormat, outField.Name, g.typeName(outField.Type.Elem()))
		b.addLine(newStmt, indent+1)
		assignFormat := "if err := %s(in.%s, out.%s, s); err != nil {\n"
		funcName := g.conversionFunctionName(inField.Type.Elem(), outField.Type.Elem())
		assignStmt = fmt.Sprintf(assignFormat, funcName, inField.Name, outField.Name)
	} else {
		assignFormat := "if err := s.Convert(&in.%s, &out.%s, 0); err != nil {\n"
		assignStmt = fmt.Sprintf(assignFormat, inField.Name, outField.Name)
	}
	b.addLine(assignStmt, indent+1)
	b.addLine("return err\n", indent+2)
	b.addLine("}\n", indent+1)
	b.addLine("} else {\n", indent)
	nilFormat := "out.%s = nil\n"
	nilStmt := fmt.Sprintf(nilFormat, outField.Name)
	b.addLine(nilStmt, indent+1)
	b.addLine("}\n", indent)
	return nil
}

func (g *conversionGenerator) writeConversionForStruct(b *buffer, inType, outType reflect.Type, indent int) error {
	for i := 0; i < inType.NumField(); i++ {
		inField := inType.Field(i)
		outField, _ := outType.FieldByName(inField.Name)

		existsConversion := g.scheme.Converter().HasConversionFunc(inField.Type, outField.Type)
		if existsConversion && !g.existsDedicatedConversionFunction(inField.Type, outField.Type) {
			// Use the conversion method that is already defined.
			assignFormat := "if err := s.Convert(&in.%s, &out.%s, 0); err != nil {\n"
			assignStmt := fmt.Sprintf(assignFormat, inField.Name, outField.Name)
			b.addLine(assignStmt, indent)
			b.addLine("return err\n", indent+1)
			b.addLine("}\n", indent)
			continue
		}

		switch inField.Type.Kind() {
		case reflect.Map:
			if err := g.writeConversionForMap(b, inField, outField, indent); err != nil {
				return err
			}
			continue
		case reflect.Ptr:
			if err := g.writeConversionForPtr(b, inField, outField, indent); err != nil {
				return err
			}
			continue
		case reflect.Slice:
			if err := g.writeConversionForSlice(b, inField, outField, indent); err != nil {
				return err
			}
			continue
		case reflect.Interface, reflect.Struct:
			// Don't copy these via assignment/conversion!
		default:
			// This should handle all simple types.
			if inField.Type.AssignableTo(outField.Type) {
				assignFormat := "out.%s = in.%s\n"
				assignStmt := fmt.Sprintf(assignFormat, outField.Name, inField.Name)
				b.addLine(assignStmt, indent)
				continue
			}
			if inField.Type.ConvertibleTo(outField.Type) {
				assignFormat := "out.%s = %s(in.%s)\n"
				assignStmt := fmt.Sprintf(assignFormat, outField.Name, g.typeName(outField.Type), inField.Name)
				b.addLine(assignStmt, indent)
				continue
			}
		}

		assignStmt := ""
		if g.existsDedicatedConversionFunction(inField.Type, outField.Type) {
			assignFormat := "if err := %s(&in.%s, &out.%s, s); err != nil {\n"
			funcName := g.conversionFunctionName(inField.Type, outField.Type)
			assignStmt = fmt.Sprintf(assignFormat, funcName, inField.Name, outField.Name)
		} else {
			assignFormat := "if err := s.Convert(&in.%s, &out.%s, 0); err != nil {\n"
			assignStmt = fmt.Sprintf(assignFormat, inField.Name, outField.Name)
		}
		b.addLine(assignStmt, indent)
		b.addLine("return err\n", indent+1)
		b.addLine("}\n", indent)
	}
	return nil
}

func (g *conversionGenerator) writeConversionForType(b *buffer, inType, outType reflect.Type, indent int) error {
	funcName := g.conversionFunctionName(inType, outType)
	g.writeHeader(b, funcName, g.typeName(inType), g.typeName(outType), indent)
	if err := g.writeDefaultingFunc(b, inType, indent+1); err != nil {
		return err
	}
	switch inType.Kind() {
	case reflect.Struct:
		if err := g.writeConversionForStruct(b, inType, outType, indent+1); err != nil {
			return err
		}
	default:
		return fmt.Errorf("type not supported: %v", inType)
	}
	g.writeFooter(b, indent)
	b.addLine("\n", 0)
	return nil
}

func (g *conversionGenerator) existsConversionFunction(inType, outType reflect.Type) bool {
	if val, found := g.convertibles[inType]; found && val == outType {
		return true
	}
	if val, found := g.convertibles[outType]; found && val == inType {
		return true
	}
	return false
}

// TODO(wojtek-t): We should somehow change the conversion methods registered under:
// pkg/runtime/scheme.go to implement the naming convention for conversion functions
// and get rid of this hack.
type typePair struct {
	inType  reflect.Type
	outType reflect.Type
}

var defaultConversions []typePair = []typePair{
	{reflect.TypeOf([]RawExtension{}), reflect.TypeOf([]Object{})},
	{reflect.TypeOf([]Object{}), reflect.TypeOf([]RawExtension{})},
	{reflect.TypeOf(RawExtension{}), reflect.TypeOf(EmbeddedObject{})},
	{reflect.TypeOf(EmbeddedObject{}), reflect.TypeOf(RawExtension{})},
}

func (g *conversionGenerator) existsDedicatedConversionFunction(inType, outType reflect.Type) bool {
	if inType == outType {
		// Assume that conversion are not defined for "deep copies".
		return false
	}

	if g.existsConversionFunction(inType, outType) {
		return true
	}

	for _, conv := range defaultConversions {
		if conv.inType == inType && conv.outType == outType {
			return false
		}
	}
	if inType.Kind() != outType.Kind() {
		// TODO(wojtek-t): Currently all conversions between types of different kinds are
		// unnamed. Thus we return false here.
		return false
	}
	return g.scheme.Converter().HasConversionFunc(inType, outType)
}

func (g *conversionGenerator) OverwritePackage(pkg, overwrite string) {
	g.pkgOverwrites[pkg] = overwrite
}
