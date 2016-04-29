/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package generators

import (
	"bytes"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

// TODO: This is created only to reduce number of changes in a single PR.
// Remove it and use PublicNamer instead.
func conversionNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Join: func(pre string, in []string, post string) string {
			return strings.Join(in, "_")
		},
		PrependPackageNames: 1,
	}
}

func defaultFnNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Prefix: "SetDefaults_",
		Join: func(pre string, in []string, post string) string {
			return pre + strings.Join(in, "_") + post
		},
	}
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public":    conversionNamer(),
		"raw":       namer.NewRawNamer("", nil),
		"defaultfn": defaultFnNamer(),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

var fallbackPackages = []string{
	"k8s.io/kubernetes/pkg/api/unversioned",
	"k8s.io/kubernetes/pkg/apis/extensions",
	"k8s.io/kubernetes/pkg/apis/batch",
}

func getInternalTypeFor(context *generator.Context, t *types.Type) (*types.Type, bool) {
	internalPackage := filepath.Dir(t.Name.Package)
	if !context.Universe.Package(internalPackage).Has(t.Name.Name) {
		for _, fallbackPackage := range fallbackPackages {
			if fallbackPackage == t.Name.Package || !context.Universe.Package(fallbackPackage).Has(t.Name.Name) {
				continue
			}
			return context.Universe.Package(fallbackPackage).Type(t.Name.Name), true
		}
		return nil, false
	}
	return context.Universe.Package(internalPackage).Type(t.Name.Name), true
}

type conversionType struct {
	inType  *types.Type
	outType *types.Type
}

// All of the types in conversions map are of type "DeclarationOf" with
// the underlying type being "Func".
type conversions map[conversionType]*types.Type

// Returns all already existing conversion functions that we are able to find.
func existingConversionFunctions(context *generator.Context) conversions {
	scopeName := types.Name{Package: conversionPackagePath, Name: "Scope"}
	errorName := types.Name{Package: "", Name: "error"}
	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")

	preexisting := make(conversions)
	for _, p := range context.Universe {
		for _, f := range p.Functions {
			if f.Underlying == nil || f.Underlying.Kind != types.Func {
				glog.Errorf("Malformed function: %#v", f)
				continue
			}
			if f.Underlying.Signature == nil {
				glog.Errorf("Function without signature: %#v", f)
				continue
			}
			signature := f.Underlying.Signature
			// Check whether the function is conversion function.
			// Note that all of them have signature:
			// func Convert_inType_To_outType(inType, outType, conversion.Scope) error
			if signature.Receiver != nil {
				continue
			}
			if len(signature.Parameters) != 3 || signature.Parameters[2].Name != scopeName {
				continue
			}
			if len(signature.Results) != 1 || signature.Results[0].Name != errorName {
				continue
			}
			inType := signature.Parameters[0]
			outType := signature.Parameters[1]
			if inType.Kind != types.Pointer || outType.Kind != types.Pointer {
				continue
			}
			// Now check if the name satisfies the convention.
			args := argsFromType(inType.Elem, outType.Elem)
			sw.Do("Convert_$.inType|public$_To_$.outType|public$", args)
			if f.Name.Name == buffer.String() {
				key := conversionType{inType.Elem, outType.Elem}
				if v, ok := preexisting[key]; ok && v != nil {
					panic(fmt.Sprintf("duplicate static conversion defined: %#v", key))
				}
				preexisting[key] = f
			}
			buffer.Reset()
		}
	}
	return preexisting
}

// All of the types in conversions map are of type "DeclarationOf" with
// the underlying type being "Func".
type defaulters map[*types.Type]*types.Type

// Returns all already existing defaulting functions that we are able to find.
func existingDefaultingFunctions(context *generator.Context) defaulters {
	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")

	preexisting := make(defaulters)
	for _, p := range context.Universe {
		for _, f := range p.Functions {
			if f.Underlying == nil || f.Underlying.Kind != types.Func {
				glog.Errorf("Malformed function: %#v", f)
				continue
			}
			if f.Underlying.Signature == nil {
				glog.Errorf("Function without signature: %#v", f)
				continue
			}
			signature := f.Underlying.Signature
			// Check whether the function is conversion function.
			// Note that all of them have signature:
			// func Convert_inType_To_outType(inType, outType, conversion.Scope) error
			if signature.Receiver != nil {
				continue
			}
			if len(signature.Parameters) != 1 {
				continue
			}
			if len(signature.Results) != 0 {
				continue
			}
			inType := signature.Parameters[0]
			if inType.Kind != types.Pointer {
				continue
			}
			// Now check if the name satisfies the convention.
			args := defaultingArgsFromType(inType.Elem)
			sw.Do("$.inType|defaultfn$", args)
			if f.Name.Name == buffer.String() {
				key := inType.Elem
				if v, ok := preexisting[key]; ok && v != nil {
					panic(fmt.Sprintf("duplicate static defaulter defined: %#v", key))
				}
				preexisting[key] = f
			}
			buffer.Reset()
		}
	}
	return preexisting
}

func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		glog.Fatalf("Failed loading boilerplate: %v", err)
	}

	inputs := sets.NewString(arguments.InputDirs...)
	packages := generator.Packages{}
	header := append([]byte(
		`// +build !ignore_autogenerated

`), boilerplate...)
	header = append(header, []byte(
		`
// This file was autogenerated by conversion-gen. Do not edit it manually!

`)...)

	// Compute all pre-existing conversion functions.
	preexisting := existingConversionFunctions(context)
	preexistingDefaults := existingDefaultingFunctions(context)

	// We are generating conversions only for packages that are explicitly
	// passed as InputDir, and only for those that have a corresponding type
	// (in the directory one above) and can be automatically converted to.
	for _, p := range context.Universe {
		path := p.Path
		if !inputs.Has(path) {
			continue
		}
		// Only generate conversions for package which explicitly requested it
		// byt setting "+genversion=true" in their doc.go file.
		filtered := false
		for _, comment := range p.DocComments {
			comment := strings.Trim(comment, "//")
			if types.ExtractCommentTags("+", comment)["genconversion"] == "true" {
				filtered = true
			}
		}
		if !filtered {
			continue
		}

		convertibleType := false
		for _, t := range p.Types {
			// Check whether this type can be auto-converted to the internal
			// version.
			internalType, exists := getInternalTypeFor(context, t)
			if !exists {
				// There is no corresponding type in the internal package.
				continue
			}
			// We won't be able to convert to private type.
			if namer.IsPrivateGoName(internalType.Name.Name) {
				continue
			}
			// If we can generate conversion in any direction, we should
			// generate this package.
			if isConvertible(t, internalType, preexisting) || isConvertible(internalType, t, preexisting) {
				convertibleType = true
			}
		}

		if convertibleType {
			packages = append(packages,
				&generator.DefaultPackage{
					PackageName: filepath.Base(path),
					PackagePath: path,
					HeaderText:  header,
					GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
						generators = []generator.Generator{}
						generators = append(
							generators, NewGenConversion("conversion_generated", path, preexisting, preexistingDefaults))
						return generators
					},
					FilterFunc: func(c *generator.Context, t *types.Type) bool {
						return t.Name.Package == path
					},
				})
		}
	}
	return packages
}

func findMember(t *types.Type, name string) (types.Member, bool) {
	if t.Kind != types.Struct {
		return types.Member{}, false
	}
	for _, member := range t.Members {
		if member.Name == name {
			return member, true
		}
	}
	return types.Member{}, false
}

func isConvertible(in, out *types.Type, preexisting conversions) bool {
	// If there is pre-existing conversion function, return true immediately.
	if _, ok := preexisting[conversionType{in, out}]; ok {
		return true
	}
	return isDirectlyConvertible(in, out, preexisting)
}

func isDirectlyConvertible(in, out *types.Type, preexisting conversions) bool {
	// If one of the types is Alias, resolve it.
	if in.Kind == types.Alias {
		return isConvertible(in.Underlying, out, preexisting)
	}
	if out.Kind == types.Alias {
		return isConvertible(in, out.Underlying, preexisting)
	}

	if in.Kind != out.Kind {
		return false
	}
	switch in.Kind {
	case types.Builtin, types.Struct, types.Map, types.Slice, types.Pointer:
	default:
		// We don't support conversion of other types yet.
		return false
	}
	switch out.Kind {
	case types.Builtin, types.Struct, types.Map, types.Slice, types.Pointer:
	default:
		// We don't support conversion of other types yet.
		return false
	}

	switch in.Kind {
	case types.Builtin:
		if in == out {
			return true
		}
		// TODO: Support more conversion types.
		return types.IsInteger(in) && types.IsInteger(out)
	case types.Struct:
		convertible := true
		for _, inMember := range in.Members {
			// Check if there is an out member with that name.
			outMember, found := findMember(out, inMember.Name)
			if !found {
				// Check if the member doesn't have comment:
				// "+ genconversion=false"
				// comment to ignore this field for conversion.
				// TODO: Switch to SecondClosestCommentLines.
				if types.ExtractCommentTags("+", inMember.CommentLines)["genconversion"] == "false" {
					continue
				}
				return false
			}
			convertible = convertible && isConvertible(inMember.Type, outMember.Type, preexisting)
		}
		return convertible
	case types.Map:
		return isConvertible(in.Key, out.Key, preexisting) && isConvertible(in.Elem, out.Elem, preexisting)
	case types.Slice:
		return isConvertible(in.Elem, out.Elem, preexisting)
	case types.Pointer:
		return isConvertible(in.Elem, out.Elem, preexisting)
	}
	glog.Fatalf("All other types should be filtered before")
	return false
}

const (
	apiPackagePath        = "k8s.io/kubernetes/pkg/api"
	conversionPackagePath = "k8s.io/kubernetes/pkg/conversion"
)

// genConversion produces a file with a autogenerated conversions.
type genConversion struct {
	generator.DefaultGen
	targetPackage string
	preexisting   conversions
	defaulters    defaulters
	imports       namer.ImportTracker
	typesForInit  []conversionType
}

func NewGenConversion(sanitizedName, targetPackage string, preexisting conversions, defaulters defaulters) generator.Generator {
	return &genConversion{
		DefaultGen: generator.DefaultGen{
			OptionalName: sanitizedName,
		},
		targetPackage: targetPackage,
		preexisting:   preexisting,
		defaulters:    defaulters,
		imports:       generator.NewImportTracker(),
		typesForInit:  make([]conversionType, 0),
	}
}

func (g *genConversion) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{"raw": namer.NewRawNamer(g.targetPackage, g.imports)}
}

func (g *genConversion) convertibleOnlyWithinPackage(inType, outType *types.Type) bool {
	var t *types.Type
	var other *types.Type
	if inType.Name.Package == g.targetPackage {
		t, other = inType, outType
	} else {
		t, other = outType, inType
	}

	if t.Name.Package != g.targetPackage {
		return false
	}
	if types.ExtractCommentTags("+", t.CommentLines)["genconversion"] == "false" {
		return false
	}
	// TODO: Consider generating functions for other kinds too.
	if t.Kind != types.Struct {
		return false
	}
	// Also, filter out private types.
	if namer.IsPrivateGoName(other.Name.Name) {
		return false
	}
	return true
}

func (g *genConversion) Filter(c *generator.Context, t *types.Type) bool {
	internalType, exists := getInternalTypeFor(c, t)
	if !exists {
		return false
	}
	if !g.convertibleOnlyWithinPackage(t, internalType) {
		return false
	}
	// We explicitly return true if any conversion is possible - this needs
	// to be checked again while generating code for that type.
	convertible := false
	if isConvertible(t, internalType, g.preexisting) {
		g.typesForInit = append(g.typesForInit, conversionType{t, internalType})
		convertible = true
	}
	if isConvertible(internalType, t, g.preexisting) {
		g.typesForInit = append(g.typesForInit, conversionType{internalType, t})
		convertible = true
	}
	return convertible
}

func (g *genConversion) isOtherPackage(pkg string) bool {
	if pkg == g.targetPackage {
		return false
	}
	if strings.HasSuffix(pkg, `"`+g.targetPackage+`"`) {
		return false
	}
	return true
}

func (g *genConversion) Imports(c *generator.Context) (imports []string) {
	var importLines []string
	if g.isOtherPackage(apiPackagePath) {
		importLines = append(importLines, "api \""+apiPackagePath+"\"")
	}
	if g.isOtherPackage(conversionPackagePath) {
		importLines = append(importLines, "conversion \""+conversionPackagePath+"\"")
	}
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func argsFromType(inType, outType *types.Type) interface{} {
	return map[string]interface{}{
		"inType":  inType,
		"outType": outType,
	}
}

func defaultingArgsFromType(inType *types.Type) interface{} {
	return map[string]interface{}{
		"inType": inType,
	}
}
func (g *genConversion) funcNameTmpl(inType, outType *types.Type) string {
	tmpl := "Convert_$.inType|public$_To_$.outType|public$"
	g.imports.AddType(inType)
	g.imports.AddType(outType)
	return tmpl
}

func (g *genConversion) preexists(inType, outType *types.Type) (*types.Type, bool) {
	function, ok := g.preexisting[conversionType{inType, outType}]
	return function, ok
}

func (g *genConversion) Init(c *generator.Context, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	sw.Do("func init() {\n", nil)
	if g.targetPackage == apiPackagePath {
		sw.Do("if err := Scheme.AddGeneratedConversionFuncs(\n", nil)
	} else {
		sw.Do("if err := api.Scheme.AddGeneratedConversionFuncs(\n", nil)
	}
	for _, conv := range g.typesForInit {
		funcName := g.funcNameTmpl(conv.inType, conv.outType)
		sw.Do(fmt.Sprintf("%s,\n", funcName), argsFromType(conv.inType, conv.outType))
	}
	sw.Do("); err != nil {\n", nil)
	sw.Do("// if one of the conversion functions is malformed, detect it immediately.\n", nil)
	sw.Do("panic(err)\n", nil)
	sw.Do("}\n", nil)
	sw.Do("}\n\n", nil)
	return sw.Error()
}

func (g *genConversion) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	internalType, _ := getInternalTypeFor(c, t)
	if isDirectlyConvertible(t, internalType, g.preexisting) {
		g.generateConversion(t, internalType, sw)
	}
	if isDirectlyConvertible(internalType, t, g.preexisting) {
		g.generateConversion(internalType, t, sw)
	}
	return sw.Error()
}

func (g *genConversion) generateConversion(inType, outType *types.Type, sw *generator.SnippetWriter) {
	funcName := g.funcNameTmpl(inType, outType)
	if g.targetPackage == conversionPackagePath {
		sw.Do(fmt.Sprintf("func auto%s(in *$.inType|raw$, out *$.outType|raw$, s Scope) error {\n", funcName), argsFromType(inType, outType))
	} else {
		sw.Do(fmt.Sprintf("func auto%s(in *$.inType|raw$, out *$.outType|raw$, s conversion.Scope) error {\n", funcName), argsFromType(inType, outType))
	}
	// if no defaulter of form SetDefaults_XXX is defined, do not inline a check for defaulting.
	if function, ok := g.defaulters[inType]; ok {
		sw.Do("$.|raw$(in)\n", function)
	}

	g.generateFor(inType, outType, sw)
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	// If there is no public preexisting Convert method, generate it.
	if _, ok := g.preexists(inType, outType); !ok {
		if g.targetPackage == conversionPackagePath {
			sw.Do(fmt.Sprintf("func %s(in *$.inType|raw$, out *$.outType|raw$, s Scope) error {\n", funcName), argsFromType(inType, outType))
		} else {
			sw.Do(fmt.Sprintf("func %s(in *$.inType|raw$, out *$.outType|raw$, s conversion.Scope) error {\n", funcName), argsFromType(inType, outType))
		}
		sw.Do(fmt.Sprintf("return auto%s(in, out, s)\n", funcName), argsFromType(inType, outType))
		sw.Do("}\n\n", nil)
	}
}

// we use the system of shadowing 'in' and 'out' so that the same code is valid
// at any nesting level. This makes the autogenerator easy to understand, and
// the compiler shouldn't care.
func (g *genConversion) generateFor(inType, outType *types.Type, sw *generator.SnippetWriter) {
	var f func(*types.Type, *types.Type, *generator.SnippetWriter)
	switch inType.Kind {
	case types.Builtin:
		f = g.doBuiltin
	case types.Map:
		f = g.doMap
	case types.Slice:
		f = g.doSlice
	case types.Struct:
		f = g.doStruct
	case types.Pointer:
		f = g.doPointer
	case types.Alias:
		f = g.doAlias
	default:
		f = g.doUnknown
	}
	f(inType, outType, sw)
}

func (g *genConversion) doBuiltin(inType, outType *types.Type, sw *generator.SnippetWriter) {
	if inType == outType {
		sw.Do("*out = *in\n", nil)
	} else {
		sw.Do("*out = $.|raw$(*in)\n", outType)
	}
}

func (g *genConversion) doMap(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = make($.|raw$, len(*in))\n", outType)
	if outType.Key.IsAssignable() {
		sw.Do("for key, val := range *in {\n", nil)
		if outType.Elem.IsAssignable() {
			if inType.Key == outType.Key {
				sw.Do("(*out)[key] = ", nil)
			} else {
				sw.Do("(*out)[$.|raw$(key)] = ", outType.Key)
			}
			if inType.Elem == outType.Elem {
				sw.Do("val\n", nil)
			} else {
				sw.Do("$.|raw$(val)\n", outType.Elem)
			}
		} else {
			sw.Do("newVal := new($.|raw$)\n", outType.Elem)
			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				sw.Do("if err := $.|raw$(&val, newVal, s); err != nil {\n", function)
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				funcName := g.funcNameTmpl(inType.Elem, outType.Elem)
				sw.Do(fmt.Sprintf("if err := %s(&val, newVal, s); err != nil {\n", funcName), argsFromType(inType.Elem, outType.Elem))
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&val, newVal, 0); err != nil {\n", nil)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
			if inType.Key == outType.Key {
				sw.Do("(*out)[key] = *newVal\n", nil)
			} else {
				sw.Do("(*out)[$.|raw$(key)] = *newVal\n", outType.Key)
			}
		}
	} else {
		// TODO: Implement it when necessary.
		sw.Do("for range *in {\n", nil)
		sw.Do("// FIXME: Converting unassignable keys unsupported $.|raw$\n", inType.Key)
	}
	sw.Do("}\n", nil)
}

func (g *genConversion) doSlice(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = make($.|raw$, len(*in))\n", outType)
	if inType.Elem == outType.Elem && inType.Elem.Kind == types.Builtin {
		sw.Do("copy(*out, *in)\n", nil)
	} else {
		sw.Do("for i := range *in {\n", nil)
		if outType.Elem.IsAssignable() {
			if inType.Elem == outType.Elem {
				sw.Do("(*out)[i] = (*in)[i]\n", nil)
			} else {
				sw.Do("(*out)[i] = $.|raw$((*in)[i])\n", outType.Elem)
			}
		} else {
			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				sw.Do("if err := $.|raw$(&(*in)[i], &(*out)[i], s); err != nil {\n", function)
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				funcName := g.funcNameTmpl(inType.Elem, outType.Elem)
				sw.Do(fmt.Sprintf("if err := %s(&(*in)[i], &(*out)[i], s); err != nil {\n", funcName), argsFromType(inType.Elem, outType.Elem))
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&(*in)[i], &(*out)[i], 0); err != nil {\n", nil)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		}
		sw.Do("}\n", nil)
	}
}

func (g *genConversion) doStruct(inType, outType *types.Type, sw *generator.SnippetWriter) {
	for _, m := range inType.Members {
		outMember, isOutMember := findMember(outType, m.Name)
		if !isOutMember {
			// Since this object wasn't filtered out, this means that
			// this field has "genconversion=false" comment to ignore it.
			continue
		}
		args := map[string]interface{}{
			"inType":  m.Type,
			"outType": outMember.Type,
			"name":    m.Name,
		}
		if function, ok := g.preexists(m.Type, outMember.Type); ok {
			args["function"] = function
			sw.Do("if err := $.function|raw$(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
			continue
		}
		switch m.Type.Kind {
		case types.Builtin:
			if m.Type == outMember.Type {
				sw.Do("out.$.name$ = in.$.name$\n", args)
			} else {
				sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
			}
		case types.Map, types.Slice, types.Pointer:
			sw.Do("if in.$.name$ != nil {\n", args)
			sw.Do("in, out := &in.$.name$, &out.$.name$\n", args)
			g.generateFor(m.Type, outMember.Type, sw)
			sw.Do("} else {\n", nil)
			sw.Do("out.$.name$ = nil\n", args)
			sw.Do("}\n", nil)
		case types.Struct:
			if g.convertibleOnlyWithinPackage(m.Type, outMember.Type) {
				funcName := g.funcNameTmpl(m.Type, outMember.Type)
				sw.Do(fmt.Sprintf("if err := %s(&in.$.name$, &out.$.name$, s); err != nil {\n", funcName), args)
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		case types.Alias:
			if outMember.Type.IsAssignable() {
				if m.Type == outMember.Type {
					sw.Do("out.$.name$ = in.$.name$\n", args)
				} else {
					sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
				}
			} else {
				if g.convertibleOnlyWithinPackage(m.Type, outMember.Type) {
					funcName := g.funcNameTmpl(m.Type, outMember.Type)
					sw.Do(fmt.Sprintf("if err := %s(&in.$.name$, &out.$.name$, s); err != nil {\n", funcName), args)
				} else {
					sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
					sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
				}
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
			}
		default:
			if g.convertibleOnlyWithinPackage(m.Type, outMember.Type) {
				funcName := g.funcNameTmpl(m.Type, outMember.Type)
				sw.Do(fmt.Sprintf("if err := %s(&in.$.name$, &out.$.name$, s); err != nil {\n", funcName), args)
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		}
	}
}

func (g *genConversion) doPointer(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = new($.Elem|raw$)\n", outType)
	if outType.Elem.IsAssignable() {
		if inType.Elem == outType.Elem {
			sw.Do("**out = **in\n", nil)
		} else {
			sw.Do("**out = $.|raw$(**in)\n", outType.Elem)
		}
	} else {
		if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
			sw.Do("if err := $.|raw$(*in, *out, s); err != nil {\n", function)
		} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
			funcName := g.funcNameTmpl(inType.Elem, outType.Elem)
			sw.Do(fmt.Sprintf("if err := %s(*in, *out, s); err != nil {\n", funcName), argsFromType(inType.Elem, outType.Elem))
		} else {
			sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
			sw.Do("if err := s.Convert(*in, *out, 0); err != nil {\n", nil)
		}
		sw.Do("return err\n", nil)
		sw.Do("}\n", nil)
	}
}

func (g *genConversion) doAlias(inType, outType *types.Type, sw *generator.SnippetWriter) {
	// TODO: Add support for aliases.
	g.doUnknown(inType, outType, sw)
}

func (g *genConversion) doUnknown(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("// FIXME: Type $.|raw$ is unsupported.\n", inType)
}
