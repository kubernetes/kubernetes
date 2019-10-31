/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/klog"

	conversionargs "k8s.io/code-generator/cmd/conversion-gen/args"
	genericconversiongenerator "k8s.io/code-generator/cmd/generic-conversion-gen/generators"
)

// These are the comment tags that carry parameters for conversion generation.
const (
	// e.g., "+k8s:conversion-gen=<peer-pkg>" in doc.go, where <peer-pkg> is the
	// import path of the package the peer types are defined in.
	// e.g., "+k8s:conversion-gen=false" in a type's comment will let
	// conversion-gen skip that type.
	tagName = "k8s:conversion-gen"

	// e.g., "+k8s:conversion-fn=copy-only". copy-only functions that are directly
	// assignable can be inlined instead of invoked. As an example, conversion functions
	// exist that allow types with private fields to be correctly copied between types.
	// These functions are equivalent to a memory assignment, and are necessary for the
	// reflection path, but should not block memory conversion.
	// e.g.,  "+k8s:conversion-fn=drop" to instruct conversion-gen to not use that conversion
	// function.
	functionTagName = "k8s:conversion-fn"

	// e.g. "+k8s:conversion-gen:explicit-from=net/url.Values" in the type comment
	// will result in generating conversion from net/url.Values.
	explicitFromTagName = "k8s:conversion-gen:explicit-from"

	// e.g., "+k8s:conversion-gen=<peer-pkg>" in doc.go, where <peer-pkg> is the
	// import path of the package the peer types are defined in.
	// e.g., "+k8s:conversion-gen=false" in a type's comment will let
	// conversion-gen skip that type.
	// tagName = "k8s:conversion-gen"
	// e.g., "+k8s:conversion-gen-external-types=<type-pkg>" in doc.go, where
	// <type-pkg> is the relative path to the package the types are defined in.
	externalTypesTagName = "k8s:conversion-gen-external-types"
)

// Other constants used by this package.
const (
	runtimePackagePath    = "k8s.io/apimachinery/pkg/runtime"
	conversionPackagePath = "k8s.io/apimachinery/pkg/conversion"

	// scopeVarName is the name of the conversion.Scope variable that's in the argument
	// list of all conversion functions.
	scopeVarName = "s"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"conversion": genericconversiongenerator.ConversionNamer(),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "conversion"
}

func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	var boilerplate []byte
	if arguments.GoHeaderFilePath != "" {
		var err error
		boilerplate, err = arguments.LoadGoBoilerplate()
		if err != nil {
			klog.Fatalf("Failed loading boilerplate: %v", err)
		}
	}

	packages := generator.Packages{}
	header := append([]byte(fmt.Sprintf("// +build !%s\n\n", arguments.GeneratedBuildTag)), boilerplate...)
	scopeVar := genericconversiongenerator.NewNamedVariable(scopeVarName, types.Ref(conversionPackagePath, "Scope"))
	manualConversionsTracker := genericconversiongenerator.NewManualConversionsTracker(scopeVar)

	processed := map[string]bool{}
	for _, i := range context.Inputs {
		// skip duplicates
		if processed[i] {
			continue
		}
		processed[i] = true

		klog.V(5).Infof("considering pkg %q", i)
		pkg := context.Universe[i]
		if pkg == nil {
			// If the input had no Go files, for example.
			continue
		}

		// Only generate conversions for packages which explicitly request it
		// by specifying one or more "+k8s:conversion-gen=<peer-pkg>"
		// in their doc.go file.
		peerPkgs := extractTag(pkg.Comments)
		if peerPkgs != nil {
			klog.V(5).Infof("  tags: %q", peerPkgs)
			if len(peerPkgs) == 1 && peerPkgs[0] == "false" {
				// If a single +k8s:conversion-gen=false tag is defined, we still want
				// the generator to fire for this package for explicit conversions, but
				// we are clearing the peerPkgs to not generate any standard conversions.
				peerPkgs = nil
			}
		} else {
			klog.V(5).Infof("  no tag")
			continue
		}
		skipUnsafe := false
		if customArgs, ok := arguments.CustomArgs.(*conversionargs.CustomArgs); ok {
			if len(peerPkgs) > 0 {
				peerPkgs = append(peerPkgs, customArgs.BasePeerDirs...)
				peerPkgs = append(peerPkgs, customArgs.ExtraPeerDirs...)
			}
			skipUnsafe = customArgs.SkipUnsafe
		}

		// typesPkg is where the versioned types are defined. Sometimes it is
		// different from pkg. For example, kubernetes core/v1 types are defined
		// in vendor/k8s.io/api/core/v1, while pkg is at pkg/api/v1.
		typesPkg := pkg

		// if the external types are not in the same package where the conversion functions to be generated
		externalTypesValues := extractExternalTypesTag(pkg.Comments)
		if externalTypesValues != nil {
			if len(externalTypesValues) != 1 {
				klog.Fatalf("  expect only one value for %q tag, got: %q", externalTypesTagName, externalTypesValues)
			}
			externalTypes := externalTypesValues[0]
			klog.V(5).Infof("  external types tags: %q", externalTypes)
			var err error
			typesPkg, err = context.AddDirectory(externalTypes)
			if err != nil {
				klog.Fatalf("cannot import package %s", externalTypes)
			}
			// update context.Order to the latest context.Universe
			orderer := namer.Orderer{Namer: namer.NewPublicNamer(1)}
			context.Order = orderer.OrderUniverse(context.Universe)
		}

		// if the source path is within a /vendor/ directory (for example,
		// k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/apis/meta/v1), allow
		// generation to output to the proper relative path (under vendor).
		// Otherwise, the generator will create the file in the wrong location
		// in the output directory.
		// TODO: build a more fundamental concept in gengo for dealing with modifications
		// to vendored packages.
		vendorless := func(pkg string) string {
			if pos := strings.LastIndex(pkg, "/vendor/"); pos != -1 {
				return pkg[pos+len("/vendor/"):]
			}
			return pkg
		}
		for i := range peerPkgs {
			peerPkgs[i] = vendorless(peerPkgs[i])
		}

		path := pkg.Path
		// if the source path is within a /vendor/ directory (for example,
		// k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/apis/meta/v1), allow
		// generation to output to the proper relative path (under vendor).
		// Otherwise, the generator will create the file in the wrong location
		// in the output directory.
		// TODO: build a more fundamental concept in gengo for dealing with modifications
		// to vendored packages.
		if strings.HasPrefix(pkg.SourcePath, arguments.OutputBase) {
			expandedPath := strings.TrimPrefix(pkg.SourcePath, arguments.OutputBase)
			if strings.Contains(expandedPath, "/vendor/") {
				path = expandedPath
			}
		}

		conversionGenerator, err := genericconversiongenerator.NewConversionGenerator(context, arguments.OutputFileBaseName, typesPkg.Path, pkg.Path, peerPkgs, manualConversionsTracker)
		if err != nil {
			klog.Fatalf(err.Error())
		}
		conversionGenerator.WithTagName(tagName).
			WithFunctionTagName(functionTagName).
			WithMissingFieldsHandler(missingFieldsHandler).
			WithInconvertibleFieldsHandler(inconvertibleFieldsHandler).
			WithUnsupportedTypesHandler(unsupportedTypesHandler).
			WithExternalConversionsHandler(externalConversionsHandler).
			WithUnsafeConversions(!skipUnsafe)

		packages = append(packages,
			&generator.DefaultPackage{
				PackageName: filepath.Base(pkg.Path),
				PackagePath: path,
				HeaderText:  header,
				GeneratorList: []generator.Generator{
					&genConversion{
						ConversionGenerator: conversionGenerator,
						outputPackage:       pkg.Path,
					},
				},
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == typesPkg.Path
				},
			})
	}

	return packages
}

func missingFieldsHandler(_, _ genericconversiongenerator.NamedVariable, member *types.Member, sw *generator.SnippetWriter) error {
	sw.Do("// WARNING: in."+member.Name+" requires manual conversion: does not exist in peer-type\n", nil)
	return fmt.Errorf("field " + member.Name + " requires manual conversion")
}

func inconvertibleFieldsHandler(_, _ genericconversiongenerator.NamedVariable, inMember, outMember *types.Member, sw *generator.SnippetWriter) error {
	sw.Do("// WARNING: in."+inMember.Name+" requires manual conversion: inconvertible types ("+
		inMember.Type.String()+" vs "+outMember.Type.String()+")\n", nil)
	return fmt.Errorf("field " + inMember.Name + " requires manual conversion")
}

func unsupportedTypesHandler(inVar, _ genericconversiongenerator.NamedVariable, sw *generator.SnippetWriter) error {
	sw.Do("// FIXME: Type $.|raw$ is unsupported.\n", inVar.Type)

	return nil
}

func externalConversionsHandler(inVar, outVar genericconversiongenerator.NamedVariable, sw *generator.SnippetWriter) (bool, error) {
	sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
	sw.Do("if err := "+scopeVarName+".Convert("+inVar.Name+", "+outVar.Name+", 0); err != nil {\n", nil)
	sw.Do("return err\n}\n", nil)
	return true, nil
}

// genConversion produces a file with autogenerated conversions.
type genConversion struct {
	*genericconversiongenerator.ConversionGenerator

	// generatedTypes are the types for which there exist conversion functions.
	generatedTypes []*types.Type
	// outputPackage is the package that the conversion funcs are going to be output to.
	outputPackage string

	explicitConversions []genericconversiongenerator.ConversionPair
}

// Namers returns the name system used by ConversionGenerators.
func (g *genConversion) Namers(context *generator.Context) namer.NameSystems {
	namers := g.ConversionGenerator.Namers(context)
	namers["raw"] = namer.NewRawNamer(g.outputPackage, g.ImportTracker)
	return namers
}

func (g *genConversion) Filter(context *generator.Context, t *types.Type) bool {
	convertibleWithPeer := g.ConversionGenerator.Filter(context, t)
	if convertibleWithPeer {
		g.generatedTypes = append(g.generatedTypes, t)
	}

	return g.explicitlyConvertible(t) || convertibleWithPeer
}

func (g *genConversion) explicitlyConvertible(t *types.Type) bool {
	inTypes := getExplicitFromTypes(t)
	if len(inTypes) == 0 {
		return false
	}

	for i := range inTypes {
		pair := genericconversiongenerator.ConversionPair{
			InType:  &types.Type{Name: inTypes[i]},
			OutType: t,
		}
		g.explicitConversions = append(g.explicitConversions, pair)
	}
	return true
}

func getExplicitFromTypes(t *types.Type) []types.Name {
	comments := append(t.SecondClosestCommentLines, t.CommentLines...)
	paths := extractExplicitFromTag(comments) // TODO wkpo
	result := []types.Name{}
	for _, path := range paths {
		items := strings.Split(path, ".")
		if len(items) != 2 {
			klog.Errorf("Unexpected k8s:conversion-gen:explicit-from tag: %s", path)
			continue
		}
		switch {
		case items[0] == "net/url" && items[1] == "Values":
		default:
			klog.Fatalf("Not supported k8s:conversion-gen:explicit-from tag: %s", path)
		}
		result = append(result, types.Name{Package: items[0], Name: items[1]})
	}
	return result
}

func (g *genConversion) GenerateType(context *generator.Context, t *types.Type, w io.Writer) error {
	if peerType := g.GetPeerTypeFor(context, t); peerType != nil {
		if err := g.ConversionGenerator.GenerateType(context, t, w); err != nil {
			return err
		}
	}

	sw := generator.NewSnippetWriter(w, context, "$", "$")
	for _, inTypeName := range getExplicitFromTypes(t) {
		inPkg, ok := context.Universe[inTypeName.Package]
		if !ok {
			klog.Errorf("Unrecognized package: %s", inTypeName.Package)
			continue
		}
		inType, ok := inPkg.Types[inTypeName.Name]
		if !ok {
			klog.Errorf("Unrecognized type in package %s: %s", inTypeName.Package, inTypeName.Name)
			continue
		}
		switch {
		case inType.Name.Package == "net/url" && inType.Name.Name == "Values":
			g.generateFromUrlValues(inType, t, sw)
		default:
			klog.Errorf("Not supported input type: %#v", inType.Name)
		}
	}

	return sw.Error()
}

func (g *genConversion) generateFromUrlValues(inType, outType *types.Type, sw *generator.SnippetWriter) {
	args := generator.Args{
		"inType":  inType,
		"outType": outType,
		"Scope":   types.Ref(conversionPackagePath, "Scope"),
	}
	conversionFunctionName := genericconversiongenerator.ConversionFunctionName(inType, outType)

	sw.Do("func auto"+conversionFunctionName+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
	for _, outMember := range outType.Members {
		jsonTag := reflect.StructTag(outMember.Tags).Get("json")
		index := strings.Index(jsonTag, ",")
		if index == -1 {
			index = len(jsonTag)
		}
		if index == 0 {
			memberArgs := generator.Args{
				"name": outMember.Name,
			}
			sw.Do("// WARNING: Field $.name$ does not have json tag, skipping.\n\n", memberArgs)
			continue
		}
		memberArgs := generator.Args{
			"name": outMember.Name,
			"tag":  jsonTag[:index],
		}
		sw.Do("if values, ok := map[string][]string(*in)[\"$.tag$\"]; ok && len(values) > 0 {\n", memberArgs)
		g.fromValuesEntry(inType.Underlying.Elem, outMember, sw)
		sw.Do("} else {\n", nil)
		g.setZeroValue(outMember, sw)
		sw.Do("}\n", nil)
	}
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	if _, found := g.Preexists(inType, outType); found {
		// There is a public manual Conversion method: use it.
	} else {
		// Emit a public conversion function.
		sw.Do("// "+conversionFunctionName+" is an autogenerated conversion function.\n", args)
		sw.Do("func "+conversionFunctionName+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
		sw.Do("return auto"+conversionFunctionName+"(in, out, s)\n", args)
		sw.Do("}\n\n", nil)
	}
}

func (g *genConversion) fromValuesEntry(inType *types.Type, outMember types.Member, sw *generator.SnippetWriter) {
	memberArgs := generator.Args{
		"name": outMember.Name,
		"type": outMember.Type,
	}
	if function, ok := g.Preexists(inType, outMember.Type); ok {
		args := memberArgs.With("function", function)
		sw.Do("if err := $.function|raw$(&values, &out.$.name$, s); err != nil {\n", args)
		sw.Do("return err\n", nil)
		sw.Do("}\n", nil)
		return
	}
	switch {
	case outMember.Type == types.String:
		sw.Do("out.$.name$ = values[0]\n", memberArgs)
	case g.CanUseUnsafeConversion(inType, outMember.Type):
		args := memberArgs.With("Pointer", types.Ref("unsafe", "Pointer"))
		switch inType.Kind {
		case types.Pointer:
			sw.Do("out.$.name$ = ($.type|raw$)($.Pointer|raw$(&values))\n", args)
		case types.Map, types.Slice:
			sw.Do("out.$.name$ = *(*$.type|raw$)($.Pointer|raw$(&values))\n", args)
		default:
			// TODO: Support other types to allow more auto-conversions.
			sw.Do("// FIXME: out.$.name$ is of not yet supported type and requires manual conversion\n", memberArgs)
		}
	default:
		// TODO: Support other types to allow more auto-conversions.
		sw.Do("// FIXME: out.$.name$ is of not yet supported type and requires manual conversion\n", memberArgs)
	}
}

func (g *genConversion) setZeroValue(outMember types.Member, sw *generator.SnippetWriter) {
	outMemberType := unwrapAlias(outMember.Type)
	memberArgs := generator.Args{
		"name":  outMember.Name,
		"alias": outMember.Type,
		"type":  outMemberType,
	}

	switch outMemberType.Kind {
	case types.Builtin:
		switch outMemberType {
		case types.String:
			sw.Do("out.$.name$ = \"\"\n", memberArgs)
		case types.Int64, types.Int32, types.Int16, types.Int, types.Uint64, types.Uint32, types.Uint16, types.Uint:
			sw.Do("out.$.name$ = 0\n", memberArgs)
		case types.Uintptr, types.Byte:
			sw.Do("out.$.name$ = 0\n", memberArgs)
		case types.Float64, types.Float32, types.Float:
			sw.Do("out.$.name$ = 0\n", memberArgs)
		case types.Bool:
			sw.Do("out.$.name$ = false\n", memberArgs)
		default:
			sw.Do("// FIXME: out.$.name$ is of unsupported type and requires manual conversion\n", memberArgs)
		}
	case types.Struct:
		if outMemberType == outMember.Type {
			sw.Do("out.$.name$ = $.type|raw${}\n", memberArgs)
		} else {
			sw.Do("out.$.name$ = $.alias|raw$($.type|raw${})\n", memberArgs)
		}
	case types.Map, types.Slice, types.Pointer:
		sw.Do("out.$.name$ = nil\n", memberArgs)
	case types.Alias:
		// outMemberType was already unwrapped from aliases - so that should never happen.
		sw.Do("// FIXME: unexpected error for out.$.name$\n", memberArgs)
	case types.Interface, types.Array:
		sw.Do("out.$.name$ = nil\n", memberArgs)
	default:
		sw.Do("// FIXME: out.$.name$ is of unsupported type and requires manual conversion\n", memberArgs)
	}
}

// unwrapAlias recurses down aliased types to find the bedrock type.
func unwrapAlias(in *types.Type) *types.Type {
	for in.Kind == types.Alias {
		in = in.Underlying
	}
	return in
}

func (g *genConversion) Init(context *generator.Context, writer io.Writer) error {
	sw := generator.NewSnippetWriter(writer, context, "$", "$")

	sw.Do("func init() {\n", nil)
	sw.Do("localSchemeBuilder.Register(RegisterConversions)\n", nil)
	sw.Do("}\n", nil)

	scheme := context.Universe.Type(types.Name{Package: runtimePackagePath, Name: "Scheme"})
	schemePtr := &types.Type{
		Kind: types.Pointer,
		Elem: scheme,
	}
	sw.Do("// RegisterConversions adds conversion functions to the given scheme.\n", nil)
	sw.Do("// Public to allow building arbitrary schemes.\n", nil)
	sw.Do("func RegisterConversions(s $.|raw$) error {\n", schemePtr)
	for _, t := range g.generatedTypes {
		peerType := g.GetPeerTypeFor(context, t)
		if _, found := g.Preexists(t, peerType); !found {
			args := argsFromType(t, peerType).With("Scope", types.Ref(conversionPackagePath, "Scope"))
			sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+genericconversiongenerator.ConversionFunctionName(t, peerType)+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
		}
		if _, found := g.Preexists(peerType, t); !found {
			args := argsFromType(peerType, t).With("Scope", types.Ref(conversionPackagePath, "Scope"))
			sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+genericconversiongenerator.ConversionFunctionName(peerType, t)+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
		}
	}

	for _, pair := range g.explicitConversions {
		args := argsFromType(pair.InType, pair.OutType).With("Scope", types.Ref(conversionPackagePath, "Scope"))
		sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+genericconversiongenerator.ConversionFunctionName(pair.InType, pair.OutType)+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
	}

	manualConversions := g.ManualConversions()
	for _, pair := range samePkgManualConversionPairs(manualConversions, g.outputPackage) {
		args := argsFromType(pair.InType, pair.OutType).With("Scope", types.Ref(conversionPackagePath, "Scope")).With("fn", manualConversions[pair])
		sw.Do("if err := s.AddConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return $.fn|raw$(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
	}

	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)
	return sw.Error()
}

// samePkgManualConversionPairs returns the list of all conversion pairs from the output package.
func samePkgManualConversionPairs(manualConversions map[genericconversiongenerator.ConversionPair]*types.Type, outputPackage string) (pairs []genericconversiongenerator.ConversionPair) {
	for pair, t := range manualConversions {
		if t.Name.Package == outputPackage {
			pairs = append(pairs, pair)
		}
	}

	// sort by name of the conversion function
	sort.Slice(pairs, func(i, j int) bool {
		if manualConversions[pairs[i]].Name.Name < manualConversions[pairs[j]].Name.Name {
			return true
		}
		return false
	})

	return
}

func extractTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[tagName]
}

func extractExplicitFromTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[explicitFromTagName]
}

func extractExternalTypesTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[externalTypesTagName]
}

func argsFromType(inType, outType *types.Type) generator.Args {
	return generator.Args{
		"inType":  inType,
		"outType": outType,
	}
}
