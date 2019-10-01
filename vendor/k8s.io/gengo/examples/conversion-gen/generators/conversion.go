/*
Copyright 2019 The Kubernetes Authors.

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
	"sort"
	"strings"

	"github.com/spf13/pflag"
	"k8s.io/klog"

	"k8s.io/gengo/args"
	conversiongenerator "k8s.io/gengo/examples/conversion-gen/generators/generator"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs struct {
	// Base peer dirs which nearly everybody will use, i.e. outside of Kubernetes core. Peer dirs
	// are declared to make the generator pick up manually written conversion funcs from external
	// packages.
	BasePeerDirs []string

	// Custom peer dirs which are application specific. Peer dirs are declared to make the
	// generator pick up manually written conversion funcs from external packages.
	ExtraPeerDirs []string

	// SkipUnsafe indicates whether to generate unsafe conversions to improve the efficiency
	// of these operations. The unsafe operation is a direct pointer assignment via unsafe
	// (within the allowed uses of unsafe) and is equivalent to a proposed Golang change to
	// allow structs that are identical to be assigned to each other.
	SkipUnsafe bool
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	pflag.CommandLine.StringSliceVar(&ca.BasePeerDirs, "base-peer-dirs", ca.BasePeerDirs,
		"Comma-separated list of apimachinery import paths which are considered, after tag-specified peers, for conversions. Only change these if you have very good reasons.")
	pflag.CommandLine.StringSliceVar(&ca.ExtraPeerDirs, "extra-peer-dirs", ca.ExtraPeerDirs,
		"Application specific comma-separated list of import paths which are considered, after tag-specified peers and base-peer-dirs, for conversions.")
	pflag.CommandLine.BoolVar(&ca.SkipUnsafe, "skip-unsafe", ca.SkipUnsafe,
		"If true, will not generate code using unsafe pointer conversions; resulting code may be slower.")
}

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
		"conversion": conversiongenerator.ConversionNamer(),
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
	scopeVar := conversiongenerator.NewNamedVariable(scopeVarName, types.Ref(conversionPackagePath, "Scope"))
	manualConversionsTracker := conversiongenerator.NewManualConversionsTracker(scopeVar)

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
		} else {
			klog.V(5).Infof("  no tag")
			continue
		}
		skipUnsafe := false
		if customArgs, ok := arguments.CustomArgs.(*CustomArgs); ok {
			peerPkgs = append(peerPkgs, customArgs.BasePeerDirs...)
			peerPkgs = append(peerPkgs, customArgs.ExtraPeerDirs...)
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

		conversionGenerator, err := conversiongenerator.NewConversionGenerator(context, arguments.OutputFileBaseName, typesPkg.Path, pkg.Path, peerPkgs, manualConversionsTracker)
		if err != nil {
			klog.Fatalf(err.Error())
		}
		conversionGenerator.WithTagName(tagName).
			WithFunctionTagName(functionTagName).
			WithMissingFieldsHandler(missingFieldsHandler).
			WithInconvertibleFieldsHandler(inconvertibleFieldsHandler).
			WithUnsupportedTypesHandler(unsupportedTypesHandler).
			WithExternalConversionsHandler(externalConversionsHandler)

		if skipUnsafe {
			conversionGenerator.WithoutUnsafeConversions()
		}

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

func missingFieldsHandler(_, _ conversiongenerator.NamedVariable, member *types.Member, sw *generator.SnippetWriter) error {
	sw.Do("// WARNING: in."+member.Name+" requires manual conversion: does not exist in peer-type\n", nil)
	return fmt.Errorf("field " + member.Name + " requires manual conversion")
}

func inconvertibleFieldsHandler(_, _ conversiongenerator.NamedVariable, inMember, outMember *types.Member, sw *generator.SnippetWriter) error {
	sw.Do("// WARNING: in."+inMember.Name+" requires manual conversion: inconvertible types ("+
		inMember.Type.String()+" vs "+outMember.Type.String()+")\n", nil)
	return fmt.Errorf("field " + inMember.Name + " requires manual conversion")
}

func unsupportedTypesHandler(inVar, _ conversiongenerator.NamedVariable, sw *generator.SnippetWriter) error {
	sw.Do("// FIXME: Type $.|raw$ is unsupported.\n", inVar.Type)

	return nil
}

func externalConversionsHandler(inVar, outVar conversiongenerator.NamedVariable, sw *generator.SnippetWriter) (bool, error) {
	sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
	sw.Do("if err := "+scopeVarName+".Convert("+inVar.Name+", "+outVar.Name+", 0); err != nil {\n", nil)
	sw.Do("return err\n}\n", nil)
	return true, nil
}

// genConversion produces a file with autogenerated conversions.
type genConversion struct {
	*conversiongenerator.ConversionGenerator

	// generatedTypes are the types for which there exist conversion functions.
	generatedTypes []*types.Type
	// outputPackage is the package that the conversion funcs are going to be output to.
	outputPackage string
}

// Namers returns the name system used by ConversionGenerators.
func (g *genConversion) Namers(context *generator.Context) namer.NameSystems {
	namers := g.ConversionGenerator.Namers(context)
	namers["raw"] = namer.NewRawNamer(g.outputPackage, g.ConversionGenerator.ImportTracker)
	return namers
}

func (g *genConversion) Filter(context *generator.Context, t *types.Type) bool {
	result := g.ConversionGenerator.Filter(context, t)
	if result {
		g.generatedTypes = append(g.generatedTypes, t)
	}
	return result
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
		args := argsFromType(t, peerType).With("Scope", types.Ref(conversionPackagePath, "Scope"))
		sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+conversiongenerator.ConversionFunctionName(t, peerType)+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
		args = argsFromType(peerType, t).With("Scope", types.Ref(conversionPackagePath, "Scope"))
		sw.Do("if err := s.AddGeneratedConversionFunc((*$.inType|raw$)(nil), (*$.outType|raw$)(nil), func(a, b interface{}, scope $.Scope|raw$) error { return "+conversiongenerator.ConversionFunctionName(peerType, t)+"(a.(*$.inType|raw$), b.(*$.outType|raw$), scope) }); err != nil { return err }\n", args)
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
func samePkgManualConversionPairs(manualConversions map[conversiongenerator.ConversionPair]*types.Type, outputPackage string) (pairs []conversiongenerator.ConversionPair) {
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

func extractExternalTypesTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[externalTypesTagName]
}

func argsFromType(inType, outType *types.Type) generator.Args {
	return generator.Args{
		"inType":  inType,
		"outType": outType,
	}
}
