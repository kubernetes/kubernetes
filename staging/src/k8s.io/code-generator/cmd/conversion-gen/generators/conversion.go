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
	"bytes"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"github.com/golang/glog"
)

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs struct {
	ExtraPeerDirs []string // Always consider these as last-ditch possibilities for conversions.
	// Skipunsafe indicates whether to generate unsafe conversions to improve the efficiency
	// of these operations. The unsafe operation is a direct pointer assignment via unsafe
	// (within the allowed uses of unsafe) and is equivalent to a proposed Golang change to
	// allow structs that are identical to be assigned to each other.
	SkipUnsafe bool
}

// These are the comment tags that carry parameters for conversion generation.
const (
	// e.g., "+k8s:conversion-gen=<peer-pkg>" in doc.go, where <peer-pkg> is the
	// import path of the package the peer types are defined in.
	// e.g., "+k8s:conversion-gen=false" in a type's comment will let
	// conversion-gen skip that type.
	tagName = "k8s:conversion-gen"
	// e.g., "+k8s:conversion-gen-external-types=<type-pkg>" in doc.go, where
	// <type-pkg> is the relative path to the package the types are defined in.
	externalTypesTagName = "k8s:conversion-gen-external-types"
)

func extractTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[tagName]
}

func extractExternalTypesTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[externalTypesTagName]
}

func isCopyOnly(comments []string) bool {
	values := types.ExtractCommentTags("+", comments)["k8s:conversion-fn"]
	return len(values) == 1 && values[0] == "copy-only"
}

func isDrop(comments []string) bool {
	values := types.ExtractCommentTags("+", comments)["k8s:conversion-fn"]
	return len(values) == 1 && values[0] == "drop"
}

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

func getPeerTypeFor(context *generator.Context, t *types.Type, potenialPeerPkgs []string) *types.Type {
	for _, ppp := range potenialPeerPkgs {
		p := context.Universe.Package(ppp)
		if p == nil {
			continue
		}
		if p.Has(t.Name.Name) {
			return p.Type(t.Name.Name)
		}
	}
	return nil
}

type conversionPair struct {
	inType  *types.Type
	outType *types.Type
}

// All of the types in conversions map are of type "DeclarationOf" with
// the underlying type being "Func".
type conversionFuncMap map[conversionPair]*types.Type

// Returns all manually-defined conversion functions in the package.
func getManualConversionFunctions(context *generator.Context, pkg *types.Package, manualMap conversionFuncMap) {
	glog.V(5).Infof("Scanning for conversion functions in %v", pkg.Name)

	scopeName := types.Ref(conversionPackagePath, "Scope").Name
	errorName := types.Ref("", "error").Name
	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")

	for _, f := range pkg.Functions {
		if f.Underlying == nil || f.Underlying.Kind != types.Func {
			glog.Errorf("Malformed function: %#v", f)
			continue
		}
		if f.Underlying.Signature == nil {
			glog.Errorf("Function without signature: %#v", f)
			continue
		}
		glog.V(8).Infof("Considering function %s", f.Name)
		signature := f.Underlying.Signature
		// Check whether the function is conversion function.
		// Note that all of them have signature:
		// func Convert_inType_To_outType(inType, outType, conversion.Scope) error
		if signature.Receiver != nil {
			glog.V(8).Infof("%s has a receiver", f.Name)
			continue
		}
		if len(signature.Parameters) != 3 || signature.Parameters[2].Name != scopeName {
			glog.V(8).Infof("%s has wrong parameters", f.Name)
			continue
		}
		if len(signature.Results) != 1 || signature.Results[0].Name != errorName {
			glog.V(8).Infof("%s has wrong results", f.Name)
			continue
		}
		inType := signature.Parameters[0]
		outType := signature.Parameters[1]
		if inType.Kind != types.Pointer || outType.Kind != types.Pointer {
			glog.V(8).Infof("%s has wrong parameter types", f.Name)
			continue
		}
		// Now check if the name satisfies the convention.
		// TODO: This should call the Namer directly.
		args := argsFromType(inType.Elem, outType.Elem)
		sw.Do("Convert_$.inType|public$_To_$.outType|public$", args)
		if f.Name.Name == buffer.String() {
			glog.V(4).Infof("Found conversion function %s", f.Name)
			key := conversionPair{inType.Elem, outType.Elem}
			// We might scan the same package twice, and that's OK.
			if v, ok := manualMap[key]; ok && v != nil && v.Name.Package != pkg.Path {
				panic(fmt.Sprintf("duplicate static conversion defined: %s -> %s", key.inType, key.outType))
			}
			manualMap[key] = f
		} else {
			glog.V(8).Infof("%s has wrong name", f.Name)
		}
		buffer.Reset()
	}
}

func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		glog.Fatalf("Failed loading boilerplate: %v", err)
	}

	packages := generator.Packages{}
	header := append([]byte(fmt.Sprintf("// +build !%s\n\n", arguments.GeneratedBuildTag)), boilerplate...)
	header = append(header, []byte("\n// This file was autogenerated by conversion-gen. Do not edit it manually!\n\n")...)

	// Accumulate pre-existing conversion functions.
	// TODO: This is too ad-hoc.  We need a better way.
	manualConversions := conversionFuncMap{}

	// Record types that are memory equivalent. A type is memory equivalent
	// if it has the same memory layout and no nested manual conversion is
	// defined.
	// TODO: in the future, relax the nested manual conversion requirement
	//   if we can show that a large enough types are memory identical but
	//   have non-trivial conversion
	memoryEquivalentTypes := equalMemoryTypes{}

	// We are generating conversions only for packages that are explicitly
	// passed as InputDir.
	processed := map[string]bool{}
	for _, i := range context.Inputs {
		// skip duplicates
		if processed[i] {
			continue
		}
		processed[i] = true

		glog.V(5).Infof("considering pkg %q", i)
		pkg := context.Universe[i]
		// typesPkg is where the versioned types are defined. Sometimes it is
		// different from pkg. For example, kubernetes core/v1 types are defined
		// in vendor/k8s.io/api/core/v1, while pkg is at pkg/api/v1.
		typesPkg := pkg
		if pkg == nil {
			// If the input had no Go files, for example.
			continue
		}

		// Add conversion and defaulting functions.
		getManualConversionFunctions(context, pkg, manualConversions)

		// Only generate conversions for packages which explicitly request it
		// by specifying one or more "+k8s:conversion-gen=<peer-pkg>"
		// in their doc.go file.
		peerPkgs := extractTag(pkg.Comments)
		if peerPkgs != nil {
			glog.V(5).Infof("  tags: %q", peerPkgs)
		} else {
			glog.V(5).Infof("  no tag")
			continue
		}
		skipUnsafe := false
		if customArgs, ok := arguments.CustomArgs.(*CustomArgs); ok {
			if len(customArgs.ExtraPeerDirs) > 0 {
				peerPkgs = append(peerPkgs, customArgs.ExtraPeerDirs...)
			}
			skipUnsafe = customArgs.SkipUnsafe
		}

		// if the external types are not in the same package where the conversion functions to be generated
		externalTypesValues := extractExternalTypesTag(pkg.Comments)
		if externalTypesValues != nil {
			if len(externalTypesValues) != 1 {
				glog.Fatalf("  expect only one value for %q tag, got: %q", externalTypesTagName, externalTypesValues)
			}
			externalTypes := externalTypesValues[0]
			glog.V(5).Infof("  external types tags: %q", externalTypes)
			var err error
			typesPkg, err = context.AddDirectory(filepath.Join(pkg.Path, externalTypes))
			if err != nil {
				glog.Fatalf("cannot import package %s", externalTypes)
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

		// Make sure our peer-packages are added and fully parsed.
		for _, pp := range peerPkgs {
			context.AddDir(pp)
			getManualConversionFunctions(context, context.Universe[pp], manualConversions)
		}

		unsafeEquality := TypesEqual(memoryEquivalentTypes)
		if skipUnsafe {
			unsafeEquality = noEquality{}
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
		packages = append(packages,
			&generator.DefaultPackage{
				PackageName: filepath.Base(pkg.Path),
				PackagePath: path,
				HeaderText:  header,
				GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
					return []generator.Generator{
						NewGenConversion(arguments.OutputFileBaseName, typesPkg.Path, pkg.Path, manualConversions, peerPkgs, unsafeEquality),
					}
				},
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == typesPkg.Path
				},
			})
	}

	// If there is a manual conversion defined between two types, exclude it
	// from being a candidate for unsafe conversion
	for k, v := range manualConversions {
		if isCopyOnly(v.CommentLines) {
			glog.V(5).Infof("Conversion function %s will not block memory copy because it is copy-only", v.Name)
			continue
		}
		// this type should be excluded from all equivalence, because the converter must be called.
		memoryEquivalentTypes.Skip(k.inType, k.outType)
	}

	return packages
}

type equalMemoryTypes map[conversionPair]bool

func (e equalMemoryTypes) Skip(a, b *types.Type) {
	e[conversionPair{a, b}] = false
	e[conversionPair{b, a}] = false
}

func (e equalMemoryTypes) Equal(a, b *types.Type) bool {
	// alreadyVisitedTypes holds all the types that have already been checked in the structural type recursion.
	alreadyVisitedTypes := make(map[*types.Type]bool)
	return e.cachingEqual(a, b, alreadyVisitedTypes)
}

func (e equalMemoryTypes) cachingEqual(a, b *types.Type, alreadyVisitedTypes map[*types.Type]bool) bool {
	if a == b {
		return true
	}
	if equal, ok := e[conversionPair{a, b}]; ok {
		return equal
	}
	if equal, ok := e[conversionPair{b, a}]; ok {
		return equal
	}
	result := e.equal(a, b, alreadyVisitedTypes)
	e[conversionPair{a, b}] = result
	e[conversionPair{b, a}] = result
	return result
}

func (e equalMemoryTypes) equal(a, b *types.Type, alreadyVisitedTypes map[*types.Type]bool) bool {
	in, out := unwrapAlias(a), unwrapAlias(b)
	switch {
	case in == out:
		return true
	case in.Kind == out.Kind:
		// if the type exists already, return early to avoid recursion
		if alreadyVisitedTypes[in] {
			return true
		}
		alreadyVisitedTypes[in] = true

		switch in.Kind {
		case types.Struct:
			if len(in.Members) != len(out.Members) {
				return false
			}
			for i, inMember := range in.Members {
				outMember := out.Members[i]
				if !e.cachingEqual(inMember.Type, outMember.Type, alreadyVisitedTypes) {
					return false
				}
			}
			return true
		case types.Pointer:
			return e.cachingEqual(in.Elem, out.Elem, alreadyVisitedTypes)
		case types.Map:
			return e.cachingEqual(in.Key, out.Key, alreadyVisitedTypes) && e.cachingEqual(in.Elem, out.Elem, alreadyVisitedTypes)
		case types.Slice:
			return e.cachingEqual(in.Elem, out.Elem, alreadyVisitedTypes)
		case types.Interface:
			// TODO: determine whether the interfaces are actually equivalent - for now, they must have the
			// same type.
			return false
		case types.Builtin:
			return in.Name.Name == out.Name.Name
		}
	}
	return false
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

// unwrapAlias recurses down aliased types to find the bedrock type.
func unwrapAlias(in *types.Type) *types.Type {
	for in.Kind == types.Alias {
		in = in.Underlying
	}
	return in
}

const (
	runtimePackagePath    = "k8s.io/apimachinery/pkg/runtime"
	conversionPackagePath = "k8s.io/apimachinery/pkg/conversion"
)

type noEquality struct{}

func (noEquality) Equal(_, _ *types.Type) bool { return false }

type TypesEqual interface {
	Equal(a, b *types.Type) bool
}

// genConversion produces a file with a autogenerated conversions.
type genConversion struct {
	generator.DefaultGen
	// the package that contains the types that conversion func are going to be
	// generated for
	typesPackage string
	// the package that the conversion funcs are going to be output to
	outputPackage string
	// packages that contain the peer of types in typesPacakge
	peerPackages      []string
	manualConversions conversionFuncMap
	imports           namer.ImportTracker
	types             []*types.Type
	skippedFields     map[*types.Type][]string
	useUnsafe         TypesEqual
}

func NewGenConversion(sanitizedName, typesPackage, outputPackage string, manualConversions conversionFuncMap, peerPkgs []string, useUnsafe TypesEqual) generator.Generator {
	return &genConversion{
		DefaultGen: generator.DefaultGen{
			OptionalName: sanitizedName,
		},
		typesPackage:      typesPackage,
		outputPackage:     outputPackage,
		peerPackages:      peerPkgs,
		manualConversions: manualConversions,
		imports:           generator.NewImportTracker(),
		types:             []*types.Type{},
		skippedFields:     map[*types.Type][]string{},
		useUnsafe:         useUnsafe,
	}
}

func (g *genConversion) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
		"publicIT": &namerPlusImportTracking{
			delegate: conversionNamer(),
			tracker:  g.imports,
		},
	}
}

type namerPlusImportTracking struct {
	delegate namer.Namer
	tracker  namer.ImportTracker
}

func (n *namerPlusImportTracking) Name(t *types.Type) string {
	n.tracker.AddType(t)
	return n.delegate.Name(t)
}

func (g *genConversion) convertibleOnlyWithinPackage(inType, outType *types.Type) bool {
	var t *types.Type
	var other *types.Type
	if inType.Name.Package == g.typesPackage {
		t, other = inType, outType
	} else {
		t, other = outType, inType
	}

	if t.Name.Package != g.typesPackage {
		return false
	}
	// If the type has opted out, skip it.
	tagvals := extractTag(t.CommentLines)
	if tagvals != nil {
		if tagvals[0] != "false" {
			glog.Fatalf("Type %v: unsupported %s value: %q", t, tagName, tagvals[0])
		}
		glog.V(5).Infof("type %v requests no conversion generation, skipping", t)
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
	peerType := getPeerTypeFor(c, t, g.peerPackages)
	if peerType == nil {
		return false
	}
	if !g.convertibleOnlyWithinPackage(t, peerType) {
		return false
	}

	g.types = append(g.types, t)
	return true
}

func (g *genConversion) isOtherPackage(pkg string) bool {
	if pkg == g.outputPackage {
		return false
	}
	if strings.HasSuffix(pkg, `"`+g.outputPackage+`"`) {
		return false
	}
	return true
}

func (g *genConversion) Imports(c *generator.Context) (imports []string) {
	var importLines []string
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func argsFromType(inType, outType *types.Type) generator.Args {
	return generator.Args{
		"inType":  inType,
		"outType": outType,
	}
}

func defaultingArgsFromType(inType *types.Type) generator.Args {
	return generator.Args{
		"inType": inType,
	}
}

const nameTmpl = "Convert_$.inType|publicIT$_To_$.outType|publicIT$"

func (g *genConversion) preexists(inType, outType *types.Type) (*types.Type, bool) {
	function, ok := g.manualConversions[conversionPair{inType, outType}]
	return function, ok
}

func (g *genConversion) Init(c *generator.Context, w io.Writer) error {
	if glog.V(5) {
		if m, ok := g.useUnsafe.(equalMemoryTypes); ok {
			var result []string
			glog.Infof("All objects without identical memory layout:")
			for k, v := range m {
				if v {
					continue
				}
				result = append(result, fmt.Sprintf("  %s -> %s = %t", k.inType, k.outType, v))
			}
			sort.Strings(result)
			for _, s := range result {
				glog.Infof(s)
			}
		}
	}
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	sw.Do("func init() {\n", nil)
	sw.Do("localSchemeBuilder.Register(RegisterConversions)\n", nil)
	sw.Do("}\n", nil)

	scheme := c.Universe.Type(types.Name{Package: runtimePackagePath, Name: "Scheme"})
	schemePtr := &types.Type{
		Kind: types.Pointer,
		Elem: scheme,
	}
	sw.Do("// RegisterConversions adds conversion functions to the given scheme.\n", nil)
	sw.Do("// Public to allow building arbitrary schemes.\n", nil)
	sw.Do("func RegisterConversions(scheme $.|raw$) error {\n", schemePtr)
	sw.Do("return scheme.AddGeneratedConversionFuncs(\n", nil)
	for _, t := range g.types {
		peerType := getPeerTypeFor(c, t, g.peerPackages)
		sw.Do(nameTmpl+",\n", argsFromType(t, peerType))
		sw.Do(nameTmpl+",\n", argsFromType(peerType, t))
	}
	sw.Do(")\n", nil)
	sw.Do("}\n\n", nil)
	return sw.Error()
}

func (g *genConversion) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	glog.V(5).Infof("generating for type %v", t)
	peerType := getPeerTypeFor(c, t, g.peerPackages)
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	g.generateConversion(t, peerType, sw)
	g.generateConversion(peerType, t, sw)
	return sw.Error()
}

func (g *genConversion) generateConversion(inType, outType *types.Type, sw *generator.SnippetWriter) {
	args := argsFromType(inType, outType).
		With("Scope", types.Ref(conversionPackagePath, "Scope"))

	sw.Do("func auto"+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
	g.generateFor(inType, outType, sw)
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	if _, found := g.preexists(inType, outType); found {
		// There is a public manual Conversion method: use it.
	} else if skipped := g.skippedFields[inType]; len(skipped) != 0 {
		// The inType had some fields we could not generate.
		glog.Errorf("Warning: could not find nor generate a final Conversion function for %v -> %v", inType, outType)
		glog.Errorf("  the following fields need manual conversion:")
		for _, f := range skipped {
			glog.Errorf("      - %v", f)
		}
	} else {
		// Emit a public conversion function.
		sw.Do("// "+nameTmpl+" is an autogenerated conversion function.\n", args)
		sw.Do("func "+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
		sw.Do("return auto"+nameTmpl+"(in, out, s)\n", args)
		sw.Do("}\n\n", nil)
	}
}

// we use the system of shadowing 'in' and 'out' so that the same code is valid
// at any nesting level. This makes the autogenerator easy to understand, and
// the compiler shouldn't care.
func (g *genConversion) generateFor(inType, outType *types.Type, sw *generator.SnippetWriter) {
	glog.V(5).Infof("generating %v -> %v", inType, outType)
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
	if isDirectlyAssignable(inType.Key, outType.Key) {
		sw.Do("for key, val := range *in {\n", nil)
		if isDirectlyAssignable(inType.Elem, outType.Elem) {
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
				sw.Do("if err := "+nameTmpl+"(&val, newVal, s); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
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
		if isDirectlyAssignable(inType.Elem, outType.Elem) {
			if inType.Elem == outType.Elem {
				sw.Do("(*out)[i] = (*in)[i]\n", nil)
			} else {
				sw.Do("(*out)[i] = $.|raw$((*in)[i])\n", outType.Elem)
			}
		} else {
			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				sw.Do("if err := $.|raw$(&(*in)[i], &(*out)[i], s); err != nil {\n", function)
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				sw.Do("if err := "+nameTmpl+"(&(*in)[i], &(*out)[i], s); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
			} else {
				// TODO: This triggers on metav1.ObjectMeta <-> metav1.ObjectMeta and
				// similar because neither package is the target package, and
				// we really don't know which package will have the conversion
				// function defined.  This fires on basically every object
				// conversion outside of pkg/api/v1.
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
	for _, inMember := range inType.Members {
		if tagvals := extractTag(inMember.CommentLines); tagvals != nil && tagvals[0] == "false" {
			// This field is excluded from conversion.
			sw.Do("// INFO: in."+inMember.Name+" opted out of conversion generation\n", nil)
			continue
		}
		outMember, found := findMember(outType, inMember.Name)
		if !found {
			// This field doesn't exist in the peer.
			sw.Do("// WARNING: in."+inMember.Name+" requires manual conversion: does not exist in peer-type\n", nil)
			g.skippedFields[inType] = append(g.skippedFields[inType], inMember.Name)
			continue
		}

		inMemberType, outMemberType := inMember.Type, outMember.Type
		// create a copy of both underlying types but give them the top level alias name (since aliases
		// are assignable)
		if underlying := unwrapAlias(inMemberType); underlying != inMemberType {
			copied := *underlying
			copied.Name = inMemberType.Name
			inMemberType = &copied
		}
		if underlying := unwrapAlias(outMemberType); underlying != outMemberType {
			copied := *underlying
			copied.Name = outMemberType.Name
			outMemberType = &copied
		}

		args := argsFromType(inMemberType, outMemberType).With("name", inMember.Name)

		// try a direct memory copy for any type that has exactly equivalent values
		if g.useUnsafe.Equal(inMemberType, outMemberType) {
			args = args.
				With("Pointer", types.Ref("unsafe", "Pointer")).
				With("SliceHeader", types.Ref("reflect", "SliceHeader"))
			switch inMemberType.Kind {
			case types.Pointer:
				sw.Do("out.$.name$ = ($.outType|raw$)($.Pointer|raw$(in.$.name$))\n", args)
				continue
			case types.Map:
				sw.Do("out.$.name$ = *(*$.outType|raw$)($.Pointer|raw$(&in.$.name$))\n", args)
				continue
			case types.Slice:
				sw.Do("out.$.name$ = *(*$.outType|raw$)($.Pointer|raw$(&in.$.name$))\n", args)
				continue
			}
		}

		// check based on the top level name, not the underlying names
		if function, ok := g.preexists(inMember.Type, outMember.Type); ok {
			if isDrop(function.CommentLines) {
				continue
			}
			// copy-only functions that are directly assignable can be inlined instead of invoked.
			// As an example, conversion functions exist that allow types with private fields to be
			// correctly copied between types. These functions are equivalent to a memory assignment,
			// and are necessary for the reflection path, but should not block memory conversion.
			// Convert_unversioned_Time_to_unversioned_Time is an example of this logic.
			if !isCopyOnly(function.CommentLines) || !g.isFastConversion(inMemberType, outMemberType) {
				args["function"] = function
				sw.Do("if err := $.function|raw$(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
				continue
			}
			glog.V(5).Infof("Skipped function %s because it is copy-only and we can use direct assignment", function.Name)
		}

		// If we can't auto-convert, punt before we emit any code.
		if inMemberType.Kind != outMemberType.Kind {
			sw.Do("// WARNING: in."+inMember.Name+" requires manual conversion: inconvertible types ("+
				inMemberType.String()+" vs "+outMemberType.String()+")\n", nil)
			g.skippedFields[inType] = append(g.skippedFields[inType], inMember.Name)
			continue
		}

		switch inMemberType.Kind {
		case types.Builtin:
			if inMemberType == outMemberType {
				sw.Do("out.$.name$ = in.$.name$\n", args)
			} else {
				sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
			}
		case types.Map, types.Slice, types.Pointer:
			if g.isDirectlyAssignable(inMemberType, outMemberType) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}

			sw.Do("if in.$.name$ != nil {\n", args)
			sw.Do("in, out := &in.$.name$, &out.$.name$\n", args)
			g.generateFor(inMemberType, outMemberType, sw)
			sw.Do("} else {\n", nil)
			sw.Do("out.$.name$ = nil\n", args)
			sw.Do("}\n", nil)
		case types.Struct:
			if g.isDirectlyAssignable(inMemberType, outMemberType) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}
			if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
				sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		case types.Alias:
			if isDirectlyAssignable(inMemberType, outMemberType) {
				if inMemberType == outMemberType {
					sw.Do("out.$.name$ = in.$.name$\n", args)
				} else {
					sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
				}
			} else {
				if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
					sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
				} else {
					sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
					sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
				}
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
			}
		default:
			if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
				sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		}
	}
}

func (g *genConversion) isFastConversion(inType, outType *types.Type) bool {
	switch inType.Kind {
	case types.Builtin:
		return true
	case types.Map, types.Slice, types.Pointer, types.Struct, types.Alias:
		return g.isDirectlyAssignable(inType, outType)
	default:
		return false
	}
}

func (g *genConversion) isDirectlyAssignable(inType, outType *types.Type) bool {
	return unwrapAlias(inType) == unwrapAlias(outType)
}

func (g *genConversion) doPointer(inType, outType *types.Type, sw *generator.SnippetWriter) {
	sw.Do("*out = new($.Elem|raw$)\n", outType)
	if isDirectlyAssignable(inType.Elem, outType.Elem) {
		if inType.Elem == outType.Elem {
			sw.Do("**out = **in\n", nil)
		} else {
			sw.Do("**out = $.|raw$(**in)\n", outType.Elem)
		}
	} else {
		if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
			sw.Do("if err := $.|raw$(*in, *out, s); err != nil {\n", function)
		} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
			sw.Do("if err := "+nameTmpl+"(*in, *out, s); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
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

func isDirectlyAssignable(inType, outType *types.Type) bool {
	// TODO: This should maybe check for actual assignability between the two
	// types, rather than superficial traits that happen to indicate it is
	// assignable in the ways we currently use this code.
	return inType.IsAssignable() && (inType.IsPrimitive() || isSamePackage(inType, outType))
}

func isSamePackage(inType, outType *types.Type) bool {
	return inType.Name.Package == outType.Name.Package
}
