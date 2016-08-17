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
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

// CustomArgs is used tby the go2idl framework to pass args specific to this
// generator.
type CustomArgs struct {
	ExtraPeerDirs []string // Always consider these as last-ditch possibilities for conversions.
}

// This is the comment tag that carries parameters for conversion generation.
const tagName = "k8s:conversion-gen"

func extractTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[tagName]
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
			key := conversionPair{inType.Elem, outType.Elem}
			// We might scan the same package twice, and that's OK.
			if v, ok := manualMap[key]; ok && v != nil && v.Name.Package != pkg.Path {
				panic(fmt.Sprintf("duplicate static conversion defined: %#v", key))
			}
			manualMap[key] = f
		}
		buffer.Reset()
	}
}

// All of the types in conversions map are of type "DeclarationOf" with
// the underlying type being "Func".
type defaulterFuncMap map[*types.Type]*types.Type

// Returns all manually-defined defaulting functions in the package.
func getManualDefaultingFunctions(context *generator.Context, pkg *types.Package, manualMap defaulterFuncMap) {
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
			// We might scan the same package twice, and that's OK.
			if v, ok := manualMap[key]; ok && v != nil && v.Name.Package != pkg.Path {
				panic(fmt.Sprintf("duplicate static defaulter defined: %#v", key))
			}
			manualMap[key] = f
		}
		buffer.Reset()
	}
}

func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		glog.Fatalf("Failed loading boilerplate: %v", err)
	}

	inputs := sets.NewString(context.Inputs...)
	packages := generator.Packages{}
	header := append([]byte(fmt.Sprintf("// +build !%s\n\n", arguments.GeneratedBuildTag)), boilerplate...)
	header = append(header, []byte(
		`
// This file was autogenerated by conversion-gen. Do not edit it manually!

`)...)

	// Accumulate pre-existing conversion and default functions.
	// TODO: This is too ad-hoc.  We need a better way.
	manualConversions := conversionFuncMap{}
	manualDefaults := defaulterFuncMap{}

	// We are generating conversions only for packages that are explicitly
	// passed as InputDir.
	for i := range inputs {
		glog.V(5).Infof("considering pkg %q", i)
		pkg := context.Universe[i]
		if pkg == nil {
			// If the input had no Go files, for example.
			continue
		}

		// Add conversion and defaulting functions.
		getManualConversionFunctions(context, pkg, manualConversions)
		getManualDefaultingFunctions(context, pkg, manualDefaults)

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
		if customArgs, ok := arguments.CustomArgs.(*CustomArgs); ok {
			if len(customArgs.ExtraPeerDirs) > 0 {
				peerPkgs = append(peerPkgs, customArgs.ExtraPeerDirs...)
			}
		}
		// Make sure our peer-packages are added and fully parsed.
		for _, pp := range peerPkgs {
			context.AddDir(pp)
			getManualConversionFunctions(context, context.Universe[pp], manualConversions)
			getManualDefaultingFunctions(context, context.Universe[pp], manualDefaults)
		}

		pkgNeedsGeneration := false
		for _, t := range pkg.Types {
			// Check whether this type can be auto-converted to the peer
			// package type.
			peerType := getPeerTypeFor(context, t, peerPkgs)
			if peerType == nil {
				// We did not find a corresponding type.
				continue
			}
			if namer.IsPrivateGoName(peerType.Name.Name) {
				// We won't be able to convert to a private type.
				glog.V(5).Infof("  found a peer type %v, but it is a private name", t)
				continue
			}

			// If we can generate conversion in any direction, we should
			// generate this package.
			if isConvertible(t, peerType, manualConversions) || isConvertible(peerType, t, manualConversions) {
				pkgNeedsGeneration = true
				break
			}
		}
		if !pkgNeedsGeneration {
			glog.V(5).Infof("  no viable conversions, not generating for this package")
			continue
		}

		packages = append(packages,
			&generator.DefaultPackage{
				PackageName: filepath.Base(pkg.Path),
				PackagePath: pkg.Path,
				HeaderText:  header,
				GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
					generators = []generator.Generator{}
					generators = append(
						generators, NewGenConversion(arguments.OutputFileBaseName, pkg.Path, manualConversions, manualDefaults, peerPkgs))
					return generators
				},
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == pkg.Path
				},
			})
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

func isConvertible(in, out *types.Type, manualConversions conversionFuncMap) bool {
	// If there is pre-existing conversion function, return true immediately.
	if _, ok := manualConversions[conversionPair{in, out}]; ok {
		return true
	}
	return isDirectlyConvertible(in, out, manualConversions)
}

func isDirectlyConvertible(in, out *types.Type, manualConversions conversionFuncMap) bool {
	// If one of the types is Alias, resolve it.
	if in.Kind == types.Alias {
		return isConvertible(in.Underlying, out, manualConversions)
	}
	if out.Kind == types.Alias {
		return isConvertible(in, out.Underlying, manualConversions)
	}

	if in == out {
		return true
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

	switch in.Kind {
	case types.Builtin:
		// TODO: Support more conversion types.
		return types.IsInteger(in) && types.IsInteger(out)
	case types.Struct:
		convertible := true
		for _, inMember := range in.Members {
			// Check if this member is excluded from conversion
			if tagvals := extractTag(inMember.CommentLines); tagvals != nil && tagvals[0] == "false" {
				continue
			}
			// Check if there is an out member with that name.
			outMember, found := findMember(out, inMember.Name)
			if !found {
				return false
			}
			convertible = convertible && isConvertible(inMember.Type, outMember.Type, manualConversions)
		}
		return convertible
	case types.Map:
		return isConvertible(in.Key, out.Key, manualConversions) && isConvertible(in.Elem, out.Elem, manualConversions)
	case types.Slice:
		return isConvertible(in.Elem, out.Elem, manualConversions)
	case types.Pointer:
		return isConvertible(in.Elem, out.Elem, manualConversions)
	}
	glog.Fatalf("All other types should be filtered before")
	return false
}

// unwrapAlias recurses down aliased types to find the bedrock type.
func unwrapAlias(in *types.Type) *types.Type {
	if in.Kind == types.Alias {
		return unwrapAlias(in.Underlying)
	}
	return in
}

func areTypesAliased(in, out *types.Type) bool {
	// If one of the types is Alias, resolve it.
	if in.Kind == types.Alias {
		return areTypesAliased(in.Underlying, out)
	}
	if out.Kind == types.Alias {
		return areTypesAliased(in, out.Underlying)
	}

	return in == out
}

const (
	runtimePackagePath    = "k8s.io/kubernetes/pkg/runtime"
	conversionPackagePath = "k8s.io/kubernetes/pkg/conversion"
)

// genConversion produces a file with a autogenerated conversions.
type genConversion struct {
	generator.DefaultGen
	targetPackage     string
	peerPackages      []string
	manualConversions conversionFuncMap
	manualDefaulters  defaulterFuncMap
	imports           namer.ImportTracker
	typesForInit      []conversionPair
}

func NewGenConversion(sanitizedName, targetPackage string, manualConversions conversionFuncMap, manualDefaulters defaulterFuncMap, peerPkgs []string) generator.Generator {
	return &genConversion{
		DefaultGen: generator.DefaultGen{
			OptionalName: sanitizedName,
		},
		targetPackage:     targetPackage,
		peerPackages:      peerPkgs,
		manualConversions: manualConversions,
		manualDefaulters:  manualDefaulters,
		imports:           generator.NewImportTracker(),
		typesForInit:      make([]conversionPair, 0),
	}
}

func (g *genConversion) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.targetPackage, g.imports),
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
	if inType.Name.Package == g.targetPackage {
		t, other = inType, outType
	} else {
		t, other = outType, inType
	}

	if t.Name.Package != g.targetPackage {
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
	// We explicitly return true if any conversion is possible - this needs
	// to be checked again while generating code for that type.
	convertible := false
	if isConvertible(t, peerType, g.manualConversions) {
		g.typesForInit = append(g.typesForInit, conversionPair{t, peerType})
		convertible = true
	}
	if isConvertible(peerType, t, g.manualConversions) {
		g.typesForInit = append(g.typesForInit, conversionPair{peerType, t})
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
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	sw.Do("func init() {\n", nil)
	sw.Do("SchemeBuilder.Register(RegisterConversions)\n", nil)
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
	for _, conv := range g.typesForInit {
		sw.Do(nameTmpl+",\n", argsFromType(conv.inType, conv.outType))
	}
	sw.Do(")\n", nil)
	sw.Do("}\n\n", nil)
	return sw.Error()
}

func (g *genConversion) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	glog.V(5).Infof("generating for type %v", t)
	sw := generator.NewSnippetWriter(w, c, "$", "$")
	peerType := getPeerTypeFor(c, t, g.peerPackages)
	didForward, didBackward := false, false
	if isDirectlyConvertible(t, peerType, g.manualConversions) {
		didForward = true
		g.generateConversion(t, peerType, sw)
	}
	if isDirectlyConvertible(peerType, t, g.manualConversions) {
		didBackward = true
		g.generateConversion(peerType, t, sw)
	}
	if didForward != didBackward {
		glog.Fatalf("Could only generate one direction of conversion for %v <-> %v", t, peerType)
	}
	if !didForward && !didBackward {
		// TODO: This should be fatal but we have at least 8 types that
		// currently fail this.  The right thing to do is to figure out why they
		// can't be generated and mark those fields as
		// +k8s:conversion-gen=false, and ONLY do manual conversions for those
		// fields, with the manual Convert_...() calling autoConvert_...()
		// first.
		glog.Errorf("Warning: could not generate autoConvert functions for %v <-> %v", t, peerType)
	}
	return sw.Error()
}

func (g *genConversion) generateConversion(inType, outType *types.Type, sw *generator.SnippetWriter) {
	args := argsFromType(inType, outType).
		With("Scope", types.Ref(conversionPackagePath, "Scope"))

	sw.Do("func auto"+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
	// if no defaulter of form SetDefaults_XXX is defined, do not inline a check for defaulting.
	if function, ok := g.manualDefaulters[inType]; ok {
		sw.Do("$.|raw$(in)\n", function)
	}
	g.generateFor(inType, outType, sw)
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	// If there is no public manual Conversion method, generate it.
	if _, ok := g.preexists(inType, outType); !ok {
		sw.Do("func "+nameTmpl+"(in *$.inType|raw$, out *$.outType|raw$, s $.Scope|raw$) error {\n", args)
		sw.Do("return auto"+nameTmpl+"(in, out, s)\n", args)
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
				// TODO: This triggers on v1.ObjectMeta <-> api.ObjectMeta and
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
	for _, m := range inType.Members {
		// Check if this member is excluded from conversion
		if tagvals := extractTag(m.CommentLines); tagvals != nil && tagvals[0] == "false" {
			continue
		}
		outMember, isOutMember := findMember(outType, m.Name)
		if !isOutMember {
			// Since this object wasn't filtered out, this means that
			// this field has "+k8s:conversion-gen=false" comment to ignore it.
			continue
		}
		t, outT := m.Type, outMember.Type
		// create a copy of both underlying types but give them the top level alias name (since aliases
		// are assignable)
		if underlying := unwrapAlias(t); underlying != t {
			copied := *underlying
			copied.Name = t.Name
			t = &copied
		}
		if underlying := unwrapAlias(outT); underlying != outT {
			copied := *underlying
			copied.Name = outT.Name
			outT = &copied
		}
		args := map[string]interface{}{
			"inType":  t,
			"outType": outT,
			"name":    m.Name,
		}
		// check based on the top level name, not the underlying names
		if function, ok := g.preexists(m.Type, outMember.Type); ok {
			args["function"] = function
			sw.Do("if err := $.function|raw$(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
			continue
		}
		switch t.Kind {
		case types.Builtin:
			if t == outT {
				sw.Do("out.$.name$ = in.$.name$\n", args)
			} else {
				sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
			}
		case types.Map, types.Slice, types.Pointer:
			if g.isDirectlyAssignable(t, outT) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}

			sw.Do("if in.$.name$ != nil {\n", args)
			sw.Do("in, out := &in.$.name$, &out.$.name$\n", args)
			g.generateFor(t, outT, sw)
			sw.Do("} else {\n", nil)
			sw.Do("out.$.name$ = nil\n", args)
			sw.Do("}\n", nil)
		case types.Struct:
			if g.isDirectlyAssignable(t, outT) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}
			if g.convertibleOnlyWithinPackage(t, outT) {
				sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
			} else {
				sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
				sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
			}
			sw.Do("return err\n", nil)
			sw.Do("}\n", nil)
		case types.Alias:
			if isDirectlyAssignable(t, outT) {
				if t == outT {
					sw.Do("out.$.name$ = in.$.name$\n", args)
				} else {
					sw.Do("out.$.name$ = $.outType|raw$(in.$.name$)\n", args)
				}
			} else {
				if g.convertibleOnlyWithinPackage(t, outT) {
					sw.Do("if err := "+nameTmpl+"(&in.$.name$, &out.$.name$, s); err != nil {\n", args)
				} else {
					sw.Do("// TODO: Inefficient conversion - can we improve it?\n", nil)
					sw.Do("if err := s.Convert(&in.$.name$, &out.$.name$, 0); err != nil {\n", args)
				}
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
			}
		default:
			if g.convertibleOnlyWithinPackage(t, outT) {
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

func (g *genConversion) isDirectlyAssignable(inType, outType *types.Type) bool {
	return inType == outType || areTypesAliased(inType, outType)
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
