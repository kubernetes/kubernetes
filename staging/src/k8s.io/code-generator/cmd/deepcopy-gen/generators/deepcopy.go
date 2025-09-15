/*
Copyright 2015 The Kubernetes Authors.

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
	"path"
	"sort"
	"strings"

	"k8s.io/code-generator/cmd/deepcopy-gen/args"
	genutil "k8s.io/code-generator/pkg/util"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// This is the comment tag that carries parameters for deep-copy generation.
const (
	tagEnabledName              = "k8s:deepcopy-gen"
	interfacesTagName           = tagEnabledName + ":interfaces"
	interfacesNonPointerTagName = tagEnabledName + ":nonpointer-interfaces" // attach the DeepCopy<Interface> methods to the
)

// Known values for the comment tag.
const tagValuePackage = "package"

// enabledTagValue holds parameters from a tagName tag.
type enabledTagValue struct {
	value    string
	register bool
}

func extractEnabledTypeTag(t *types.Type) *enabledTagValue {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	return extractEnabledTag(comments)
}

func extractEnabledTag(comments []string) *enabledTagValue {
	tags, err := genutil.ExtractCommentTagsWithoutArguments("+", []string{tagEnabledName}, comments)
	if err != nil {
		klog.Fatalf("Error extracting %s tags: %v", tagEnabledName, err)
	}
	if tags[tagEnabledName] == nil {
		// No match for the tag.
		return nil
	}
	// If there are multiple values, abort.
	if len(tags[tagEnabledName]) > 1 {
		klog.Fatalf("Found %d %s tags: %q", len(tags[tagEnabledName]), tagEnabledName, tags[tagEnabledName])
	}

	// If we got here we are returning something.
	tag := &enabledTagValue{}

	// Get the primary value.
	parts := strings.Split(tags[tagEnabledName][0], ",")
	if len(parts) >= 1 {
		tag.value = parts[0]
	}

	// Parse extra arguments.
	parts = parts[1:]
	for i := range parts {
		kv := strings.SplitN(parts[i], "=", 2)
		k := kv[0]
		v := ""
		if len(kv) == 2 {
			v = kv[1]
		}
		switch k {
		case "register":
			if v != "false" {
				tag.register = true
			}
		default:
			klog.Fatalf("Unsupported %s param: %q", tagEnabledName, parts[i])
		}
	}
	return tag
}

// TODO: This is created only to reduce number of changes in a single PR.
// Remove it and use PublicNamer instead.
func deepCopyNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Join: func(pre string, in []string, post string) string {
			return strings.Join(in, "_")
		},
		PrependPackageNames: 1,
	}
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public": deepCopyNamer(),
		"raw":    namer.NewRawNamer("", nil),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, gengo.StdBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	boundingDirs := []string{}
	if args.BoundingDirs == nil {
		args.BoundingDirs = context.Inputs
	}
	for i := range args.BoundingDirs {
		// Strip any trailing slashes - they are not exactly "correct" but
		// this is friendlier.
		boundingDirs = append(boundingDirs, strings.TrimRight(args.BoundingDirs[i], "/"))
	}

	targets := []generator.Target{}

	for _, i := range context.Inputs {
		klog.V(3).Infof("Considering pkg %q", i)

		pkg := context.Universe[i]

		ptag := extractEnabledTag(pkg.Comments)
		ptagValue := ""
		ptagRegister := false
		if ptag != nil {
			ptagValue = ptag.value
			if ptagValue != tagValuePackage {
				klog.Fatalf("Package %v: unsupported %s value: %q", i, tagEnabledName, ptagValue)
			}
			ptagRegister = ptag.register
			klog.V(3).Infof("  tag.value: %q, tag.register: %t", ptagValue, ptagRegister)
		} else {
			klog.V(3).Infof("  no tag")
		}

		// If the pkg-scoped tag says to generate, we can skip scanning types.
		pkgNeedsGeneration := (ptagValue == tagValuePackage)
		if !pkgNeedsGeneration {
			// If the pkg-scoped tag did not exist, scan all types for one that
			// explicitly wants generation. Ensure all types that want generation
			// can be copied.
			var uncopyable []string
			for _, t := range pkg.Types {
				klog.V(3).Infof("  considering type %q", t.Name.String())
				ttag := extractEnabledTypeTag(t)
				if ttag != nil && ttag.value == "true" {
					klog.V(3).Infof("    tag=true")
					if !copyableType(t) {
						uncopyable = append(uncopyable, fmt.Sprintf("%v", t))
					} else {
						pkgNeedsGeneration = true
					}
				}
			}
			if len(uncopyable) > 0 {
				klog.Fatalf("Types requested deepcopy generation but are not copyable: %s",
					strings.Join(uncopyable, ", "))
			}
		}

		if pkgNeedsGeneration {
			klog.V(3).Infof("Package %q needs generation", i)
			targets = append(targets,
				&generator.SimpleTarget{
					PkgName:       strings.Split(path.Base(pkg.Path), ".")[0],
					PkgPath:       pkg.Path,
					PkgDir:        pkg.Dir, // output pkg is the same as the input
					HeaderComment: boilerplate,
					FilterFunc: func(c *generator.Context, t *types.Type) bool {
						return t.Name.Package == pkg.Path
					},
					GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
						return []generator.Generator{
							NewGenDeepCopy(args.OutputFile, pkg.Path, boundingDirs, (ptagValue == tagValuePackage), ptagRegister),
						}
					},
				})
		}
	}
	return targets
}

// genDeepCopy produces a file with autogenerated deep-copy functions.
type genDeepCopy struct {
	generator.GoGenerator
	targetPackage string
	boundingDirs  []string
	allTypes      bool
	registerTypes bool
	imports       namer.ImportTracker
	typesForInit  []*types.Type
}

func NewGenDeepCopy(outputFilename, targetPackage string, boundingDirs []string, allTypes, registerTypes bool) generator.Generator {
	return &genDeepCopy{
		GoGenerator: generator.GoGenerator{
			OutputFilename: outputFilename,
		},
		targetPackage: targetPackage,
		boundingDirs:  boundingDirs,
		allTypes:      allTypes,
		registerTypes: registerTypes,
		imports:       generator.NewImportTrackerForPackage(targetPackage),
		typesForInit:  make([]*types.Type, 0),
	}
}

func (g *genDeepCopy) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.targetPackage, g.imports),
	}
}

func (g *genDeepCopy) Filter(c *generator.Context, t *types.Type) bool {
	// Filter out types not being processed or not copyable within the package.
	enabled := g.allTypes
	if !enabled {
		ttag := extractEnabledTypeTag(t)
		if ttag != nil && ttag.value == "true" {
			enabled = true
		}
	}
	if !enabled {
		return false
	}
	if !copyableType(t) {
		klog.V(3).Infof("Type %v is not copyable", t)
		return false
	}
	klog.V(3).Infof("Type %v is copyable", t)
	g.typesForInit = append(g.typesForInit, t)
	return true
}

// deepCopyMethod returns the signature of a DeepCopy() method, nil or an error
// if the type does not match. This allows more efficient deep copy
// implementations to be defined by the type's author.  The correct signature
// for a type T is:
//
//	func (t T) DeepCopy() T
//
// or:
//
//	func (t *T) DeepCopy() *T
func deepCopyMethod(t *types.Type) (*types.Signature, error) {
	f, found := t.Methods["DeepCopy"]
	if !found {
		return nil, nil
	}
	if len(f.Signature.Parameters) != 0 {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected no parameters", t)
	}
	if len(f.Signature.Results) != 1 {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected exactly one result", t)
	}

	ptrResult := f.Signature.Results[0].Type.Kind == types.Pointer && f.Signature.Results[0].Type.Elem.Name == t.Name
	nonPtrResult := f.Signature.Results[0].Type.Name == t.Name

	if !ptrResult && !nonPtrResult {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected to return %s or *%s", t, t.Name.Name, t.Name.Name)
	}

	ptrRcvr := f.Signature.Receiver != nil && f.Signature.Receiver.Kind == types.Pointer && f.Signature.Receiver.Elem.Name == t.Name
	nonPtrRcvr := f.Signature.Receiver != nil && f.Signature.Receiver.Name == t.Name

	if ptrRcvr && !ptrResult {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected a *%s result for a *%s receiver", t, t.Name.Name, t.Name.Name)
	}
	if nonPtrRcvr && !nonPtrResult {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected a %s result for a %s receiver", t, t.Name.Name, t.Name.Name)
	}

	return f.Signature, nil
}

// deepCopyMethodOrDie returns the signatrue of a DeepCopy method, nil or calls klog.Fatalf
// if the type does not match.
func deepCopyMethodOrDie(t *types.Type) *types.Signature {
	ret, err := deepCopyMethod(t)
	if err != nil {
		klog.Fatal(err)
	}
	return ret
}

// deepCopyIntoMethod returns the signature of a DeepCopyInto() method, nil or an error
// if the type is wrong. DeepCopyInto allows more efficient deep copy
// implementations to be defined by the type's author.  The correct signature
// for a type T is:
//
//	func (t T) DeepCopyInto(t *T)
//
// or:
//
//	func (t *T) DeepCopyInto(t *T)
func deepCopyIntoMethod(t *types.Type) (*types.Signature, error) {
	f, found := t.Methods["DeepCopyInto"]
	if !found {
		return nil, nil
	}
	if len(f.Signature.Parameters) != 1 {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected exactly one parameter", t)
	}
	if len(f.Signature.Results) != 0 {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected no result type", t)
	}

	ptrParam := f.Signature.Parameters[0].Type.Kind == types.Pointer && f.Signature.Parameters[0].Type.Elem.Name == t.Name

	if !ptrParam {
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected parameter of type *%s", t, t.Name.Name)
	}

	ptrRcvr := f.Signature.Receiver != nil && f.Signature.Receiver.Kind == types.Pointer && f.Signature.Receiver.Elem.Name == t.Name
	nonPtrRcvr := f.Signature.Receiver != nil && f.Signature.Receiver.Name == t.Name

	if !ptrRcvr && !nonPtrRcvr {
		// this should never happen
		return nil, fmt.Errorf("type %v: invalid DeepCopy signature, expected a receiver of type %s or *%s", t, t.Name.Name, t.Name.Name)
	}

	return f.Signature, nil
}

// deepCopyIntoMethodOrDie returns the signature of a DeepCopyInto() method, nil or calls klog.Fatalf
// if the type is wrong.
func deepCopyIntoMethodOrDie(t *types.Type) *types.Signature {
	ret, err := deepCopyIntoMethod(t)
	if err != nil {
		klog.Fatal(err)
	}
	return ret
}

func copyableType(t *types.Type) bool {
	// If the type opts out of copy-generation, stop.
	ttag := extractEnabledTypeTag(t)
	if ttag != nil && ttag.value == "false" {
		return false
	}

	// Filter out private types.
	if namer.IsPrivateGoName(t.Name.Name) {
		return false
	}

	if t.Kind == types.Alias {
		// if the underlying built-in is not deepcopy-able, deepcopy is opt-in through definition of custom methods.
		// Note that aliases of builtins, maps, slices can have deepcopy methods.
		if deepCopyMethodOrDie(t) != nil || deepCopyIntoMethodOrDie(t) != nil {
			return true
		} else {
			return t.Underlying.Kind != types.Builtin || copyableType(t.Underlying)
		}
	}

	if t.Kind != types.Struct {
		return false
	}

	return true
}

func underlyingType(t *types.Type) *types.Type {
	for t.Kind == types.Alias {
		t = t.Underlying
	}
	return t
}

func (g *genDeepCopy) isOtherPackage(pkg string) bool {
	if pkg == g.targetPackage {
		return false
	}
	if strings.HasSuffix(pkg, "\""+g.targetPackage+"\"") {
		return false
	}
	return true
}

func (g *genDeepCopy) Imports(c *generator.Context) (imports []string) {
	importLines := []string{}
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func argsFromType(ts ...*types.Type) generator.Args {
	a := generator.Args{
		"type": ts[0],
	}
	for i, t := range ts {
		a[fmt.Sprintf("type%d", i+1)] = t
	}
	return a
}

func (g *genDeepCopy) Init(c *generator.Context, w io.Writer) error {
	return nil
}

func (g *genDeepCopy) needsGeneration(t *types.Type) bool {
	tag := extractEnabledTypeTag(t)
	tv := ""
	if tag != nil {
		tv = tag.value
		if tv != "true" && tv != "false" {
			klog.Fatalf("Type %v: unsupported %s value: %q", t, tagEnabledName, tag.value)
		}
	}
	if g.allTypes && tv == "false" {
		// The whole package is being generated, but this type has opted out.
		klog.V(2).Infof("Not generating for type %v because type opted out", t)
		return false
	}
	if !g.allTypes && tv != "true" {
		// The whole package is NOT being generated, and this type has NOT opted in.
		klog.V(2).Infof("Not generating for type %v because type did not opt in", t)
		return false
	}
	return true
}

func extractInterfacesTag(t *types.Type) []string {
	var result []string
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	tags, err := genutil.ExtractCommentTagsWithoutArguments("+", []string{interfacesTagName}, comments)
	if err != nil {
		klog.Fatalf("Error extracting %s tags: %v", interfacesTagName, err)
	}
	for _, v := range tags[interfacesTagName] {
		if len(v) == 0 {
			continue
		}
		intfs := strings.Split(v, ",")
		for _, intf := range intfs {
			if intf == "" {
				continue
			}
			result = append(result, intf)
		}
	}
	return result
}

func extractNonPointerInterfaces(t *types.Type) (bool, error) {
	comments := append(append([]string{}, t.SecondClosestCommentLines...), t.CommentLines...)
	tags, err := genutil.ExtractCommentTagsWithoutArguments("+", []string{interfacesNonPointerTagName}, comments)
	if err != nil {
		return false, fmt.Errorf("failed to parse comments: %w", err)
	}

	values := tags[interfacesNonPointerTagName]
	if len(values) == 0 {
		return false, nil
	}
	result := values[0] == "true"
	for _, v := range values {
		if v == "true" != result {
			return false, fmt.Errorf("contradicting %v value %q found to previous value %v", interfacesNonPointerTagName, v, result)
		}
	}
	return result, nil
}

func (g *genDeepCopy) deepCopyableInterfacesInner(c *generator.Context, t *types.Type) ([]*types.Type, error) {
	if t.Kind != types.Struct {
		return nil, nil
	}

	intfs := extractInterfacesTag(t)

	var ts []*types.Type
	for _, intf := range intfs {
		t := types.ParseFullyQualifiedName(intf)
		klog.V(3).Infof("Loading package for interface %v", intf)
		_, err := c.LoadPackages(t.Package)
		if err != nil {
			return nil, err
		}
		intfT := c.Universe.Type(t)
		if intfT == nil {
			return nil, fmt.Errorf("unknown type %q in %s tag of type %s", intf, interfacesTagName, intfT)
		}
		if intfT.Kind != types.Interface {
			return nil, fmt.Errorf("type %q in %s tag of type %s is not an interface, but: %q", intf, interfacesTagName, t, intfT.Kind)
		}
		g.imports.AddType(intfT)
		ts = append(ts, intfT)
	}

	return ts, nil
}

// deepCopyableInterfaces returns the interface types to implement and whether they apply to a non-pointer receiver.
func (g *genDeepCopy) deepCopyableInterfaces(c *generator.Context, t *types.Type) ([]*types.Type, bool, error) {
	ts, err := g.deepCopyableInterfacesInner(c, t)
	if err != nil {
		return nil, false, err
	}

	set := map[string]*types.Type{}
	for _, t := range ts {
		set[t.String()] = t
	}

	result := []*types.Type{}
	for _, t := range set {
		result = append(result, t)
	}

	TypeSlice(result).Sort() // we need a stable sorting because it determines the order in generation

	nonPointerReceiver, err := extractNonPointerInterfaces(t)
	if err != nil {
		return nil, false, err
	}

	return result, nonPointerReceiver, nil
}

type TypeSlice []*types.Type

func (s TypeSlice) Len() int           { return len(s) }
func (s TypeSlice) Less(i, j int) bool { return s[i].String() < s[j].String() }
func (s TypeSlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s TypeSlice) Sort()              { sort.Sort(s) }

func (g *genDeepCopy) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	if !g.needsGeneration(t) {
		return nil
	}
	klog.V(2).Infof("Generating deepcopy functions for type %v", t)

	sw := generator.NewSnippetWriter(w, c, "$", "$")
	args := argsFromType(t)

	if deepCopyIntoMethodOrDie(t) == nil {
		sw.Do("// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.\n", args)
		if isReference(t) {
			sw.Do("func (in $.type|raw$) DeepCopyInto(out *$.type|raw$) {\n", args)
			sw.Do("{in:=&in\n", nil)
		} else {
			sw.Do("func (in *$.type|raw$) DeepCopyInto(out *$.type|raw$) {\n", args)
		}
		if deepCopyMethodOrDie(t) != nil {
			if t.Methods["DeepCopy"].Signature.Receiver.Kind == types.Pointer {
				sw.Do("clone := in.DeepCopy()\n", nil)
				sw.Do("*out = *clone\n", nil)
			} else {
				sw.Do("*out = in.DeepCopy()\n", nil)
			}
			sw.Do("return\n", nil)
		} else {
			g.generateFor(t, sw)
			sw.Do("return\n", nil)
		}
		if isReference(t) {
			sw.Do("}\n", nil)
		}
		sw.Do("}\n\n", nil)
	}

	if deepCopyMethodOrDie(t) == nil {
		sw.Do("// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new $.type|raw$.\n", args)
		if isReference(t) {
			sw.Do("func (in $.type|raw$) DeepCopy() $.type|raw$ {\n", args)
		} else {
			sw.Do("func (in *$.type|raw$) DeepCopy() *$.type|raw$ {\n", args)
		}
		sw.Do("if in == nil { return nil }\n", nil)
		sw.Do("out := new($.type|raw$)\n", args)
		sw.Do("in.DeepCopyInto(out)\n", nil)
		if isReference(t) {
			sw.Do("return *out\n", nil)
		} else {
			sw.Do("return out\n", nil)
		}
		sw.Do("}\n\n", nil)
	}

	intfs, nonPointerReceiver, err := g.deepCopyableInterfaces(c, t)
	if err != nil {
		return err
	}
	for _, intf := range intfs {
		sw.Do(fmt.Sprintf("// DeepCopy%s is an autogenerated deepcopy function, copying the receiver, creating a new $.type2|raw$.\n", intf.Name.Name), argsFromType(t, intf))
		if nonPointerReceiver {
			sw.Do(fmt.Sprintf("func (in $.type|raw$) DeepCopy%s() $.type2|raw$ {\n", intf.Name.Name), argsFromType(t, intf))
			sw.Do("return *in.DeepCopy()", nil)
			sw.Do("}\n\n", nil)
		} else {
			sw.Do(fmt.Sprintf("func (in *$.type|raw$) DeepCopy%s() $.type2|raw$ {\n", intf.Name.Name), argsFromType(t, intf))
			sw.Do("if c := in.DeepCopy(); c != nil {\n", nil)
			sw.Do("return c\n", nil)
			sw.Do("}\n", nil)
			sw.Do("return nil\n", nil)
			sw.Do("}\n\n", nil)
		}
	}

	return sw.Error()
}

// isReference return true for pointer, maps, slices and aliases of those.
func isReference(t *types.Type) bool {
	if t.Kind == types.Pointer || t.Kind == types.Map || t.Kind == types.Slice {
		return true
	}
	return t.Kind == types.Alias && isReference(underlyingType(t))
}

// we use the system of shadowing 'in' and 'out' so that the same code is valid
// at any nesting level. This makes the autogenerator easy to understand, and
// the compiler shouldn't care.
func (g *genDeepCopy) generateFor(t *types.Type, sw *generator.SnippetWriter) {
	// derive inner types if t is an alias. We call the do* methods below with the alias type.
	// basic rule: generate according to inner type, but construct objects with the alias type.
	ut := underlyingType(t)

	var f func(*types.Type, *generator.SnippetWriter)
	switch ut.Kind {
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
	case types.Interface:
		// interfaces are handled in-line in the other cases
		klog.Fatalf("Hit an interface type %v. This should never happen.", t)
	case types.Alias:
		// can never happen because we branch on the underlying type which is never an alias
		klog.Fatalf("Hit an alias type %v. This should never happen.", t)
	default:
		klog.Fatalf("Hit an unsupported type %v.", t)
	}
	f(t, sw)
}

// doBuiltin generates code for a builtin or an alias to a builtin. The generated code is
// is the same for both cases, i.e. it's the code for the underlying type.
func (g *genDeepCopy) doBuiltin(t *types.Type, sw *generator.SnippetWriter) {
	if deepCopyMethodOrDie(t) != nil || deepCopyIntoMethodOrDie(t) != nil {
		sw.Do("*out = in.DeepCopy()\n", nil)
		return
	}

	sw.Do("*out = *in\n", nil)
}

// doMap generates code for a map or an alias to a map. The generated code is
// is the same for both cases, i.e. it's the code for the underlying type.
func (g *genDeepCopy) doMap(t *types.Type, sw *generator.SnippetWriter) {
	ut := underlyingType(t)
	uet := underlyingType(ut.Elem)

	if deepCopyMethodOrDie(t) != nil || deepCopyIntoMethodOrDie(t) != nil {
		sw.Do("*out = in.DeepCopy()\n", nil)
		return
	}

	if !ut.Key.IsAssignable() {
		klog.Fatalf("Hit an unsupported type %v for: %v", uet, t)
	}

	sw.Do("*out = make($.|raw$, len(*in))\n", t)
	sw.Do("for key, val := range *in {\n", nil)
	dc, dci := deepCopyMethodOrDie(ut.Elem), deepCopyIntoMethodOrDie(ut.Elem)
	switch {
	case dc != nil || dci != nil:
		// Note: a DeepCopy exists because it is added if DeepCopyInto is manually defined
		leftPointer := ut.Elem.Kind == types.Pointer
		rightPointer := !isReference(ut.Elem)
		if dc != nil {
			rightPointer = dc.Results[0].Type.Kind == types.Pointer
		}
		if leftPointer == rightPointer {
			sw.Do("(*out)[key] = val.DeepCopy()\n", nil)
		} else if leftPointer {
			sw.Do("x := val.DeepCopy()\n", nil)
			sw.Do("(*out)[key] = &x\n", nil)
		} else {
			sw.Do("(*out)[key] = *val.DeepCopy()\n", nil)
		}
	case ut.Elem.IsAnonymousStruct(): // not uet here because it needs type cast
		sw.Do("(*out)[key] = val\n", nil)
	case uet.IsAssignable():
		sw.Do("(*out)[key] = val\n", nil)
	case uet.Kind == types.Interface:
		// Note: do not generate code that won't compile as `DeepCopyinterface{}()` is not a valid function
		if uet.Name.Name == "interface{}" {
			klog.Fatalf("DeepCopy of %q is unsupported. Instead, use named interfaces with DeepCopy<named-interface> as one of the methods.", uet.Name.Name)
		}
		sw.Do("if val == nil {(*out)[key]=nil} else {\n", nil)
		// Note: if t.Elem has been an alias "J" of an interface "I" in Go, we will see it
		// as kind Interface of name "J" here, i.e. generate val.DeepCopyJ(). The golang
		// parser does not give us the underlying interface name. So we cannot do any better.
		sw.Do(fmt.Sprintf("(*out)[key] = val.DeepCopy%s()\n", uet.Name.Name), nil)
		sw.Do("}\n", nil)
	case uet.Kind == types.Slice || uet.Kind == types.Map || uet.Kind == types.Pointer:
		sw.Do("var outVal $.|raw$\n", uet)
		sw.Do("if val == nil { (*out)[key] = nil } else {\n", nil)
		sw.Do("in, out := &val, &outVal\n", uet)
		g.generateFor(ut.Elem, sw)
		sw.Do("}\n", nil)
		sw.Do("(*out)[key] = outVal\n", nil)
	case uet.Kind == types.Struct:
		sw.Do("(*out)[key] = *val.DeepCopy()\n", uet)
	default:
		klog.Fatalf("Hit an unsupported type %v for %v", uet, t)
	}
	sw.Do("}\n", nil)
}

// doSlice generates code for a slice or an alias to a slice. The generated code is
// is the same for both cases, i.e. it's the code for the underlying type.
func (g *genDeepCopy) doSlice(t *types.Type, sw *generator.SnippetWriter) {
	ut := underlyingType(t)
	uet := underlyingType(ut.Elem)

	if deepCopyMethodOrDie(t) != nil || deepCopyIntoMethodOrDie(t) != nil {
		sw.Do("*out = in.DeepCopy()\n", nil)
		return
	}

	sw.Do("*out = make($.|raw$, len(*in))\n", t)
	if deepCopyMethodOrDie(ut.Elem) != nil || deepCopyIntoMethodOrDie(ut.Elem) != nil {
		sw.Do("for i := range *in {\n", nil)
		// Note: a DeepCopyInto exists because it is added if DeepCopy is manually defined
		sw.Do("(*in)[i].DeepCopyInto(&(*out)[i])\n", nil)
		sw.Do("}\n", nil)
	} else if uet.Kind == types.Builtin || uet.IsAssignable() {
		sw.Do("copy(*out, *in)\n", nil)
	} else {
		sw.Do("for i := range *in {\n", nil)
		if uet.Kind == types.Slice || uet.Kind == types.Map || uet.Kind == types.Pointer || deepCopyMethodOrDie(ut.Elem) != nil || deepCopyIntoMethodOrDie(ut.Elem) != nil {
			sw.Do("if (*in)[i] != nil {\n", nil)
			sw.Do("in, out := &(*in)[i], &(*out)[i]\n", nil)
			g.generateFor(ut.Elem, sw)
			sw.Do("}\n", nil)
		} else if uet.Kind == types.Interface {
			// Note: do not generate code that won't compile as `DeepCopyinterface{}()` is not a valid function
			if uet.Name.Name == "interface{}" {
				klog.Fatalf("DeepCopy of %q is unsupported. Instead, use named interfaces with DeepCopy<named-interface> as one of the methods.", uet.Name.Name)
			}
			sw.Do("if (*in)[i] != nil {\n", nil)
			// Note: if t.Elem has been an alias "J" of an interface "I" in Go, we will see it
			// as kind Interface of name "J" here, i.e. generate val.DeepCopyJ(). The golang
			// parser does not give us the underlying interface name. So we cannot do any better.
			sw.Do(fmt.Sprintf("(*out)[i] = (*in)[i].DeepCopy%s()\n", uet.Name.Name), nil)
			sw.Do("}\n", nil)
		} else if uet.Kind == types.Struct {
			sw.Do("(*in)[i].DeepCopyInto(&(*out)[i])\n", nil)
		} else {
			klog.Fatalf("Hit an unsupported type %v for %v", uet, t)
		}
		sw.Do("}\n", nil)
	}
}

// doStruct generates code for a struct or an alias to a struct. The generated code is
// is the same for both cases, i.e. it's the code for the underlying type.
func (g *genDeepCopy) doStruct(t *types.Type, sw *generator.SnippetWriter) {
	ut := underlyingType(t)

	if deepCopyMethodOrDie(t) != nil || deepCopyIntoMethodOrDie(t) != nil {
		sw.Do("*out = in.DeepCopy()\n", nil)
		return
	}

	// Simple copy covers a lot of cases.
	sw.Do("*out = *in\n", nil)

	// Now fix-up fields as needed.
	for _, m := range ut.Members {
		ft := m.Type
		uft := underlyingType(ft)

		args := generator.Args{
			"type": ft,
			"kind": ft.Kind,
			"name": m.Name,
		}
		dc, dci := deepCopyMethodOrDie(ft), deepCopyIntoMethodOrDie(ft)
		switch {
		case dc != nil || dci != nil:
			// Note: a DeepCopyInto exists because it is added if DeepCopy is manually defined
			leftPointer := ft.Kind == types.Pointer
			rightPointer := !isReference(ft)
			if dc != nil {
				rightPointer = dc.Results[0].Type.Kind == types.Pointer
			}
			if leftPointer == rightPointer {
				sw.Do("out.$.name$ = in.$.name$.DeepCopy()\n", args)
			} else if leftPointer {
				sw.Do("x := in.$.name$.DeepCopy()\n", args)
				sw.Do("out.$.name$ =  = &x\n", args)
			} else {
				sw.Do("in.$.name$.DeepCopyInto(&out.$.name$)\n", args)
			}
		case uft.Kind == types.Builtin:
			// the initial *out = *in was enough
		case uft.Kind == types.Map, uft.Kind == types.Slice, uft.Kind == types.Pointer:
			// Fixup non-nil reference-semantic types.
			sw.Do("if in.$.name$ != nil {\n", args)
			sw.Do("in, out := &in.$.name$, &out.$.name$\n", args)
			g.generateFor(ft, sw)
			sw.Do("}\n", nil)
		case uft.Kind == types.Array:
			sw.Do("out.$.name$ = in.$.name$\n", args)
		case uft.Kind == types.Struct:
			if ft.IsAssignable() {
				sw.Do("out.$.name$ = in.$.name$\n", args)
			} else {
				sw.Do("in.$.name$.DeepCopyInto(&out.$.name$)\n", args)
			}
		case uft.Kind == types.Interface:
			// Note: do not generate code that won't compile as `DeepCopyinterface{}()` is not a valid function
			if uft.Name.Name == "interface{}" {
				klog.Fatalf("DeepCopy of %q is unsupported. Instead, use named interfaces with DeepCopy<named-interface> as one of the methods.", uft.Name.Name)
			}
			sw.Do("if in.$.name$ != nil {\n", args)
			// Note: if t.Elem has been an alias "J" of an interface "I" in Go, we will see it
			// as kind Interface of name "J" here, i.e. generate val.DeepCopyJ(). The golang
			// parser does not give us the underlying interface name. So we cannot do any better.
			sw.Do(fmt.Sprintf("out.$.name$ = in.$.name$.DeepCopy%s()\n", uft.Name.Name), args)
			sw.Do("}\n", nil)
		default:
			klog.Fatalf("Hit an unsupported type '%v' for '%v', from %v.%v", uft, ft, t, m.Name)
		}
	}
}

// doPointer generates code for a pointer or an alias to a pointer. The generated code is
// is the same for both cases, i.e. it's the code for the underlying type.
func (g *genDeepCopy) doPointer(t *types.Type, sw *generator.SnippetWriter) {
	ut := underlyingType(t)
	uet := underlyingType(ut.Elem)

	dc, dci := deepCopyMethodOrDie(ut.Elem), deepCopyIntoMethodOrDie(ut.Elem)
	switch {
	case dc != nil || dci != nil:
		rightPointer := !isReference(ut.Elem)
		if dc != nil {
			rightPointer = dc.Results[0].Type.Kind == types.Pointer
		}
		if rightPointer {
			sw.Do("*out = (*in).DeepCopy()\n", nil)
		} else {
			sw.Do("x := (*in).DeepCopy()\n", nil)
			sw.Do("*out = &x\n", nil)
		}
	case uet.IsAssignable():
		sw.Do("*out = new($.Elem|raw$)\n", ut)
		sw.Do("**out = **in", nil)
	case uet.Kind == types.Map, uet.Kind == types.Slice, uet.Kind == types.Pointer:
		sw.Do("*out = new($.Elem|raw$)\n", ut)
		sw.Do("if **in != nil {\n", nil)
		sw.Do("in, out := *in, *out\n", nil)
		g.generateFor(uet, sw)
		sw.Do("}\n", nil)
	case uet.Kind == types.Struct:
		sw.Do("*out = new($.Elem|raw$)\n", ut)
		sw.Do("(*in).DeepCopyInto(*out)\n", nil)
	default:
		klog.Fatalf("Hit an unsupported type %v for %v", uet, t)
	}
}
