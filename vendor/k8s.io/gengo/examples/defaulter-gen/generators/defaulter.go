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
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"reflect"
	"strings"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/klog/v2"
)

// CustomArgs is used tby the go2idl framework to pass args specific to this
// generator.
type CustomArgs struct {
	ExtraPeerDirs []string // Always consider these as last-ditch possibilities for conversions.
}

var typeZeroValue = map[string]interface{}{
	"uint":        0.,
	"uint8":       0.,
	"uint16":      0.,
	"uint32":      0.,
	"uint64":      0.,
	"int":         0.,
	"int8":        0.,
	"int16":       0.,
	"int32":       0.,
	"int64":       0.,
	"byte":        0,
	"float64":     0.,
	"float32":     0.,
	"bool":        false,
	"time.Time":   "",
	"string":      "",
	"integer":     0.,
	"number":      0.,
	"boolean":     false,
	"[]byte":      "", // base64 encoded characters
	"interface{}": interface{}(nil),
}

// These are the comment tags that carry parameters for defaulter generation.
const tagName = "k8s:defaulter-gen"
const inputTagName = "k8s:defaulter-gen-input"
const defaultTagName = "default"

func extractDefaultTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[defaultTagName]
}

func extractTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[tagName]
}

func extractInputTag(comments []string) []string {
	return types.ExtractCommentTags("+", comments)[inputTagName]
}

func checkTag(comments []string, require ...string) bool {
	values := types.ExtractCommentTags("+", comments)[tagName]
	if len(require) == 0 {
		return len(values) == 1 && values[0] == ""
	}
	return reflect.DeepEqual(values, require)
}

func defaultFnNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Prefix: "SetDefaults_",
		Join: func(pre string, in []string, post string) string {
			return pre + strings.Join(in, "_") + post
		},
	}
}

func objectDefaultFnNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Prefix: "SetObjectDefaults_",
		Join: func(pre string, in []string, post string) string {
			return pre + strings.Join(in, "_") + post
		},
	}
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public":          namer.NewPublicNamer(1),
		"raw":             namer.NewRawNamer("", nil),
		"defaultfn":       defaultFnNamer(),
		"objectdefaultfn": objectDefaultFnNamer(),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// defaults holds the declared defaulting functions for a given type (all defaulting functions
// are expected to be func(1))
type defaults struct {
	// object is the defaulter function for a top level type (typically one with TypeMeta) that
	// invokes all child defaulters. May be nil if the object defaulter has not yet been generated.
	object *types.Type
	// base is a defaulter function defined for a type SetDefaults_Pod which does not invoke all
	// child defaults - the base defaulter alone is insufficient to default a type
	base *types.Type
	// additional is zero or more defaulter functions of the form SetDefaults_Pod_XXXX that can be
	// included in the Object defaulter.
	additional []*types.Type
}

// All of the types in conversions map are of type "DeclarationOf" with
// the underlying type being "Func".
type defaulterFuncMap map[*types.Type]defaults

// Returns all manually-defined defaulting functions in the package.
func getManualDefaultingFunctions(context *generator.Context, pkg *types.Package, manualMap defaulterFuncMap) {
	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")

	for _, f := range pkg.Functions {
		if f.Underlying == nil || f.Underlying.Kind != types.Func {
			klog.Errorf("Malformed function: %#v", f)
			continue
		}
		if f.Underlying.Signature == nil {
			klog.Errorf("Function without signature: %#v", f)
			continue
		}
		signature := f.Underlying.Signature
		// Check whether the function is defaulting function.
		// Note that all of them have signature:
		// object: func SetObjectDefaults_inType(*inType)
		// base: func SetDefaults_inType(*inType)
		// additional: func SetDefaults_inType_Qualifier(*inType)
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
		// Check if this is the primary defaulter.
		args := defaultingArgsFromType(inType.Elem)
		sw.Do("$.inType|defaultfn$", args)
		switch {
		case f.Name.Name == buffer.String():
			key := inType.Elem
			// We might scan the same package twice, and that's OK.
			v, ok := manualMap[key]
			if ok && v.base != nil && v.base.Name.Package != pkg.Path {
				panic(fmt.Sprintf("duplicate static defaulter defined: %#v", key))
			}
			v.base = f
			manualMap[key] = v
			klog.V(6).Infof("found base defaulter function for %s from %s", key.Name, f.Name)
		// Is one of the additional defaulters - a top level defaulter on a type that is
		// also invoked.
		case strings.HasPrefix(f.Name.Name, buffer.String()+"_"):
			key := inType.Elem
			v, ok := manualMap[key]
			if ok {
				exists := false
				for _, existing := range v.additional {
					if existing.Name == f.Name {
						exists = true
						break
					}
				}
				if exists {
					continue
				}
			}
			v.additional = append(v.additional, f)
			manualMap[key] = v
			klog.V(6).Infof("found additional defaulter function for %s from %s", key.Name, f.Name)
		}
		buffer.Reset()
		sw.Do("$.inType|objectdefaultfn$", args)
		if f.Name.Name == buffer.String() {
			key := inType.Elem
			// We might scan the same package twice, and that's OK.
			v, ok := manualMap[key]
			if ok && v.base != nil && v.base.Name.Package != pkg.Path {
				panic(fmt.Sprintf("duplicate static defaulter defined: %#v", key))
			}
			v.object = f
			manualMap[key] = v
			klog.V(6).Infof("found object defaulter function for %s from %s", key.Name, f.Name)
		}
		buffer.Reset()
	}
}

func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	packages := generator.Packages{}
	header := append([]byte(fmt.Sprintf("// +build !%s\n\n", arguments.GeneratedBuildTag)), boilerplate...)

	// Accumulate pre-existing default functions.
	// TODO: This is too ad-hoc.  We need a better way.
	existingDefaulters := defaulterFuncMap{}

	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")

	// We are generating defaults only for packages that are explicitly
	// passed as InputDir.
	for _, i := range context.Inputs {
		klog.V(5).Infof("considering pkg %q", i)
		pkg := context.Universe[i]
		if pkg == nil {
			// If the input had no Go files, for example.
			continue
		}
		// typesPkg is where the types that needs defaulter are defined.
		// Sometimes it is different from pkg. For example, kubernetes core/v1
		// types are defined in vendor/k8s.io/api/core/v1, while pkg is at
		// pkg/api/v1.
		typesPkg := pkg

		// Add defaulting functions.
		getManualDefaultingFunctions(context, pkg, existingDefaulters)

		var peerPkgs []string
		if customArgs, ok := arguments.CustomArgs.(*CustomArgs); ok {
			for _, pkg := range customArgs.ExtraPeerDirs {
				if i := strings.Index(pkg, "/vendor/"); i != -1 {
					pkg = pkg[i+len("/vendor/"):]
				}
				peerPkgs = append(peerPkgs, pkg)
			}
		}
		// Make sure our peer-packages are added and fully parsed.
		for _, pp := range peerPkgs {
			context.AddDir(pp)
			getManualDefaultingFunctions(context, context.Universe[pp], existingDefaulters)
		}

		typesWith := extractTag(pkg.Comments)
		shouldCreateObjectDefaulterFn := func(t *types.Type) bool {
			if defaults, ok := existingDefaulters[t]; ok && defaults.object != nil {
				// A default generator is defined
				baseTypeName := "<unknown>"
				if defaults.base != nil {
					baseTypeName = defaults.base.Name.String()
				}
				klog.V(5).Infof("  an object defaulter already exists as %s", baseTypeName)
				return false
			}
			// opt-out
			if checkTag(t.SecondClosestCommentLines, "false") {
				return false
			}
			// opt-in
			if checkTag(t.SecondClosestCommentLines, "true") {
				return true
			}
			// For every k8s:defaulter-gen tag at the package level, interpret the value as a
			// field name (like TypeMeta, ListMeta, ObjectMeta) and trigger defaulter generation
			// for any type with any of the matching field names. Provides a more useful package
			// level defaulting than global (because we only need defaulters on a subset of objects -
			// usually those with TypeMeta).
			if t.Kind == types.Struct && len(typesWith) > 0 {
				for _, field := range t.Members {
					for _, s := range typesWith {
						if field.Name == s {
							return true
						}
					}
				}
			}
			return false
		}

		// if the types are not in the same package where the defaulter functions to be generated
		inputTags := extractInputTag(pkg.Comments)
		if len(inputTags) > 1 {
			panic(fmt.Sprintf("there could only be one input tag, got %#v", inputTags))
		}
		if len(inputTags) == 1 {
			var err error
			typesPkg, err = context.AddDirectory(filepath.Join(pkg.Path, inputTags[0]))
			if err != nil {
				klog.Fatalf("cannot import package %s", inputTags[0])
			}
			// update context.Order to the latest context.Universe
			orderer := namer.Orderer{Namer: namer.NewPublicNamer(1)}
			context.Order = orderer.OrderUniverse(context.Universe)
		}

		newDefaulters := defaulterFuncMap{}
		for _, t := range typesPkg.Types {
			if !shouldCreateObjectDefaulterFn(t) {
				continue
			}
			if namer.IsPrivateGoName(t.Name.Name) {
				// We won't be able to convert to a private type.
				klog.V(5).Infof("  found a type %v, but it is a private name", t)
				continue
			}

			// create a synthetic type we can use during generation
			newDefaulters[t] = defaults{}
		}

		// only generate defaulters for objects that actually have defined defaulters
		// prevents empty defaulters from being registered
		for {
			promoted := 0
			for t, d := range newDefaulters {
				if d.object != nil {
					continue
				}
				if newCallTreeForType(existingDefaulters, newDefaulters).build(t, true) != nil {
					args := defaultingArgsFromType(t)
					sw.Do("$.inType|objectdefaultfn$", args)
					newDefaulters[t] = defaults{
						object: &types.Type{
							Name: types.Name{
								Package: pkg.Path,
								Name:    buffer.String(),
							},
							Kind: types.Func,
						},
					}
					buffer.Reset()
					promoted++
				}
			}
			if promoted != 0 {
				continue
			}

			// prune any types that were not used
			for t, d := range newDefaulters {
				if d.object == nil {
					klog.V(6).Infof("did not generate defaulter for %s because no child defaulters were registered", t.Name)
					delete(newDefaulters, t)
				}
			}
			break
		}

		if len(newDefaulters) == 0 {
			klog.V(5).Infof("no defaulters in package %s", pkg.Name)
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
						NewGenDefaulter(arguments.OutputFileBaseName, typesPkg.Path, pkg.Path, existingDefaulters, newDefaulters, peerPkgs),
					}
				},
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == typesPkg.Path
				},
			})
	}
	return packages
}

// callTreeForType contains fields necessary to build a tree for types.
type callTreeForType struct {
	existingDefaulters     defaulterFuncMap
	newDefaulters          defaulterFuncMap
	currentlyBuildingTypes map[*types.Type]bool
}

func newCallTreeForType(existingDefaulters, newDefaulters defaulterFuncMap) *callTreeForType {
	return &callTreeForType{
		existingDefaulters:     existingDefaulters,
		newDefaulters:          newDefaulters,
		currentlyBuildingTypes: make(map[*types.Type]bool),
	}
}

func resolveTypeAndDepth(t *types.Type) (*types.Type, int) {
	var prev *types.Type
	depth := 0
	for prev != t {
		prev = t
		if t.Kind == types.Alias {
			t = t.Underlying
		} else if t.Kind == types.Pointer {
			t = t.Elem
			depth += 1
		}
	}
	return t, depth
}

// getNestedDefault returns the first default value when resolving alias types
func getNestedDefault(t *types.Type) string {
	var prev *types.Type
	for prev != t {
		prev = t
		defaultMap := extractDefaultTag(t.CommentLines)
		if len(defaultMap) == 1 && defaultMap[0] != "" {
			return defaultMap[0]
		}
		if t.Kind == types.Alias {
			t = t.Underlying
		} else if t.Kind == types.Pointer {
			t = t.Elem
		}
	}
	return ""
}

func mustEnforceDefault(t *types.Type, depth int, omitEmpty bool) (interface{}, error) {
	if depth > 0 {
		return nil, nil
	}
	switch t.Kind {
	case types.Pointer, types.Map, types.Slice, types.Array, types.Interface:
		return nil, nil
	case types.Struct:
		return map[string]interface{}{}, nil
	case types.Builtin:
		if !omitEmpty {
			if zero, ok := typeZeroValue[t.String()]; ok {
				return zero, nil
			} else {
				return nil, fmt.Errorf("please add type %v to typeZeroValue struct", t)
			}
		}
		return nil, nil
	default:
		return nil, fmt.Errorf("not sure how to enforce default for %v", t.Kind)
	}
}

func populateDefaultValue(node *callNode, t *types.Type, tags string, commentLines []string) *callNode {
	defaultMap := extractDefaultTag(commentLines)
	var defaultString string
	if len(defaultMap) == 1 {
		defaultString = defaultMap[0]
	}

	t, depth := resolveTypeAndDepth(t)
	if depth > 0 && defaultString == "" {
		defaultString = getNestedDefault(t)
	}
	if len(defaultMap) > 1 {
		klog.Fatalf("Found more than one default tag for %v", t.Kind)
	} else if len(defaultMap) == 0 {
		return node
	}
	var defaultValue interface{}
	if err := json.Unmarshal([]byte(defaultString), &defaultValue); err != nil {
		klog.Fatalf("Failed to unmarshal default: %v", err)
	}

	omitEmpty := strings.Contains(reflect.StructTag(tags).Get("json"), "omitempty")
	if enforced, err := mustEnforceDefault(t, depth, omitEmpty); err != nil {
		klog.Fatal(err)
	} else if enforced != nil {
		if defaultValue != nil {
			if reflect.DeepEqual(defaultValue, enforced) {
				// If the default value annotation matches the default value for the type,
				// do not generate any defaulting function
				return node
			} else {
				enforcedJSON, _ := json.Marshal(enforced)
				klog.Fatalf("Invalid default value (%#v) for non-pointer/non-omitempty. If specified, must be: %v", defaultValue, string(enforcedJSON))
			}
		}
	}

	// callNodes are not automatically generated for primitive types. Generate one if the callNode does not exist
	if node == nil {
		node = &callNode{}
		node.markerOnly = true
	}

	node.defaultIsPrimitive = t.IsPrimitive()
	node.defaultType = t.String()
	node.defaultValue = defaultString
	node.defaultDepth = depth
	return node
}

// build creates a tree of paths to fields (based on how they would be accessed in Go - pointer, elem,
// slice, or key) and the functions that should be invoked on each field. An in-order traversal of the resulting tree
// can be used to generate a Go function that invokes each nested function on the appropriate type. The return
// value may be nil if there are no functions to call on type or the type is a primitive (Defaulters can only be
// invoked on structs today). When root is true this function will not use a newDefaulter. existingDefaulters should
// contain all defaulting functions by type defined in code - newDefaulters should contain all object defaulters
// that could be or will be generated. If newDefaulters has an entry for a type, but the 'object' field is nil,
// this function skips adding that defaulter - this allows us to avoid generating object defaulter functions for
// list types that call empty defaulters.
func (c *callTreeForType) build(t *types.Type, root bool) *callNode {
	parent := &callNode{}

	if root {
		// the root node is always a pointer
		parent.elem = true
	}

	defaults, _ := c.existingDefaulters[t]
	newDefaults, generated := c.newDefaulters[t]
	switch {
	case !root && generated && newDefaults.object != nil:
		parent.call = append(parent.call, newDefaults.object)
		// if we will be generating the defaulter, it by definition is a covering
		// defaulter, so we halt recursion
		klog.V(6).Infof("the defaulter %s will be generated as an object defaulter", t.Name)
		return parent

	case defaults.object != nil:
		// object defaulters are always covering
		parent.call = append(parent.call, defaults.object)
		return parent

	case defaults.base != nil:
		parent.call = append(parent.call, defaults.base)
		// if the base function indicates it "covers" (it already includes defaulters)
		// we can halt recursion
		if checkTag(defaults.base.CommentLines, "covers") {
			klog.V(6).Infof("the defaulter %s indicates it covers all sub generators", t.Name)
			return parent
		}
	}

	// base has been added already, now add any additional defaulters defined for this object
	parent.call = append(parent.call, defaults.additional...)

	// if the type already exists, don't build the tree for it and don't generate anything.
	// This is used to avoid recursion for nested recursive types.
	if c.currentlyBuildingTypes[t] {
		return nil
	}
	// if type doesn't exist, mark it as existing
	c.currentlyBuildingTypes[t] = true

	defer func() {
		// The type will now acts as a parent, not a nested recursive type.
		// We can now build the tree for it safely.
		c.currentlyBuildingTypes[t] = false
	}()

	switch t.Kind {
	case types.Pointer:
		if child := c.build(t.Elem, false); child != nil {
			child.elem = true
			parent.children = append(parent.children, *child)
		}
	case types.Slice, types.Array:
		if child := c.build(t.Elem, false); child != nil {
			child.index = true
			if t.Elem.Kind == types.Pointer {
				child.elem = true
			}
			parent.children = append(parent.children, *child)
		} else if member := populateDefaultValue(nil, t.Elem, "", t.Elem.CommentLines); member != nil {
			member.index = true
			parent.children = append(parent.children, *member)
		}
	case types.Map:
		if child := c.build(t.Elem, false); child != nil {
			child.key = true
			parent.children = append(parent.children, *child)
		} else if member := populateDefaultValue(nil, t.Elem, "", t.Elem.CommentLines); member != nil {
			member.key = true
			parent.children = append(parent.children, *member)
		}

	case types.Struct:
		for _, field := range t.Members {
			name := field.Name
			if len(name) == 0 {
				if field.Type.Kind == types.Pointer {
					name = field.Type.Elem.Name.Name
				} else {
					name = field.Type.Name.Name
				}
			}
			if child := c.build(field.Type, false); child != nil {
				child.field = name
				populateDefaultValue(child, field.Type, field.Tags, field.CommentLines)
				parent.children = append(parent.children, *child)
			} else if member := populateDefaultValue(nil, field.Type, field.Tags, field.CommentLines); member != nil {
				member.field = name
				parent.children = append(parent.children, *member)
			}
		}
	case types.Alias:
		if child := c.build(t.Underlying, false); child != nil {
			parent.children = append(parent.children, *child)
		}
	}
	if len(parent.children) == 0 && len(parent.call) == 0 {
		//klog.V(6).Infof("decided type %s needs no generation", t.Name)
		return nil
	}
	return parent
}

const (
	runtimePackagePath    = "k8s.io/apimachinery/pkg/runtime"
	conversionPackagePath = "k8s.io/apimachinery/pkg/conversion"
)

// genDefaulter produces a file with a autogenerated conversions.
type genDefaulter struct {
	generator.DefaultGen
	typesPackage       string
	outputPackage      string
	peerPackages       []string
	newDefaulters      defaulterFuncMap
	existingDefaulters defaulterFuncMap
	imports            namer.ImportTracker
	typesForInit       []*types.Type
}

func NewGenDefaulter(sanitizedName, typesPackage, outputPackage string, existingDefaulters, newDefaulters defaulterFuncMap, peerPkgs []string) generator.Generator {
	return &genDefaulter{
		DefaultGen: generator.DefaultGen{
			OptionalName: sanitizedName,
		},
		typesPackage:       typesPackage,
		outputPackage:      outputPackage,
		peerPackages:       peerPkgs,
		newDefaulters:      newDefaulters,
		existingDefaulters: existingDefaulters,
		imports:            generator.NewImportTracker(),
		typesForInit:       make([]*types.Type, 0),
	}
}

func (g *genDefaulter) Namers(c *generator.Context) namer.NameSystems {
	// Have the raw namer for this file track what it imports.
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *genDefaulter) isOtherPackage(pkg string) bool {
	if pkg == g.outputPackage {
		return false
	}
	if strings.HasSuffix(pkg, `"`+g.outputPackage+`"`) {
		return false
	}
	return true
}

func (g *genDefaulter) Filter(c *generator.Context, t *types.Type) bool {
	defaults, ok := g.newDefaulters[t]
	if !ok || defaults.object == nil {
		return false
	}
	g.typesForInit = append(g.typesForInit, t)
	return true
}

func (g *genDefaulter) Imports(c *generator.Context) (imports []string) {
	var importLines []string
	for _, singleImport := range g.imports.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func (g *genDefaulter) Init(c *generator.Context, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "$", "$")

	scheme := c.Universe.Type(types.Name{Package: runtimePackagePath, Name: "Scheme"})
	schemePtr := &types.Type{
		Kind: types.Pointer,
		Elem: scheme,
	}
	sw.Do("// RegisterDefaults adds defaulters functions to the given scheme.\n", nil)
	sw.Do("// Public to allow building arbitrary schemes.\n", nil)
	sw.Do("// All generated defaulters are covering - they call all nested defaulters.\n", nil)
	sw.Do("func RegisterDefaults(scheme $.|raw$) error {\n", schemePtr)
	for _, t := range g.typesForInit {
		args := defaultingArgsFromType(t)
		sw.Do("scheme.AddTypeDefaultingFunc(&$.inType|raw${}, func(obj interface{}) { $.inType|objectdefaultfn$(obj.(*$.inType|raw$)) })\n", args)
	}
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)
	return sw.Error()
}

func (g *genDefaulter) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	if _, ok := g.newDefaulters[t]; !ok {
		return nil
	}

	klog.V(5).Infof("generating for type %v", t)

	callTree := newCallTreeForType(g.existingDefaulters, g.newDefaulters).build(t, true)
	if callTree == nil {
		klog.V(5).Infof("  no defaulters defined")
		return nil
	}
	i := 0
	callTree.VisitInOrder(func(ancestors []*callNode, current *callNode) {
		if len(current.call) == 0 {
			return
		}
		path := callPath(append(ancestors, current))
		klog.V(5).Infof("  %d: %s", i, path)
		i++
	})

	sw := generator.NewSnippetWriter(w, c, "$", "$")
	g.generateDefaulter(t, callTree, sw)
	return sw.Error()
}

func defaultingArgsFromType(inType *types.Type) generator.Args {
	return generator.Args{
		"inType": inType,
	}
}

func (g *genDefaulter) generateDefaulter(inType *types.Type, callTree *callNode, sw *generator.SnippetWriter) {
	sw.Do("func $.inType|objectdefaultfn$(in *$.inType|raw$) {\n", defaultingArgsFromType(inType))
	callTree.WriteMethod("in", 0, nil, sw)
	sw.Do("}\n\n", nil)
}

// callNode represents an entry in a tree of Go type accessors - the path from the root to a leaf represents
// how in Go code an access would be performed. For example, if a defaulting function exists on a container
// lifecycle hook, to invoke that defaulter correctly would require this Go code:
//
//     for i := range pod.Spec.Containers {
//       o := &pod.Spec.Containers[i]
//       if o.LifecycleHook != nil {
//         SetDefaults_LifecycleHook(o.LifecycleHook)
//       }
//     }
//
// That would be represented by a call tree like:
//
//   callNode
//     field: "Spec"
//     children:
//     - field: "Containers"
//       children:
//       - index: true
//         children:
//         - field: "LifecycleHook"
//           elem: true
//           call:
//           - SetDefaults_LifecycleHook
//
// which we can traverse to build that Go struct (you must call the field Spec, then Containers, then range over
// that field, then check whether the LifecycleHook field is nil, before calling SetDefaults_LifecycleHook on
// the pointer to that field).
type callNode struct {
	// field is the name of the Go member to access
	field string
	// key is true if this is a map and we must range over the key and values
	key bool
	// index is true if this is a slice and we must range over the slice values
	index bool
	// elem is true if the previous elements refer to a pointer (typically just field)
	elem bool

	// call is all of the functions that must be invoked on this particular node, in order
	call []*types.Type
	// children is the child call nodes that must also be traversed
	children []callNode

	// defaultValue is the defaultValue of a callNode struct
	// Only primitive types and pointer types are eligible to have a default value
	defaultValue string

	// defaultIsPrimitive is used to determine how to assign the default value.
	// Primitive types will be directly assigned while complex types will use JSON unmarshalling
	defaultIsPrimitive bool

	// markerOnly is true if the callNode exists solely to fill in a default value
	markerOnly bool

	// defaultDepth is used to determine pointer level of the default value
	// For example 1 corresponds to setting a default value and taking its pointer while
	// 2 corresponds to setting a default value and taking its pointer's pointer
	// 0 implies that no pointers are used
	defaultDepth int

	// defaultType is the type of the default value.
	// Only populated if defaultIsPrimitive is true
	defaultType string
}

// CallNodeVisitorFunc is a function for visiting a call tree. ancestors is the list of all parents
// of this node to the root of the tree - will be empty at the root.
type CallNodeVisitorFunc func(ancestors []*callNode, node *callNode)

func (n *callNode) VisitInOrder(fn CallNodeVisitorFunc) {
	n.visitInOrder(nil, fn)
}

func (n *callNode) visitInOrder(ancestors []*callNode, fn CallNodeVisitorFunc) {
	fn(ancestors, n)
	ancestors = append(ancestors, n)
	for i := range n.children {
		n.children[i].visitInOrder(ancestors, fn)
	}
}

var (
	indexVariables = "ijklmnop"
	localVariables = "abcdefgh"
)

// varsForDepth creates temporary variables guaranteed to be unique within lexical Go scopes
// of this depth in a function. It uses canonical Go loop variables for the first 7 levels
// and then resorts to uglier prefixes.
func varsForDepth(depth int) (index, local string) {
	if depth > len(indexVariables) {
		index = fmt.Sprintf("i%d", depth)
	} else {
		index = indexVariables[depth : depth+1]
	}
	if depth > len(localVariables) {
		local = fmt.Sprintf("local%d", depth)
	} else {
		local = localVariables[depth : depth+1]
	}
	return
}

// writeCalls generates a list of function calls based on the calls field for the provided variable
// name and pointer.
func (n *callNode) writeCalls(varName string, isVarPointer bool, sw *generator.SnippetWriter) {
	accessor := varName
	if !isVarPointer {
		accessor = "&" + accessor
	}
	for _, fn := range n.call {
		sw.Do("$.fn|raw$($.var$)\n", generator.Args{
			"fn":  fn,
			"var": accessor,
		})
	}
}

func (n *callNode) writeDefaulter(varName string, index string, isVarPointer bool, sw *generator.SnippetWriter) {
	if n.defaultValue == "" {
		return
	}
	varPointer := varName
	if !isVarPointer {
		varPointer = "&" + varPointer
	}

	args := generator.Args{
		"defaultValue": n.defaultValue,
		"varPointer":   varPointer,
		"varName":      varName,
		"index":        index,
		"varDepth":     n.defaultDepth,
		"varType":      n.defaultType,
	}

	if n.index {
		sw.Do("if reflect.ValueOf($.var$[$.index$]).IsZero() {\n", generator.Args{"var": varName, "index": index})
		if n.defaultIsPrimitive {
			if n.defaultDepth > 0 {
				sw.Do("var ptrVar$.varDepth$ $.varType$ = $.defaultValue$\n", args)
				for i := n.defaultDepth; i > 0; i-- {
					sw.Do("ptrVar$.ptri$ := &ptrVar$.i$\n", generator.Args{"i": fmt.Sprintf("%d", i), "ptri": fmt.Sprintf("%d", (i - 1))})
				}
				sw.Do("$.varName$[$.index$] = ptrVar0", args)
			} else {
				sw.Do("$.varName$[$.index$] = $.defaultValue$", args)
			}
		} else {
			sw.Do("if err := json.Unmarshal([]byte(`$.defaultValue$`), $.varPointer$[$.index$]); err != nil {\n", args)
			sw.Do("panic(err)\n", nil)
			sw.Do("}\n", nil)
		}
	} else if n.key {
		mapDefaultVar := index + "_default"
		args["mapDefaultVar"] = mapDefaultVar
		sw.Do("if reflect.ValueOf($.var$[$.index$]).IsZero() {\n", generator.Args{"var": varName, "index": index})

		if n.defaultIsPrimitive {
			if n.defaultDepth > 0 {
				sw.Do("var ptrVar$.varDepth$ $.varType$ = $.defaultValue$\n", args)
				for i := n.defaultDepth; i > 0; i-- {
					sw.Do("ptrVar$.ptri$ := &ptrVar$.i$\n", generator.Args{"i": fmt.Sprintf("%d", i), "ptri": fmt.Sprintf("%d", (i - 1))})
				}
				sw.Do("$.varName$[$.index$] = ptrVar0", args)
			} else {
				sw.Do("$.varName$[$.index$] = $.defaultValue$", args)
			}
		} else {
			sw.Do("$.mapDefaultVar$ := $.varName$[$.index$]\n", args)
			sw.Do("if err := json.Unmarshal([]byte(`$.defaultValue$`), &$.mapDefaultVar$); err != nil {\n", args)
			sw.Do("panic(err)\n", nil)
			sw.Do("}\n", nil)
			sw.Do("$.varName$[$.index$] = $.mapDefaultVar$\n", args)
		}
	} else {
		sw.Do("if reflect.ValueOf($.var$).IsZero() {\n", generator.Args{"var": varName})

		if n.defaultIsPrimitive {
			if n.defaultDepth > 0 {
				sw.Do("var ptrVar$.varDepth$ $.varType$ = $.defaultValue$\n", args)
				for i := n.defaultDepth; i > 0; i-- {
					sw.Do("ptrVar$.ptri$ := &ptrVar$.i$\n", generator.Args{"i": fmt.Sprintf("%d", i), "ptri": fmt.Sprintf("%d", (i - 1))})
				}
				sw.Do("$.varName$ = ptrVar0", args)
			} else {
				sw.Do("$.varName$ = $.defaultValue$", args)
			}
		} else {
			sw.Do("if err := json.Unmarshal([]byte(`$.defaultValue$`), $.varPointer$); err != nil {\n", args)
			sw.Do("panic(err)\n", nil)
			sw.Do("}\n", nil)
		}
	}
	sw.Do("}\n", nil)
}

// WriteMethod performs an in-order traversal of the calltree, generating loops and if blocks as necessary
// to correctly turn the call tree into a method body that invokes all calls on all child nodes of the call tree.
// Depth is used to generate local variables at the proper depth.
func (n *callNode) WriteMethod(varName string, depth int, ancestors []*callNode, sw *generator.SnippetWriter) {
	// if len(n.call) > 0 {
	// 	sw.Do(fmt.Sprintf("// %s\n", callPath(append(ancestors, n)).String()), nil)
	// }

	if len(n.field) > 0 {
		varName = varName + "." + n.field
	}

	index, local := varsForDepth(depth)
	vars := generator.Args{
		"index": index,
		"local": local,
		"var":   varName,
	}

	isPointer := n.elem && !n.index
	if isPointer && len(ancestors) > 0 {
		sw.Do("if $.var$ != nil {\n", vars)
	}

	switch {
	case n.index:
		sw.Do("for $.index$ := range $.var$ {\n", vars)
		if !n.markerOnly {
			if n.elem {
				sw.Do("$.local$ := $.var$[$.index$]\n", vars)
			} else {
				sw.Do("$.local$ := &$.var$[$.index$]\n", vars)
			}
		}

		n.writeDefaulter(varName, index, isPointer, sw)
		n.writeCalls(local, true, sw)
		for i := range n.children {
			n.children[i].WriteMethod(local, depth+1, append(ancestors, n), sw)
		}
		sw.Do("}\n", nil)
	case n.key:
		if n.defaultValue != "" {
			// Map keys are typed and cannot share the same index variable as arrays and other maps
			index = index + "_" + ancestors[len(ancestors)-1].field
			vars["index"] = index
			sw.Do("for $.index$ := range $.var$ {\n", vars)
			n.writeDefaulter(varName, index, isPointer, sw)
			sw.Do("}\n", nil)
		}
	default:
		n.writeDefaulter(varName, index, isPointer, sw)
		n.writeCalls(varName, isPointer, sw)
		for i := range n.children {
			n.children[i].WriteMethod(varName, depth, append(ancestors, n), sw)
		}
	}

	if isPointer && len(ancestors) > 0 {
		sw.Do("}\n", nil)
	}
}

type callPath []*callNode

// String prints a representation of a callPath that roughly approximates what a Go accessor
// would look like. Used for debugging only.
func (path callPath) String() string {
	if len(path) == 0 {
		return "<none>"
	}
	var parts []string
	for _, p := range path {
		last := len(parts) - 1
		switch {
		case p.elem:
			if len(parts) > 0 {
				parts[last] = "*" + parts[last]
			} else {
				parts = append(parts, "*")
			}
		case p.index:
			if len(parts) > 0 {
				parts[last] = parts[last] + "[i]"
			} else {
				parts = append(parts, "[i]")
			}
		case p.key:
			if len(parts) > 0 {
				parts[last] = parts[last] + "[key]"
			} else {
				parts = append(parts, "[key]")
			}
		default:
			if len(p.field) > 0 {
				parts = append(parts, p.field)
			} else {
				parts = append(parts, "<root>")
			}
		}
	}
	var calls []string
	for _, fn := range path[len(path)-1].call {
		calls = append(calls, fn.Name.String())
	}
	if len(calls) == 0 {
		calls = append(calls, "<none>")
	}

	return strings.Join(parts, ".") + " calls " + strings.Join(calls, ", ")
}
