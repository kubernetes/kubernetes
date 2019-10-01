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

package generator

import (
	"fmt"
	"io"
	"strings"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/klog"
)

type ConversionGenerator struct {
	generator.DefaultGen

	// ImportTracker tracks the raw namer's imports.
	// It can be re-used by wrapper generators.
	ImportTracker namer.ImportTracker

	/* Internal state */

	// typesPackage is the package that contains the types that conversion func are going to be
	// generated for.
	typesPackage string
	// outputPackage is the package that the conversion funcs are going to be output to.
	outputPackage string
	// peerPackages are the packages that contain the peer of types in typesPacakge.
	peerPackages []string
	// manualConversionsTracker finds and caches which manually defined exist.
	manualConversionsTracker *ManualConversionsTracker
	// unsafeConversionArbitrator allows comparing types' memory layouts to decide whether
	// to use unsafe conversions.
	unsafeConversionArbitrator *unsafeConversionArbitrator
	// peerTypes caches the peer types found so far.
	peerTypes map[string]*types.Type

	/* Configurable settings (through the With* methods) */

	// see comment on WithTagName
	tagName string
	// see comment on WithFunctionTagName
	functionTagName string
	// see comment on WithMissingFieldsHandler
	missingFieldsHandler func(inVar, outVar NamedVariable, member *types.Member, sw *generator.SnippetWriter) error
	// see comment on WithInconvertibleFieldsHandler
	inconvertibleFieldsHandler func(inVar, outVar NamedVariable, inMember, outMember *types.Member, sw *generator.SnippetWriter) error
	// see comment on WithUnsupportedTypesHandler
	unsupportedTypesHandler func(inVar, outVar NamedVariable, sw *generator.SnippetWriter) error
	// see comment on WithExternalConversionsHandler
	externalConversionsHandler func(inVar, outVar NamedVariable, sw *generator.SnippetWriter) (bool, error)
}

// NewConversionGenerator builds a new ConversionGenerator.
// The manual conversion tracker can be nil, but should be set either if there are additional conversion
// arguments, or to re-use a single tracker across several generators, for efficiency.
func NewConversionGenerator(context *generator.Context, outputFileName, typesPackage, outputPackage string, peerPackages []string, manualConversionsTracker *ManualConversionsTracker) (*ConversionGenerator, error) {
	tracker := manualConversionsTracker
	if tracker == nil {
		tracker = NewManualConversionsTracker()
	}
	if err := findManualConversionFunctions(context, tracker, append(peerPackages, outputPackage, typesPackage)); err != nil {
		return nil, err
	}

	return &ConversionGenerator{
		DefaultGen: generator.DefaultGen{
			OptionalName: outputFileName,
		},
		ImportTracker: generator.NewImportTracker(),

		typesPackage:  typesPackage,
		outputPackage: outputPackage,
		peerPackages:  peerPackages,

		manualConversionsTracker:   tracker,
		unsafeConversionArbitrator: newUnsafeConversionArbitrator(tracker),
		peerTypes:                  make(map[string]*types.Type),
	}, nil
}

func findManualConversionFunctions(context *generator.Context, tracker *ManualConversionsTracker, packagePaths []string) error {
	for _, packagePath := range packagePaths {
		if errors := tracker.findManualConversionFunctions(context, packagePath); len(errors) != 0 {
			errMsg := "Errors when looking for manual conversion functions in " + packagePath + ":"
			for _, err := range errors {
				errMsg += "\n" + err.Error()
			}
			return fmt.Errorf(errMsg)
		}
	}
	return nil
}

// WithTagName allows setting the tag name, ie the marker that this generator
// will look for in comments on types.
// * "+<tag-name>=false" in a type's comment will instruct conversion-gen to skip that type.
func (g *ConversionGenerator) WithTagName(tagName string) *ConversionGenerator {
	g.tagName = tagName
	return g
}

// WithFunctionTagName allows setting the function tag name, ie the marker that this generator
// will look for in comments on manual conversion functions. In a function's comments:
// * "+<tag-name>=copy-only" : copy-only functions that are directly assignable can be inlined
// 	 instead of invoked. As an example, conversion functions exist that allow types with private
//   fields to be correctly copied between types. These functions are equivalent to a memory assignment,
//	 and are necessary for the reflection path, but should not block memory conversion.
// * "+<tag-name>=drop" means to drop that conversion altogether.
func (g *ConversionGenerator) WithFunctionTagName(functionTagName string) *ConversionGenerator {
	g.functionTagName = functionTagName
	if g.unsafeConversionArbitrator != nil {
		g.unsafeConversionArbitrator.setFunctionTagName(functionTagName)
	}
	return g
}

// WithManualConversionsTracker allows setting the ManualConversionsTracker that this generator uses.
// This is convenient to re-use the same tracker for multiple generators, thus avoiding to re-do the
// work of looking for manual conversions in the same packages several times - which is especially
// notably for peer packages, which often are the same across multiple generators.
// Note that also sets the additional conversion arguments to be those of the tracker
// (see WithAdditionalConversionArguments).
func (g *ConversionGenerator) WithManualConversionsTracker(tracker *ManualConversionsTracker) *ConversionGenerator {
	g.manualConversionsTracker = tracker
	return g
}

// WithoutUnsafeConversions allows disabling the use of unsafe conversions between types that share
// the same memory layouts.
func (g *ConversionGenerator) WithoutUnsafeConversions() *ConversionGenerator {
	g.unsafeConversionArbitrator = nil
	return g
}

// WithMissingFieldsHandler allows setting a callback to decide what happens when converting
// from inVar.Type to outVar.Type, and when inVar.Type's member doesn't exist in outType.
// The callback can freely write into the snippet writer, at the spot in the auto-generated
// conversion function where the conversion code for that field should be.
// If the handler returns an error, the auto-generated private conversion function
// (i.e. autoConvert_a_X_To_b_Y) will still be generated, but not the public wrapper for it
// (i.e. Convert_a_X_To_b_Y).
// The handler can also choose to panic to stop the generation altogether, e.g. by calling
// klog.Fatalf.
// If this is not set, missing fields are silently ignored.
// Note that the snippet writer's context is that of the generator (in particular, it can use
// any namers defined by the generator).
func (g *ConversionGenerator) WithMissingFieldsHandler(handler func(inVar, outVar NamedVariable, member *types.Member, sw *generator.SnippetWriter) error) *ConversionGenerator {
	g.missingFieldsHandler = handler
	return g
}

// WithInconvertibleFieldsHandler allows setting a callback to decide what happens when converting
// from inVar.Type to outVar.Type, and when inVar.Type's inMember and outVar.Type's outMember are of
// inconvertible types.
// Same as for other handlers, the callback can freely write into the snippet writer, at the spot in
// the auto-generated conversion function where the conversion code for that field should be.
// If the handler returns an error, the auto-generated private conversion function
// (i.e. autoConvert_a_X_To_b_Y) will still be generated, but not the public wrapper for it
// (i.e. Convert_a_X_To_b_Y).
// The handler can also choose to panic to stop the generation altogether, e.g. by calling
// klog.Fatalf.
// If this is not set, missing fields are silently ignored.
// Note that the snippet writer's context is that of the generator (in particular, it can use
// any namers defined by the generator).
func (g *ConversionGenerator) WithInconvertibleFieldsHandler(handler func(inVar, outVar NamedVariable, inMember, outMember *types.Member, sw *generator.SnippetWriter) error) *ConversionGenerator {
	g.inconvertibleFieldsHandler = handler
	return g
}

// WithUnsupportedTypesHandler allows setting a callback to decide what happens when converting
// from inVar.Type to outVar.Type, and this generator has no idea how to handle that conversion.
// Same as for other handlers, the callback can freely write into the snippet writer, at the spot in
// the auto-generated conversion function where the conversion code for that type should be.
// If the handler returns an error, the auto-generated private conversion function
// (i.e. autoConvert_a_X_To_b_Y) will still be generated, but not the public wrapper for it
// (i.e. Convert_a_X_To_b_Y).
// The handler can also choose to panic to stop the generation altogether, e.g. by calling
// klog.Fatalf.
// If this is not set, missing fields are silently ignored.
// Note that the snippet writer's context is that of the generator (in particular, it can use
// any namers defined by the generator).
func (g *ConversionGenerator) WithUnsupportedTypesHandler(handler func(inVar, outVar NamedVariable, sw *generator.SnippetWriter) error) *ConversionGenerator {
	g.unsupportedTypesHandler = handler
	return g
}

// WithExternalConversionsHandler allows setting a callback to decide what happens when converting
// from inVar.Type to outVar.Type, but outVar.Type is in a different package than inVar.Type - and so
// this generator can't know where to find a conversion function for that.
// Same as for other handlers, the callback can freely write into the snippet writer, at the spot in
// the auto-generated conversion function where the conversion code for that type should be.
// If the handler returns an error, the auto-generated private conversion function
// (i.e. autoConvert_a_X_To_b_Y) will still be generated, but not the public wrapper for it
// (i.e. Convert_a_X_To_b_Y).
// The handler can also choose to panic to stop the generation altogether, e.g. by calling
// klog.Fatalf.
// If this is not set, missing fields are silently ignored.
// The boolean returned by the handler should indicate whether it has written code to handle
// the conversion.
// Note that the snippet writer's context is that of the generator (in particular, it can use
// any namers defined by the generator).
func (g *ConversionGenerator) WithExternalConversionsHandler(handler func(inVar, outVar NamedVariable, sw *generator.SnippetWriter) (bool, error)) *ConversionGenerator {
	g.externalConversionsHandler = handler
	return g
}

// The names of the namers used by ConversionGenerators.
// They're chosen to hopefully not conflict with namers from wrapper generators.
const (
	rawNamer                  = "ConversionGenerator_raw"
	publicImportTrackingNamer = "ConversionGenerator_publicIT"
)

// Namers returns the name system used by ConversionGenerators.
func (g *ConversionGenerator) Namers(context *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		rawNamer: namer.NewRawNamer(g.outputPackage, g.ImportTracker),
		publicImportTrackingNamer: &namerPlusImportTracking{
			delegate: ConversionNamer(),
			tracker:  g.ImportTracker,
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

// Filter filters the types this generator operates on.
func (g *ConversionGenerator) Filter(context *generator.Context, t *types.Type) bool {
	peerType := g.GetPeerTypeFor(context, t)
	return peerType != nil && g.convertibleOnlyWithinPackage(t, peerType)
}

// Imports returns the imports to add to generated files.
func (g *ConversionGenerator) Imports(context *generator.Context) (imports []string) {
	var importLines []string
	for _, singleImport := range g.ImportTracker.ImportLines() {
		if g.isOtherPackage(singleImport) {
			importLines = append(importLines, singleImport)
		}
	}
	return importLines
}

func (g *ConversionGenerator) isOtherPackage(pkg string) bool {
	if pkg == g.outputPackage {
		return false
	}
	if strings.HasSuffix(pkg, `"`+g.outputPackage+`"`) {
		return false
	}
	return true
}

// GenerateType processes the given type.
func (g *ConversionGenerator) GenerateType(context *generator.Context, t *types.Type, writer io.Writer) error {
	klog.V(5).Infof("generating for type %v", t)
	peerType := g.GetPeerTypeFor(context, t)
	sw := generator.NewSnippetWriter(writer, context, snippetDelimiter, snippetDelimiter)
	g.generateConversion(t, peerType, sw)
	g.generateConversion(peerType, t, sw)
	return sw.Error()

}

func (g *ConversionGenerator) generateConversion(inType, outType *types.Type, sw *generator.SnippetWriter) {
	// function signature
	sw.Do("func auto", nil)
	g.writeConversionFunctionSignature(inType, outType, sw, true)
	sw.Do(" {\n", nil)

	// body
	errors := g.generateFor(inType, outType, sw)

	// close function body
	sw.Do("return nil\n", nil)
	sw.Do("}\n\n", nil)

	if _, found := g.preexists(inType, outType); found {
		// there is a public manual Conversion method: use it.
		return
	}

	if len(errors) == 0 {
		// Emit a public conversion function.
		sw.Do("// "+conversionFunctionNameTemplate(publicImportTrackingNamer)+" is an autogenerated conversion function.\nfunc ", argsFromType(inType, outType))
		g.writeConversionFunctionSignature(inType, outType, sw, true)
		sw.Do(" {\nreturn auto", nil)
		g.writeConversionFunctionSignature(inType, outType, sw, false)
		sw.Do("\n}\n\n", nil)
		return
	}

	// there were errors generating the private conversion function
	klog.Errorf("Warning: could not find nor generate a final Conversion function for %v -> %v", inType, outType)
	klog.Errorf("  you need to add manual conversions:")
	for _, err := range errors {
		klog.Errorf("      - %v", err)
	}
}

// writeConversionFunctionSignature writes the signature of the conversion function from inType to outType
// into the given snippet writer.
// includeArgsTypes controls whether the arguments' types' will be included.
func (g *ConversionGenerator) writeConversionFunctionSignature(inType, outType *types.Type, sw *generator.SnippetWriter, includeArgsTypes bool) {
	args := argsFromType(inType, outType)
	sw.Do(conversionFunctionNameTemplate(publicImportTrackingNamer), args)
	sw.Do("(in", nil)
	if includeArgsTypes {
		sw.Do(" *$.inType|"+rawNamer+"$", args)
	}
	sw.Do(", out", nil)
	if includeArgsTypes {
		sw.Do(" *$.outType|"+rawNamer+"$", args)
	}
	for _, namedArgument := range g.manualConversionsTracker.additionalConversionArguments {
		sw.Do(fmt.Sprintf(", %s", namedArgument.Name), nil)
		if includeArgsTypes {
			sw.Do(" $.|"+rawNamer+"$", namedArgument.Type)
		}
	}
	sw.Do(")", nil)
	if includeArgsTypes {
		sw.Do(" error", nil)
	}
}

// we use the system of shadowing 'in' and 'out' so that the same code is valid
// at any nesting level. This makes the autogenerator easy to understand, and
// the compiler shouldn't care.
func (g *ConversionGenerator) generateFor(inType, outType *types.Type, sw *generator.SnippetWriter) []error {
	klog.V(5).Infof("generating %v -> %v", inType, outType)
	var f func(*types.Type, *types.Type, *generator.SnippetWriter) []error

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

	return f(inType, outType, sw)
}

func (g *ConversionGenerator) doBuiltin(inType, outType *types.Type, sw *generator.SnippetWriter) []error {
	if inType == outType {
		sw.Do("*out = *in\n", nil)
	} else {
		sw.Do("*out = $.|"+rawNamer+"$(*in)\n", outType)
	}
	return nil
}

func (g *ConversionGenerator) doMap(inType, outType *types.Type, sw *generator.SnippetWriter) (errors []error) {
	sw.Do("*out = make($.|"+rawNamer+"$, len(*in))\n", outType)
	if isDirectlyAssignable(inType.Key, outType.Key) {
		sw.Do("for key, val := range *in {\n", nil)
		if isDirectlyAssignable(inType.Elem, outType.Elem) {
			if inType.Key == outType.Key {
				sw.Do("(*out)[key] = ", nil)
			} else {
				sw.Do("(*out)[$.|"+rawNamer+"$(key)] = ", outType.Key)
			}
			if inType.Elem == outType.Elem {
				sw.Do("val\n", nil)
			} else {
				sw.Do("$.|"+rawNamer+"$(val)\n", outType.Elem)
			}
		} else {
			sw.Do("newVal := new($.|"+rawNamer+"$)\n", outType.Elem)

			manualOrInternal := false

			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				manualOrInternal = true
				sw.Do("if err := $.|"+rawNamer+"$(&val, newVal"+g.extraArgumentsString()+"); err != nil {\n", function)
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				manualOrInternal = true
				sw.Do("if err := "+conversionFunctionNameTemplate(publicImportTrackingNamer)+"(&val, newVal"+g.extraArgumentsString()+"); err != nil {\n",
					argsFromType(inType.Elem, outType.Elem))
			}

			if manualOrInternal {
				sw.Do("return err\n}\n", nil)
			} else if g.externalConversionsHandler == nil {
				klog.Warningf("%s's values of type %s require manual conversion to external type %s",
					inType.Name, inType.Elem, outType.Name)
			} else if _, err := g.externalConversionsHandler(NewNamedVariable("&val", inType.Elem), NewNamedVariable("newVal", outType.Elem), sw); err != nil {
				errors = append(errors, err)
			}

			if inType.Key == outType.Key {
				sw.Do("(*out)[key] = *newVal\n", nil)
			} else {
				sw.Do("(*out)[$.|"+rawNamer+"$(key)] = *newVal\n", outType.Key)
			}
		}
	} else {
		// TODO: Implement it when necessary.
		sw.Do("for range *in {\n", nil)
		sw.Do("// FIXME: Converting unassignable keys unsupported $.|"+rawNamer+"$\n", inType.Key)
	}
	sw.Do("}\n", nil)

	return
}

func (g *ConversionGenerator) doSlice(inType, outType *types.Type, sw *generator.SnippetWriter) (errors []error) {
	sw.Do("*out = make($.|"+rawNamer+"$, len(*in))\n", outType)
	if inType.Elem == outType.Elem && inType.Elem.Kind == types.Builtin {
		sw.Do("copy(*out, *in)\n", nil)
	} else {
		sw.Do("for i := range *in {\n", nil)
		if isDirectlyAssignable(inType.Elem, outType.Elem) {
			if inType.Elem == outType.Elem {
				sw.Do("(*out)[i] = (*in)[i]\n", nil)
			} else {
				sw.Do("(*out)[i] = $.|"+rawNamer+"$((*in)[i])\n", outType.Elem)
			}
		} else {
			manualOrInternal := false

			if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
				manualOrInternal = true
				sw.Do("if err := $.|"+rawNamer+"$(&(*in)[i], &(*out)[i]"+g.extraArgumentsString()+"); err != nil {\n", function)
			} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
				manualOrInternal = true
				sw.Do("if err := "+conversionFunctionNameTemplate(publicImportTrackingNamer)+"(&(*in)[i], &(*out)[i]"+g.extraArgumentsString()+"); err != nil {\n",
					argsFromType(inType.Elem, outType.Elem))
			}

			if manualOrInternal {
				sw.Do("return err\n}\n", nil)
			} else {
				conversionHandled := false
				var err error

				if g.externalConversionsHandler == nil {
					klog.Warningf("%s's items of type %s require manual conversion to external type %s",
						inType.Name, inType.Name, outType.Name)
				} else if conversionHandled, err = g.externalConversionsHandler(NewNamedVariable("&(*in)[i]", inType.Elem), NewNamedVariable("&(*out)[i]", outType.Elem), sw); err != nil {
					errors = append(errors, err)
				}

				if !conversionHandled {
					// so that the compiler doesn't barf
					sw.Do("_ = i\n", nil)
				}
			}
		}
		sw.Do("}\n", nil)
	}
	return
}

func (g *ConversionGenerator) doStruct(inType, outType *types.Type, sw *generator.SnippetWriter) (errors []error) {
	for _, inMember := range inType.Members {
		if g.optedOut(inMember) {
			// This field is excluded from conversion.
			sw.Do("// INFO: in."+inMember.Name+" opted out of conversion generation\n", nil)
			continue
		}
		outMember, found := findMember(outType, inMember.Name)
		if !found {
			// This field doesn't exist in the peer.
			if g.missingFieldsHandler == nil {
				klog.Warningf("%s.%s requires manual conversion: does not exist in peer-type %s", inType.Name, inMember.Name, outType.Name)
			} else if err := g.missingFieldsHandler(NewNamedVariable("in", inType), NewNamedVariable("out", outType), &inMember, sw); err != nil {
				errors = append(errors, err)
			}
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
		if g.useUnsafeConversion(inMemberType, outMemberType) {
			args = args.With("Pointer", types.Ref("unsafe", "Pointer"))
			switch inMemberType.Kind {
			case types.Pointer:
				sw.Do("out.$.name$ = ($.outType|"+rawNamer+"$)($.Pointer|"+rawNamer+"$(in.$.name$))\n", args)
				continue
			case types.Map:
				sw.Do("out.$.name$ = *(*$.outType|"+rawNamer+"$)($.Pointer|"+rawNamer+"$(&in.$.name$))\n", args)
				continue
			case types.Slice:
				sw.Do("out.$.name$ = *(*$.outType|"+rawNamer+"$)($.Pointer|"+rawNamer+"$(&in.$.name$))\n", args)
				continue
			}
		}

		// check based on the top level name, not the underlying names
		if function, ok := g.preexists(inMember.Type, outMember.Type); ok {
			if g.functionHasTag(function, "drop") {
				continue
			}
			if !g.functionHasTag(function, "copy-only") || !isFastConversion(inMemberType, outMemberType) {
				args["function"] = function
				sw.Do("if err := $.function|"+rawNamer+"$(&in.$.name$, &out.$.name$"+g.extraArgumentsString()+"); err != nil {\n", args)
				sw.Do("return err\n", nil)
				sw.Do("}\n", nil)
				continue
			}
			klog.V(5).Infof("Skipped function %s because it is copy-only and we can use direct assignment", function.Name)
		}

		// If we can't auto-convert, punt before we emit any code.
		if inMemberType.Kind != outMemberType.Kind {
			if g.inconvertibleFieldsHandler == nil {
				klog.Warningf("%s.%s requires manual conversion: inconvertible types: %s VS %s for %s.%s",
					inType.Name, inMember.Name, inMemberType, outMemberType, outType.Name, outMember.Name)
			} else if err := g.inconvertibleFieldsHandler(NewNamedVariable("in", inType), NewNamedVariable("out", outType), &inMember, &outMember, sw); err != nil {
				errors = append(errors, err)
			}
			continue
		}

		switch inMemberType.Kind {
		case types.Builtin:
			if inMemberType == outMemberType {
				sw.Do("out.$.name$ = in.$.name$\n", args)
			} else {
				sw.Do("out.$.name$ = $.outType|"+rawNamer+"$(in.$.name$)\n", args)
			}
		case types.Map, types.Slice, types.Pointer:
			if isDirectlyAssignable(inMemberType, outMemberType) {
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
			if isDirectlyAssignable(inMemberType, outMemberType) {
				sw.Do("out.$.name$ = in.$.name$\n", args)
				continue
			}
			if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
				sw.Do("if err := "+conversionFunctionNameTemplate(publicImportTrackingNamer)+"(&in.$.name$, &out.$.name$"+g.extraArgumentsString()+"); err != nil {\n", args)
				sw.Do("return err\n}\n", nil)
			} else {
				errors = g.callExternalConversionsHandlerForStructField(inType, outType, inMemberType, outMemberType, &inMember, &outMember, sw, errors)
			}
		case types.Alias:
			if isDirectlyAssignable(inMemberType, outMemberType) {
				if inMemberType == outMemberType {
					sw.Do("out.$.name$ = in.$.name$\n", args)
				} else {
					sw.Do("out.$.name$ = $.outType|"+rawNamer+"$(in.$.name$)\n", args)
				}
			} else {
				if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
					sw.Do("if err := "+conversionFunctionNameTemplate(publicImportTrackingNamer)+"(&in.$.name$, &out.$.name$"+g.extraArgumentsString()+"); err != nil {\n", args)
					sw.Do("return err\n}\n", nil)
				} else {
					errors = g.callExternalConversionsHandlerForStructField(inType, outType, inMemberType, outMemberType, &inMember, &outMember, sw, errors)
				}
			}
		default:
			if g.convertibleOnlyWithinPackage(inMemberType, outMemberType) {
				sw.Do("if err := "+conversionFunctionNameTemplate(publicImportTrackingNamer)+"(&in.$.name$, &out.$.name$"+g.extraArgumentsString()+"); err != nil {\n", args)
				sw.Do("return err\n}\n", nil)
			} else {
				errors = g.callExternalConversionsHandlerForStructField(inType, outType, inMemberType, outMemberType, &inMember, &outMember, sw, errors)
			}
		}
	}
	return
}

func (g *ConversionGenerator) callExternalConversionsHandlerForStructField(inType, outType, inMemberType, outMemberType *types.Type, inMember, outMember *types.Member, sw *generator.SnippetWriter, errors []error) []error {
	if g.externalConversionsHandler == nil {
		klog.Warningf("%s.%s requires manual conversion to external type %s.%s",
			inType.Name, inMember.Name, outType.Name, outMember.Name)
	} else {
		inVar := NewNamedVariable(fmt.Sprintf("&in.%s", inMember.Name), inMemberType)
		outVar := NewNamedVariable(fmt.Sprintf("&out.%s", outMember.Name), outMemberType)
		if _, err := g.externalConversionsHandler(inVar, outVar, sw); err != nil {
			errors = append(errors, err)
		}
	}
	return errors
}

func (g *ConversionGenerator) doPointer(inType, outType *types.Type, sw *generator.SnippetWriter) (errors []error) {
	sw.Do("*out = new($.Elem|"+rawNamer+"$)\n", outType)
	if isDirectlyAssignable(inType.Elem, outType.Elem) {
		if inType.Elem == outType.Elem {
			sw.Do("**out = **in\n", nil)
		} else {
			sw.Do("**out = $.|"+rawNamer+"$(**in)\n", outType.Elem)
		}
	} else {
		manualOrInternal := false

		if function, ok := g.preexists(inType.Elem, outType.Elem); ok {
			manualOrInternal = true
			sw.Do("if err := $.|"+rawNamer+"$(*in, *out"+g.extraArgumentsString()+"); err != nil {\n", function)
		} else if g.convertibleOnlyWithinPackage(inType.Elem, outType.Elem) {
			manualOrInternal = true
			sw.Do("if err := "+conversionFunctionNameTemplate(publicImportTrackingNamer)+"(*in, *out"+g.extraArgumentsString()+"); err != nil {\n", argsFromType(inType.Elem, outType.Elem))
		}

		if manualOrInternal {
			sw.Do("return err\n}\n", nil)
		} else if g.externalConversionsHandler == nil {
			klog.Warningf("%s's values of type %s require manual conversion to external type %s",
				inType.Name, inType.Elem, outType.Name)
		} else if _, err := g.externalConversionsHandler(NewNamedVariable("*in", inType), NewNamedVariable("*out", outType), sw); err != nil {
			errors = append(errors, err)
		}
	}
	return
}

func (g *ConversionGenerator) doAlias(inType, outType *types.Type, sw *generator.SnippetWriter) []error {
	// TODO: Add support for aliases.
	return g.doUnknown(inType, outType, sw)
}

func (g *ConversionGenerator) doUnknown(inType, outType *types.Type, sw *generator.SnippetWriter) []error {
	if g.unsupportedTypesHandler == nil {
		klog.Warningf("Don't know how to convert %s to %s", inType.Name, outType.Name)
	} else if err := g.unsupportedTypesHandler(NewNamedVariable("in", inType), NewNamedVariable("out", outType), sw); err != nil {
		return []error{err}
	}
	return nil
}

func (g *ConversionGenerator) extraArgumentsString() string {
	result := ""
	for _, namedArgument := range g.manualConversionsTracker.additionalConversionArguments {
		result += ", " + namedArgument.Name
	}
	return result
}

// GetPeerTypeFor returns the peer type for type t.
func (g *ConversionGenerator) GetPeerTypeFor(context *generator.Context, t *types.Type) *types.Type {
	if peerType, found := g.peerTypes[t.Name.Name]; found {
		return peerType
	}

	var peerType *types.Type
	for _, peerPkgPath := range g.peerPackages {
		peerPkg := context.Universe[peerPkgPath]
		if peerPkg != nil && peerPkg.Has(t.Name.Name) {
			peerType = peerPkg.Types[t.Name.Name]
			break
		}
	}

	g.peerTypes[t.Name.Name] = peerType
	return peerType
}

func (g *ConversionGenerator) convertibleOnlyWithinPackage(inType, outType *types.Type) bool {
	var t, other *types.Type
	if inType.Name.Package == g.typesPackage {
		t, other = inType, outType
	} else {
		t, other = outType, inType
	}

	if t.Name.Package != g.typesPackage {
		return false
	}

	if g.optedOut(t) {
		klog.V(5).Infof("type %v requests no conversion generation, skipping", t)
		return false
	}

	return t.Kind == types.Struct && // TODO: Consider generating functions for other kinds too
		!namer.IsPrivateGoName(other.Name.Name) // filter out private types
}

// optedOut returns true iff type (or member) t has a comment tag of the form "<tag-name>=false"
// indicating that it's opting out of the conversion generation.
func (g *ConversionGenerator) optedOut(t interface{}) bool {
	var commentLines []string
	switch in := t.(type) {
	case *types.Type:
		commentLines = in.CommentLines
	case types.Member:
		commentLines = in.CommentLines
	default:
		klog.Fatalf("don't know how to extract comment lines from %#v", t)
	}

	tagVals := g.extractTag(commentLines)
	if len(tagVals) > 0 {
		if tagVals[0] != "false" {
			klog.Fatalf(fmt.Sprintf("Type %v: unsupported %s value: %q", t, g.tagName, tagVals[0]))
		}
		return true
	}
	return false
}

func (g *ConversionGenerator) extractTag(comments []string) []string {
	if g.tagName == "" {
		return nil
	}
	return types.ExtractCommentTags("+", comments)[g.tagName]
}

func (g *ConversionGenerator) functionHasTag(function *types.Type, tagValue string) bool {
	return functionHasTag(function, g.functionTagName, tagValue)
}

func (g *ConversionGenerator) preexists(inType, outType *types.Type) (*types.Type, bool) {
	return g.manualConversionsTracker.preexists(inType, outType)
}

func (g *ConversionGenerator) useUnsafeConversion(t1, t2 *types.Type) bool {
	return g.unsafeConversionArbitrator != nil &&
		g.unsafeConversionArbitrator.canUseUnsafeConversion(t1, t2)
}

func (g *ConversionGenerator) ManualConversions() map[ConversionPair]*types.Type {
	return g.manualConversionsTracker.conversionFunctions
}
