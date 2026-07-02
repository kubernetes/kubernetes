/*
Copyright 2024 The Kubernetes Authors.

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

package main

import (
	"cmp"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/code-generator/pkg/apidefinitions"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// These are the comment tags that carry parameters for validation generation.
const (
	// Defines which types to generate validation for.  There are two places
	// this can be used:
	//   Per-package:
	//     * "*": generate validation for all types in this package
	//	   * "TypesWithField=FooBar": generate validation for all types with a
	//	     field named "FooBar"
	//   Per-type:
	//	   * "true": generate validation for this type
	//	   * "false": do not generate validation for this type
	mainTagName = "k8s:validation-gen"
	// Defines the type of the scheme used to register validations. Defaults to
	// "k8s.io/apimachinery/pkg.runtime.Scheme", but can be set to another type
	// (e.g. in tests), or set to "nil" to disable scheme registration for this
	// package.
	schemeRegistryTagName = "k8s:validation-gen-scheme-registry"
	// If set, generate go test files for test fixtures.  Supported values: "validateFalse".
	testFixtureTagName = "k8s:validation-gen-test-fixture"

	// name of the subresource that this type represents and can validate declaratively.
	isSubresourceTagName = "k8s:isSubresource"

	// name of a subresource that this type can validate declaratively, tag may be
	// repeated to support multiple subresources.
	supportsSubresourceTagName = "k8s:supportsSubresource"

	// if set on a package, generates declarative coverage test targets even if it's not a versioned API package.
	generateTestTargetsTagName = "k8s:validation-gen-test-targets"
)

var (
	runtimePkg   = "k8s.io/apimachinery/pkg/runtime"
	schemeType   = types.Name{Package: runtimePkg, Name: "Scheme"}
	metav1Pkg    = "k8s.io/apimachinery/pkg/apis/meta/v1"
	listMetaType = types.Name{Package: metav1Pkg, Name: "ListMeta"}
)

// extractAndParseTag extracts all the values for a given tag, according to the
// tag grammar.
func extractAndParseTag(tagName string, comments []string) ([]codetags.Tag, error) {
	extracted := codetags.Extract("+", comments)
	var tags []codetags.Tag
	for key, lines := range extracted {
		if key != tagName {
			continue
		}
		t, err := codetags.ParseAll(lines)
		if err != nil {
			return nil, fmt.Errorf("failed to parse tags: %w: %s", err, lines)
		}
		tags = append(tags, t...)
	}
	return tags, nil
}

// validationTypeMatch returns the +k8s:validation-gen tag values for pkg,
// or false if validation-gen should not run.
func validationTypeMatch(pkg *types.Package, idOpts []apidefinitions.Option) ([]string, bool) {
	info, err := apidefinitions.Identify(pkg, apidefinitions.Validation, idOpts...)
	if err != nil {
		klog.Fatal(err)
	}
	if !info.ShouldGenerate() {
		return nil, false
	}
	return info.TypeFilters(), true
}

// TODO: this can just accept a single bool
func checkMainTag(comments []string, require ...string) bool {
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{mainTagName}, comments)
	if err != nil {
		klog.Fatalf("Failed to extract tags: %v", err)
	}
	values, found := tags[mainTagName]
	if !found {
		return false
	}

	if len(require) == 0 {
		return len(values) == 1 && values[0].Value == ""
	}

	valueStrings := make([]string, len(values))
	for i, tag := range values {
		valueStrings[i] = tag.Value
	}

	return reflect.DeepEqual(valueStrings, require)
}

func schemeRegistryTag(pkg *types.Package) (types.Name, bool) {
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{schemeRegistryTagName}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract scheme registry tags: %v", err)
	}
	values, found := tags[schemeRegistryTagName]
	if !found || len(values) == 0 {
		return schemeType, true // default
	}
	if len(values) > 1 {
		panic(fmt.Sprintf("Package %q contains more than one usage of %q", pkg.Path, schemeRegistryTagName))
	}
	val := values[0].Value
	if val == "nil" {
		// no registration wanted for this package
		return types.Name{}, false
	}
	return types.ParseFullyQualifiedName(val), true
}

func isSubresourceTag(t *types.Type) (string, bool) {
	var comments []string
	comments = append(comments, t.SecondClosestCommentLines...)
	comments = append(comments, t.CommentLines...)
	tags, err := extractAndParseTag(isSubresourceTagName, comments)
	if err != nil {
		klog.Fatalf("Failed to extract isSubresource tags: %v", err)
	}
	if len(tags) == 0 {
		return "", false
	}
	if len(tags) > 1 {
		panic(fmt.Sprintf("Type %q contains more than one usage of %q", t.Name.String(), isSubresourceTagName))
	}
	return tags[0].Value, true
}

func supportedSubresourceTags(t *types.Type) sets.Set[string] {
	var comments []string
	comments = append(comments, t.SecondClosestCommentLines...)
	comments = append(comments, t.CommentLines...)
	tags, err := extractAndParseTag(supportsSubresourceTagName, comments)
	if err != nil {
		klog.Fatalf("Failed to extract supportedSubresource tags: %v", err)
	}
	if len(tags) == 0 {
		return sets.New[string]()
	}
	subresources := sets.New[string]()
	for _, tag := range tags {
		subresources.Insert(tag.Value)
	}
	return subresources
}

var testFixtureTagValues = sets.New("validateFalse")

func testFixtureTag(pkg *types.Package) sets.Set[string] {
	result := sets.New[string]()
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{testFixtureTagName}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract test fixture tags: %v", err)
	}
	values, found := tags[testFixtureTagName]
	if !found {
		return result
	}

	for _, tag := range values {
		if !testFixtureTagValues.Has(tag.Value) {
			panic(fmt.Sprintf("Package %q: %s must be one of '%s', but got: %s", pkg.Path, testFixtureTagName, testFixtureTagValues.UnsortedList(), tag.Value))
		}
		result.Insert(tag.Value)
	}
	return result
}

func generateTestTargetsTag(pkg *types.Package) bool {
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{generateTestTargetsTagName}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract %s tags: %v", generateTestTargetsTagName, err)
	}
	_, found := tags[generateTestTargetsTagName]
	return found
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public":             namer.NewPublicNamer(1),
		"raw":                namer.NewRawNamer("", nil),
		"objectvalidationfn": validationFnNamer(),
		"private":            namer.NewPrivateNamer(0),
		"name":               namer.NewPublicNamer(0),
	}
}

func validationFnNamer() *namer.NameStrategy {
	return &namer.NameStrategy{
		Prefix: "Validate_",
		Join: func(pre string, in []string, post string) string {
			return pre + strings.Join(in, "_") + post
		},
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

func GetTargets(context *generator.Context, args *Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, gengo.StdBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	var idOpts []apidefinitions.Option
	if len(args.LintRules) > 0 {
		idOpts = append(idOpts, apidefinitions.WithLintRules(args.LintRules...))
	}

	var targetList []generator.Target

	// First load other "input" packages.  We do this as a single call because
	// it is MUCH faster.
	inputPkgs := make([]string, 0, len(context.Inputs))
	pkgToInput := map[string]string{}
	inputToPkg := map[string]string{} // reverse of pkgToInput
	for _, input := range context.Inputs {
		klog.V(4).Infof("considering pkg %q", input)
		pkg := context.Universe[input]

		info, err := apidefinitions.Identify(pkg, apidefinitions.Validation, idOpts...)
		if err != nil {
			klog.Fatal(err)
		}
		if !info.ShouldGenerate() {
			continue
		}

		// +k8s:validation-gen-input may direct the generator at types in
		// a different package than the one where validators will be emitted.
		inputPath := info.ExternalTypes()
		if inputPath != pkg.Path {
			klog.V(4).Infof("  input pkg %v", inputPath)
			inputPkgs = append(inputPkgs, inputPath)
			pkgToInput[input] = inputPath
			inputToPkg[inputPath] = input
		} else {
			pkgToInput[input] = input
			inputToPkg[input] = input
		}
	}

	// Make sure explicit extra-packages are added.
	var readOnlyPkgs []string
	for _, pkg := range args.ReadOnlyPkgs {
		// In case someone specifies an extra as a path into vendor, convert
		// it to its "real" package path.
		if i := strings.Index(pkg, "/vendor/"); i != -1 {
			pkg = pkg[i+len("/vendor/"):]
		}
		readOnlyPkgs = append(readOnlyPkgs, pkg)
	}
	if expanded, err := context.FindPackages(readOnlyPkgs...); err != nil {
		klog.Fatalf("cannot find extra packages: %v", err)
	} else {
		readOnlyPkgs = expanded // now in fully canonical form
	}
	for _, extra := range readOnlyPkgs {
		inputPkgs = append(inputPkgs, extra)
		pkgToInput[extra] = extra
		// Don't let a read-only package override a generation mapping.
		if _, ok := inputToPkg[extra]; !ok {
			inputToPkg[extra] = extra
		}
	}

	if len(inputPkgs) > 0 {
		if _, err := context.LoadPackages(inputPkgs...); err != nil {
			klog.Fatalf("cannot load packages: %v", err)
		}
	}
	// update context.Order to the latest context.Universe
	orderer := namer.Orderer{Namer: namer.NewPublicNamer(1)}
	context.Order = orderer.OrderUniverse(context.Universe)

	// Initialize all validator plugins exactly once.
	validator := validators.InitGlobalValidator(context, inputToPkg)

	// Create a type discoverer for all types of all inputs.
	td := NewTypeDiscoverer(validator, inputToPkg)
	if err := td.Init(context); err != nil {
		klog.Fatalf("Error discovering constants: %v", err)
	}

	// Create a linter to collect errors as we go.
	linter := newLinter(lintRules(validator)...)

	// groupKindReports accumulates Reports across every input, keyed by
	// GroupKind so testTargets emits exactly one SimpleTarget per Kind.
	groupKindReports := map[schema.GroupKind][]*report{}

	// Build a cache of type->callNode for every type we need.
	for _, input := range context.Inputs {
		klog.V(2).InfoS("processing", "pkg", input)

		pkg := context.Universe[input]

		schemeRegistry, registerThisPkg := schemeRegistryTag(pkg)

		criteria, found := validationTypeMatch(pkg, idOpts)
		if !found {
			klog.V(2).InfoS("  did not find required tag", "tag", mainTagName)
			continue
		}
		if len(criteria) == 1 && criteria[0] == "" {
			klog.Fatalf("%s: found package tag %q with no value", input, mainTagName)
		}
		for _, crit := range criteria {
			if crit == "*" {
				continue
			}
			if val, found := strings.CutPrefix(crit, "TypesWithField="); found {
				if val == "" {
					klog.Fatalf("%s: found package tag \"%s=%s\" with empty value", input, mainTagName, crit)
				}
				continue
			}
			klog.Fatalf("%s: unknown value for package tag %q: %q", input, mainTagName, crit)
		}
		shouldCreateObjectValidationFn := func(t *types.Type) bool {
			// Never generate validation for unexported types.
			if namer.IsPrivateGoName(t.Name.Name) {
				return false
			}
			// opt-out
			if checkMainTag(t.CommentLines, "false") {
				return false
			}
			if checkMainTag(t.SecondClosestCommentLines, "false") {
				return false
			}
			// opt-in
			if checkMainTag(t.CommentLines, "true") {
				return true
			}
			if checkMainTag(t.SecondClosestCommentLines, "true") {
				return true
			}

			// skip types that embed metav1.ListMeta
			if t.Kind == types.Struct {
				for _, member := range t.Members {
					if member.Embedded && member.Type.Name == listMetaType {
						return false
					}
				}
			}

			// all types
			for _, v := range criteria {
				if v == "*" {
					return true
				}
				if field, found := strings.CutPrefix(v, "TypesWithField="); found {
					if isTypeWithField(t, field) {
						return true
					}
				}
			}
			return false
		}

		// Find the right input pkg, which might not be this one.
		inputPath := pkgToInput[input]
		// typesPkg is where the types that need validation are defined.
		// Sometimes it is different from pkg. For example, kubernetes core/v1
		// types are defined in k8s.io/api/core/v1, while the pkg which holds
		// defaulter code is at k/k/pkg/api/v1.
		typesPkg := context.Universe[inputPath]

		// Figure out which types we should be considering further.
		var rootTypes []*types.Type
		for _, t := range typesPkg.Types {
			if shouldCreateObjectValidationFn(t) {
				rootTypes = append(rootTypes, t)
			} else {
				klog.V(6).InfoS("skipping type", "type", t)
			}
		}
		// Deterministic ordering helps in logs and debugging.
		slices.SortFunc(rootTypes, func(a, b *types.Type) int {
			return cmp.Compare(a.Name.String(), b.Name.String())
		})

		for _, t := range rootTypes {
			klog.V(3).InfoS("pre-processing", "type", t)
			if err := td.DiscoverType(t); err != nil {
				klog.Fatalf("failed to generate validations: %v", err)
			}
		}

		extracted := codetags.Extract("+", pkg.Comments)
		if _, ok := extracted["k8s:validation-gen-nolint"]; !ok {
			for _, t := range rootTypes {
				klog.V(3).InfoS("linting root-type", "type", t)
				if err := linter.lintType(t); err != nil {
					klog.Fatalf("failed to lint type %q: %v", t.Name, err)
				}
			}
		}

		targetList = append(targetList,
			&generator.SimpleTarget{
				PkgName:       pkg.Name,
				PkgPath:       pkg.Path,
				PkgDir:        pkg.Dir, // output pkg is the same as the input
				HeaderComment: boilerplate,

				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == typesPkg.Path
				},

				GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
					generators = []generator.Generator{
						NewGenValidations(args.OutputFile, pkg.Path, rootTypes, td, inputToPkg, schemeRegistry, registerThisPkg),
					}
					testFixtureTags := testFixtureTag(pkg)
					if testFixtureTags.Len() > 0 {
						if !strings.HasSuffix(args.OutputFile, ".go") {
							panic(fmt.Sprintf("%s requires that output file have .go suffix", testFixtureTagName))
						}
						filename := args.OutputFile[0:len(args.OutputFile)-3] + "_test.go"
						generators = append(generators, FixtureTests(filename, testFixtureTags))
					}
					if generateTestTargetsTag(pkg) {
						var reports []*report
						for _, t := range rootTypes {
							rules := collectRules(td.typeNodes[t])
							if len(rules) == 0 {
								continue
							}
							reports = append(reports, &report{
								Group:   pkg.Path,
								Version: pkg.Name,
								Kind:    t.Name.Name,
								Rules:   rules,
							})
						}
						if len(reports) > 0 {
							filename := args.OutputFile[0:len(args.OutputFile)-3] + "_coverage_test.go"
							generators = append(generators, newCoverageTestGen(pkg.Path, filename, reports, true, nil))
						}
					}
					return generators
				},
			})

		// Accumulate per-Kind rules; testTargets emits after the loop.
		if args.TestOutputRoot != "" {
			collectReports(typesPkg, rootTypes, td, groupKindReports)
		}
	}

	// All inputs processed: fail if a ValidateCustom_* function lacks a tag.
	if err := validators.VerifyCustomValidationsHaveTags(); err != nil {
		klog.Fatalf("%v", err)
	}

	// Emit per-Kind coverage test targets. No-op when --test-output-root is empty.
	allowlist, err := loadAllowlist(args.TestAllowlist)
	if err != nil {
		klog.Fatalf("loading allowlist: %v", err)
	}
	targetList = append(targetList, testTargets(args.TestOutputRoot, args.TestOutputFilePrefix, groupKindReports, allowlist, boilerplate)...)

	if len(linter.lintErrors) > 0 {
		buf := strings.Builder{}

		for t, errs := range linter.lintErrors {
			buf.WriteString(fmt.Sprintf("  type %v:\n", t))
			for _, err := range errs {
				buf.WriteString(fmt.Sprintf("    %s\n", err.Error()))
			}
		}
		klog.Fatalf("lint failed:\n%s", buf.String())
	}
	return targetList
}

func isTypeWithField(t *types.Type, fieldName string) bool {
	if t.Kind == types.Struct {
		for _, field := range t.Members {
			if field.Name == fieldName {
				return true
			}
		}
	}
	return false
}
