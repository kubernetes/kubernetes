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

package generators

import (
	"cmp"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/args"
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
// Each is qualified by the configured tag prefix, e.g. "+k8s:validation-gen"
// for prefix "k8s:".
const (
	// Defines which types to generate validation for.  There are two places
	// this can be used:
	//   Per-package:
	//     * "*": generate validation for all types in this package
	//	   * "TypesWithField=FooBar": generate validation for all types with a
	//	     field named "FooBar"
	//     * "TypesWithSuffix=FooBar": generate validation for all types whose
	//       name ends with "FooBar"
	//   Per-type:
	//	   * "true": generate validation for this type
	//	   * "false": do not generate validation for this type
	mainTagName = "validation-gen"
	// Defines the type of the scheme used to register validations. Defaults to
	// "k8s.io/apimachinery/pkg.runtime.Scheme", but can be set to another type
	// (e.g. in tests), or set to "nil" to disable scheme registration for this
	// package.
	schemeRegistryTagName = "validation-gen-scheme-registry"
	// Defines the deep-equal function used wherever generated code needs
	// value equality.  The value names a function which must be generic
	// over a single type parameter, e.g:
	//     func Equal[T any](a, b T) bool
	// T is instantiated with whatever a call site compares, commonly a
	// pointer to a struct, slice, or map, so the function must handle any T.
	// Directly comparable types are compared with == and never reach it.
	// An unqualified name refers to the package being generated into.
	// Defaults to "k8s.io/apimachinery/pkg/api/validate.SemanticDeepEqual".
	deepEqualFuncTagName = "validation-gen-deep-equal-func"
	// If set, generate go test files for test fixtures.  Supported values: "validateFalse".
	testFixtureTagName = "validation-gen-test-fixture"

	// name of the subresource that this type represents and can validate declaratively.
	isSubresourceTagName = "isSubresource"

	// name of a subresource that this type can validate declaratively, tag may be
	// repeated to support multiple subresources.
	supportsSubresourceTagName = "supportsSubresource"

	// if set on a package, generates declarative coverage test targets even if it's not a versioned API package.
	generateTestTargetsTagName = "validation-gen-test-targets"
	// if set on a package or type, disables validation-gen's lint rules for it.
	noLintTagName = "validation-gen-nolint"
)

var (
	runtimePkg           = "k8s.io/apimachinery/pkg/runtime"
	schemeType           = types.Name{Package: runtimePkg, Name: "Scheme"}
	defaultDeepEqualFunc = types.Name{Package: "k8s.io/apimachinery/pkg/api/validate", Name: "SemanticDeepEqual"}
	metav1Pkg            = "k8s.io/apimachinery/pkg/apis/meta/v1"
	listMetaType         = types.Name{Package: metav1Pkg, Name: "ListMeta"}
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

// validationTypeMatch returns the +<prefix>validation-gen tag values for pkg,
// or false if validation-gen should not run.
func validationTypeMatch(pkg *types.Package, spec apidefinitions.Spec, idOpts []apidefinitions.Option) ([]string, bool) {
	info, err := apidefinitions.Identify(pkg, spec, idOpts...)
	if err != nil {
		klog.Fatal(err)
	}
	if !info.ShouldGenerate() {
		return nil, false
	}
	return info.TypeFilters(), true
}

// TODO: this can just accept a single bool
func checkMainTag(prefix string, comments []string, require ...string) bool {
	mainTag := prefix + mainTagName
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{mainTag}, comments)
	if err != nil {
		klog.Fatalf("Failed to extract tags: %v", err)
	}
	values, found := tags[mainTag]
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

func schemeRegistryTag(prefix string, pkg *types.Package) (types.Name, bool) {
	schemeRegistryTag := prefix + schemeRegistryTagName
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{schemeRegistryTag}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract scheme registry tags: %v", err)
	}
	values, found := tags[schemeRegistryTag]
	if !found || len(values) == 0 {
		return schemeType, true // default
	}
	if len(values) > 1 {
		panic(fmt.Sprintf("Package %q contains more than one usage of %q", pkg.Path, schemeRegistryTag))
	}
	val := values[0].Value
	if val == "nil" {
		// no registration wanted for this package
		return types.Name{}, false
	}
	return types.ParseFullyQualifiedName(val), true
}

// registerScheme reports whether pkg registers its validations with a scheme.
func registerScheme(prefix string, pkg *types.Package) bool {
	_, ok := schemeRegistryTag(prefix, pkg)
	return ok
}

func deepEqualFuncTag(prefix string, pkg *types.Package) types.Name {
	deepEqualFuncTag := prefix + deepEqualFuncTagName
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{deepEqualFuncTag}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract deep equal func tags: %v", err)
	}
	values, found := tags[deepEqualFuncTag]
	if !found || len(values) == 0 {
		return defaultDeepEqualFunc
	}
	if len(values) > 1 {
		panic(fmt.Sprintf("Package %q contains more than one usage of %q", pkg.Path, deepEqualFuncTag))
	}
	val := values[0].Value
	if val == "" {
		return defaultDeepEqualFunc
	}
	return types.ParseFullyQualifiedName(val)
}

func isSubresourceTag(prefix string, t *types.Type) (string, bool) {
	var comments []string
	comments = append(comments, t.SecondClosestCommentLines...)
	comments = append(comments, t.CommentLines...)
	tags, err := extractAndParseTag(prefix+isSubresourceTagName, comments)
	if err != nil {
		klog.Fatalf("Failed to extract isSubresource tags: %v", err)
	}
	if len(tags) == 0 {
		return "", false
	}
	if len(tags) > 1 {
		panic(fmt.Sprintf("Type %q contains more than one usage of %q", t.Name.String(), prefix+isSubresourceTagName))
	}
	return tags[0].Value, true
}

func supportedSubresourceTags(prefix string, t *types.Type) sets.Set[string] {
	var comments []string
	comments = append(comments, t.SecondClosestCommentLines...)
	comments = append(comments, t.CommentLines...)
	tags, err := extractAndParseTag(prefix+supportsSubresourceTagName, comments)
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

func testFixtureTag(prefix string, pkg *types.Package) sets.Set[string] {
	result := sets.New[string]()
	testFixtureTag := prefix + testFixtureTagName
	// TODO: convert to extractAndParseTag() and update all callers to use quoted values
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{testFixtureTag}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract test fixture tags: %v", err)
	}
	values, found := tags[testFixtureTag]
	if !found {
		return result
	}

	for _, tag := range values {
		if !testFixtureTagValues.Has(tag.Value) {
			panic(fmt.Sprintf("Package %q: %s must be one of '%s', but got: %s", pkg.Path, testFixtureTag, testFixtureTagValues.UnsortedList(), tag.Value))
		}
		result.Insert(tag.Value)
	}
	return result
}

func generateTestTargetsTag(prefix string, pkg *types.Package) bool {
	generateTestTargetsTag := prefix + generateTestTargetsTagName
	tags, err := gengo.ExtractFunctionStyleCommentTags("+", []string{generateTestTargetsTag}, pkg.Comments)
	if err != nil {
		klog.Fatalf("Failed to extract %s tags: %v", generateTestTargetsTag, err)
	}
	_, found := tags[generateTestTargetsTag]
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

func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, gengo.StdBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	prefix := args.TagPrefix
	spec := apidefinitions.ValidationSpec(prefix)

	var idOpts []apidefinitions.Option
	if len(args.LintRules) > 0 {
		idOpts = append(idOpts, apidefinitions.WithLintRules(args.LintRules...))
	}

	var targetList []generator.Target

	// First load other "input" packages.  We do this as a single call because
	// it is MUCH faster.
	inputPkgs := make([]string, 0, len(context.Inputs))
	pkgToInput := map[string]string{}
	inputToCanonicalPkg := map[string]string{} // types package -> the output package cross-package references resolve to
	for _, input := range context.Inputs {
		klog.V(4).Infof("considering pkg %q", input)
		pkg := context.Universe[input]

		info, err := apidefinitions.Identify(pkg, spec, idOpts...)
		if err != nil {
			klog.Fatal(err)
		}
		if !info.ShouldGenerate() {
			continue
		}

		// +<prefix>validation-gen-input may direct the generator at types in
		// a different package than the one where validators will be emitted.
		inputPath := info.ExternalTypes()
		pkgToInput[input] = inputPath
		if inputPath != pkg.Path {
			klog.V(4).Infof("  input pkg %v", inputPath)
			inputPkgs = append(inputPkgs, inputPath)
		}
		// An input's validation may be generated into more than one package. One
		// is canonical -- the package cross-package references resolve to. The
		// registering package is canonical; if none registers, the first one seen
		// wins. At most one package may register.
		if prev, ok := inputToCanonicalPkg[inputPath]; !ok {
			inputToCanonicalPkg[inputPath] = input
		} else if registerScheme(prefix, pkg) {
			if registerScheme(prefix, context.Universe[prev]) {
				klog.Fatalf("input %q is generated into two registering packages (%q, %q); mark one +%s=nil", inputPath, prev, input, prefix+schemeRegistryTagName)
			}
			inputToCanonicalPkg[inputPath] = input // a registering package displaces a non-registering one
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
		if _, ok := inputToCanonicalPkg[extra]; !ok {
			inputToCanonicalPkg[extra] = extra
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
	validator := validators.InitGlobalValidator(context, inputToCanonicalPkg, prefix)

	// Create a type discoverer for all types of all inputs.
	td := NewTypeDiscoverer(validator, inputToCanonicalPkg, prefix)
	if err := td.Init(context); err != nil {
		klog.Fatalf("Error discovering constants: %v", err)
	}

	// Create a linter to collect errors as we go.
	linter := newLinter(prefix, lintRules(validator, prefix)...)

	// groupKindReports accumulates Reports across every input, keyed by
	// GroupKind so testTargets emits exactly one SimpleTarget per Kind.
	groupKindReports := map[schema.GroupKind][]*report{}

	// Build a cache of type->callNode for every type we need.
	for _, input := range context.Inputs {
		klog.V(2).InfoS("processing", "pkg", input)

		pkg := context.Universe[input]

		schemeRegistry, registerThisPkg := schemeRegistryTag(prefix, pkg)
		deepEqualFunc := deepEqualFuncTag(prefix, pkg)

		criteria, found := validationTypeMatch(pkg, spec, idOpts)
		if !found {
			klog.V(2).InfoS("  did not find required tag", "tag", spec.ActivationTag)
			continue
		}
		if len(criteria) == 1 && criteria[0] == "" {
			klog.Fatalf("%s: found package tag %q with no value", input, spec.ActivationTag)
		}
		for _, crit := range criteria {
			if crit == "*" {
				continue
			}
			if val, found := strings.CutPrefix(crit, "TypesWithField="); found {
				if val == "" {
					klog.Fatalf("%s: found package tag \"%s=%s\" with empty value", input, spec.ActivationTag, crit)
				}
				continue
			}
			if val, found := strings.CutPrefix(crit, "TypesWithSuffix="); found {
				if val == "" {
					klog.Fatalf("%s: found package tag \"%s=%s\" with empty value", input, spec.ActivationTag, crit)
				}
				continue
			}
			klog.Fatalf("%s: unknown value for package tag %q: %q", input, spec.ActivationTag, crit)
		}
		shouldCreateObjectValidationFn := func(t *types.Type) bool {
			// Never generate validation for unexported types.
			if namer.IsPrivateGoName(t.Name.Name) {
				return false
			}
			// opt-out
			if checkMainTag(prefix, t.CommentLines, "false") {
				return false
			}
			if checkMainTag(prefix, t.SecondClosestCommentLines, "false") {
				return false
			}
			// opt-in
			if checkMainTag(prefix, t.CommentLines, "true") {
				return true
			}
			if checkMainTag(prefix, t.SecondClosestCommentLines, "true") {
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
				if field, found := strings.CutPrefix(v, "TypesWithSuffix="); found {
					if isTypeWithSuffix(t, field) {
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
		if _, ok := extracted[prefix+noLintTagName]; !ok {
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
						NewGenValidations(args.OutputFile, pkg.Path, typesPkg.Path, rootTypes, td, inputToCanonicalPkg, schemeRegistry, registerThisPkg, deepEqualFunc, prefix),
					}
					testFixtureTags := testFixtureTag(prefix, pkg)
					if testFixtureTags.Len() > 0 {
						if !strings.HasSuffix(args.OutputFile, ".go") {
							panic(fmt.Sprintf("%s requires that output file have .go suffix", prefix+testFixtureTagName))
						}
						filename := args.OutputFile[0:len(args.OutputFile)-3] + "_test.go"
						generators = append(generators, FixtureTests(filename, testFixtureTags))
					}
					if generateTestTargetsTag(prefix, pkg) {
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
		// Only the registering package contributes coverage; a non-registering
		// package generated from the same input has identical rules, so counting
		// it too would double-list the version for the Kind.
		if args.TestOutputRoot != "" && registerThisPkg {
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
		buf := &strings.Builder{}

		for t, errs := range linter.lintErrors {
			fmt.Fprintf(buf, "  type %v:\n", t)
			for _, err := range errs {
				fmt.Fprintf(buf, "    %s\n", err.Error())
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

func isTypeWithSuffix(t *types.Type, suffix string) bool {
	return strings.HasSuffix(t.Name.Name, suffix)
}
