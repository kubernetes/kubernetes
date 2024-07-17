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
	"fmt"
	"path"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/args"
	"k8s.io/code-generator/cmd/validation-gen/generators/validators"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// These are the comment tags that carry parameters for validation generation.
const (
	tagName         = "k8s:validation-gen"
	inputTagName    = "k8s:validation-gen-input"
	enabledTagName  = "k8s:validation-gen-enabled-tags"
	disabledTagName = "k8s:validation-gen-disabled-tags"
)

func extractTag(comments []string) []string {
	return gengo.ExtractCommentTags("+", comments)[tagName]
}

func extractInputTag(comments []string) []string {
	return gengo.ExtractCommentTags("+", comments)[inputTagName]
}

func extractFiltersTags(comments []string) (enabled, disabled []string) {
	return gengo.ExtractCommentTags("+", comments)[enabledTagName],
		gengo.ExtractCommentTags("+", comments)[disabledTagName]
}

func checkTag(comments []string, require ...string) bool {
	values := gengo.ExtractCommentTags("+", comments)[tagName]
	if len(require) == 0 {
		return len(values) == 1 && values[0] == ""
	}
	return reflect.DeepEqual(values, require)
}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public":             namer.NewPublicNamer(1),
		"raw":                namer.NewRawNamer("", nil),
		"objectvalidationfn": validationFnNamer(),
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

	var targets []generator.Target

	// First load other "input" packages.  We do this as a single call because
	// it is MUCH faster.
	inputPkgs := make([]string, 0, len(context.Inputs))
	pkgToInput := map[string]string{}
	for _, i := range context.Inputs {
		klog.V(5).Infof("considering pkg %q", i)

		pkg := context.Universe[i]

		// if the types are not in the same package where the validation functions are to be visited
		inputTags := extractInputTag(pkg.Comments)
		if len(inputTags) > 1 {
			panic(fmt.Sprintf("there may only be one input tag, got %#v", inputTags))
		}
		if len(inputTags) == 1 {
			inputPath := inputTags[0]
			if strings.HasPrefix(inputPath, "./") || strings.HasPrefix(inputPath, "../") {
				// this is a relative dir, which will not work under gomodules.
				// join with the local package path, but warn
				klog.Warningf("relative path %s=%s will not work under gomodule mode; use full package path (as used by 'import') instead", inputTagName, inputPath)
				inputPath = path.Join(pkg.Path, inputTags[0])
			}

			klog.V(5).Infof("  input pkg %v", inputPath)
			inputPkgs = append(inputPkgs, inputPath)
			pkgToInput[i] = inputPath
		} else {
			pkgToInput[i] = i
		}
	}

	// Make sure explicit peer-packages are added.
	var peerPkgs []string
	for _, pkg := range args.ExtraPeerDirs {
		// In case someone specifies a peer as a path into vendor, convert
		// it to its "real" package path.
		if i := strings.Index(pkg, "/vendor/"); i != -1 {
			pkg = pkg[i+len("/vendor/"):]
		}
		peerPkgs = append(peerPkgs, pkg)
	}
	if expanded, err := context.FindPackages(peerPkgs...); err != nil {
		klog.Fatalf("cannot find peer packages: %v", err)
	} else {
		peerPkgs = expanded // now in fully canonical form
	}
	inputPkgs = append(inputPkgs, peerPkgs...)

	if len(inputPkgs) > 0 {
		if _, err := context.LoadPackages(inputPkgs...); err != nil {
			klog.Fatalf("cannot load packages: %v", err)
		}
	}
	// update context.Order to the latest context.Universe
	orderer := namer.Orderer{Namer: namer.NewPublicNamer(1)}
	context.Order = orderer.OrderUniverse(context.Universe)

	// We also need the to be able to look up the packages of inputs
	inputToPkg := make(map[string]string, len(pkgToInput))
	for k, v := range pkgToInput {
		inputToPkg[v] = k
	}

	for _, i := range context.Inputs {
		pkg := context.Universe[i]

		// typesPkg is where the types that need validation are defined.
		// Sometimes it is different from pkg. For example, kubernetes core/v1
		// types are defined in k8s.io/api/core/v1, while the pkg which holds
		// defaulter code is at k/k/pkg/api/v1.
		typesPkg := pkg

		enabledTags, disabledTags := extractFiltersTags(pkg.Comments)
		declarativeValidator := validators.NewValidator(context, enabledTags, disabledTags)

		typesWith := extractTag(pkg.Comments)
		shouldCreateObjectValidationFn := func(t *types.Type) bool {
			// opt-out
			if checkTag(t.SecondClosestCommentLines, "false") {
				return false
			}
			// opt-in
			if checkTag(t.SecondClosestCommentLines, "true") {
				return true
			}
			// For every k8s:validation-gen tag at the package level, interpret the value as a
			// field name (like TypeMeta, ListMeta, ObjectMeta) and trigger validation generation
			// for any type with any of the matching field names. Provides a more useful package
			// level validation than global (because we only need validations on a subset of objects -
			// usually those with TypeMeta).
			return isTypeWith(t, typesWith)
		}

		// Find the right input pkg, which might not be this one.
		inputPath := pkgToInput[i]
		typesPkg = context.Universe[inputPath]

		var rootTypes []*types.Type
		for _, t := range typesPkg.Types {
			if shouldCreateObjectValidationFn(t) {
				rootTypes = append(rootTypes, t)
			}
		}

		validationFunctionTypes := sets.New[*types.Type]()
		visited := sets.New[*types.Type]()
		for _, t := range rootTypes {
			if validationFunctionTypes.Has(t) { // already found
				continue
			}
			callTree, err := buildCallTree(declarativeValidator, inputToPkg, t)
			if err != nil {
				klog.Fatalf("Failed to build call tree to generate validations for type: %v: %v", t.Name, err)
			}
			if callTree == nil {
				continue
			}
			callTree.VisitInOrder(func(ancestors []*callNode, current *callNode) {
				if visited.Has(current.underlyingType) {
					return
				}
				visited.Insert(current.underlyingType)
				// Generate a validation function for each struct.
				if current.underlyingType != nil && current.underlyingType.Kind == types.Struct {
					validationFunctionTypes.Insert(current.underlyingType)
				}
			})
		}

		if len(validationFunctionTypes) == 0 {
			klog.V(5).Infof("no validations in package %s", pkg.Name)
		}

		targets = append(targets,
			&generator.SimpleTarget{
				PkgName:       path.Base(pkg.Path),
				PkgPath:       pkg.Path,
				PkgDir:        pkg.Dir, // output pkg is the same as the input
				HeaderComment: boilerplate,

				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == typesPkg.Path
				},

				GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
					return []generator.Generator{
						NewGenValidations(args.OutputFile, typesPkg.Path, pkg.Path, rootTypes, validationFunctionTypes, peerPkgs, inputToPkg, declarativeValidator),
					}
				},
			})
	}
	return targets
}

func isTypeWith(t *types.Type, typesWith []string) bool {
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
