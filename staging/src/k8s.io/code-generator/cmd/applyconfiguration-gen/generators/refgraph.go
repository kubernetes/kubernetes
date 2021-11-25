/*
Copyright 2021 The Kubernetes Authors.

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
	"k8s.io/gengo/types"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// refGraph maps existing types to the package the corresponding applyConfig types will be generated in
// so that references between apply configurations can be correctly generated.
type refGraph map[types.Name]string

// refGraphForReachableTypes returns a refGraph that contains all reachable types from
// the root clientgen types of the provided packages.
func refGraphForReachableTypes(universe types.Universe, pkgTypes map[string]*types.Package, initialTypes map[types.Name]string) refGraph {
	var refs refGraph = initialTypes

	// Include only types that are reachable from the root clientgen types.
	// We don't want to generate apply configurations for types that are not reachable from a root
	// clientgen type.
	reachableTypes := map[types.Name]*types.Type{}
	for _, p := range pkgTypes {
		for _, t := range p.Types {
			tags := genclientTags(t)
			hasApply := tags.HasVerb("apply") || tags.HasVerb("applyStatus")
			if tags.GenerateClient && hasApply {
				findReachableTypes(t, reachableTypes)
			}
			// If any apply extensions have custom inputs, add them.
			for _, extension := range tags.Extensions {
				if extension.HasVerb("apply") {
					if len(extension.InputTypeOverride) > 0 {
						inputType := *t
						if name, pkg := extension.Input(); len(pkg) > 0 {
							inputType = *(universe.Type(types.Name{Package: pkg, Name: name}))
						} else {
							inputType.Name.Name = extension.InputTypeOverride
						}
						findReachableTypes(&inputType, reachableTypes)
					}
				}
			}
		}
	}
	for pkg, p := range pkgTypes {
		for _, t := range p.Types {
			if _, ok := reachableTypes[t.Name]; !ok {
				continue
			}
			if requiresApplyConfiguration(t) {
				refs[t.Name] = pkg
			}
		}
	}

	return refs
}

// applyConfigForType find the type used in the generate apply configurations for a field.
// This may either be an existing type or one of the other generated applyConfig types.
func (t refGraph) applyConfigForType(field *types.Type) *types.Type {
	switch field.Kind {
	case types.Struct:
		if pkg, ok := t[field.Name]; ok { // TODO(jpbetz): Refs to types defined in a separate system (e.g. TypeMeta if generating a 3rd party controller) end up referencing the go struct, not the apply configuration type
			return types.Ref(pkg, field.Name.Name+ApplyConfigurationTypeSuffix)
		}
		return field
	case types.Map:
		if _, ok := t[field.Elem.Name]; ok {
			return &types.Type{
				Kind: types.Map,
				Elem: t.applyConfigForType(field.Elem),
			}
		}
		return field
	case types.Slice:
		if _, ok := t[field.Elem.Name]; ok {
			return &types.Type{
				Kind: types.Slice,
				Elem: t.applyConfigForType(field.Elem),
			}
		}
		return field
	case types.Pointer:
		return t.applyConfigForType(field.Elem)
	default:
		return field
	}
}

func (t refGraph) isApplyConfig(field *types.Type) bool {
	switch field.Kind {
	case types.Struct:
		_, ok := t[field.Name]
		return ok
	case types.Pointer:
		return t.isApplyConfig(field.Elem)
	}
	return false
}

// genclientTags returns the genclient Tags for the given type.
func genclientTags(t *types.Type) util.Tags {
	return util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
}

// findReachableTypes finds all types transitively reachable from a given root type, including
// the root type itself.
func findReachableTypes(t *types.Type, referencedTypes map[types.Name]*types.Type) {
	if _, ok := referencedTypes[t.Name]; ok {
		return
	}
	referencedTypes[t.Name] = t

	if t.Elem != nil {
		findReachableTypes(t.Elem, referencedTypes)
	}
	if t.Underlying != nil {
		findReachableTypes(t.Underlying, referencedTypes)
	}
	if t.Key != nil {
		findReachableTypes(t.Key, referencedTypes)
	}
	for _, m := range t.Members {
		findReachableTypes(m.Type, referencedTypes)
	}
}

// excludeTypes contains well known types that we do not generate apply configurations for.
// Hard coding because we only have two, very specific types that serve a special purpose
// in the type system here.
var excludeTypes = map[types.Name]struct{}{
	rawExtension.Name: {},
	unknown.Name:      {},
	// DO NOT ADD TO THIS LIST. If we need to exclude other types, we should consider allowing the
	// go type declarations to be annotated as excluded from this generator.
}

// requiresApplyConfiguration returns true if a type applyConfig should be generated for the given type.
// types applyConfig are only generated for struct types that contain fields with json tags.
func requiresApplyConfiguration(t *types.Type) bool {
	for t.Kind == types.Alias {
		t = t.Underlying
	}
	if t.Kind != types.Struct {
		return false
	}
	if _, ok := excludeTypes[t.Name]; ok {
		return false
	}
	var hasJSONTaggedMembers bool
	for _, member := range t.Members {
		if _, ok := lookupJSONTags(member); ok {
			hasJSONTaggedMembers = true
		}
	}
	if !hasJSONTaggedMembers {
		return false
	}

	return true
}
