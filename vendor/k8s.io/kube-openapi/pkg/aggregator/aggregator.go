/*
Copyright 2017 The Kubernetes Authors.

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

package aggregator

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"

	"k8s.io/kube-openapi/pkg/schemamutation"
	"k8s.io/kube-openapi/pkg/util"
)

const gvkKey = "x-kubernetes-group-version-kind"

// usedDefinitionForSpec returns a map with all used definitions in the provided spec as keys and true as values.
func usedDefinitionForSpec(root *spec.Swagger) map[string]bool {
	usedDefinitions := map[string]bool{}
	walkOnAllReferences(func(ref *spec.Ref) {
		if refStr := ref.String(); refStr != "" && strings.HasPrefix(refStr, definitionPrefix) {
			usedDefinitions[refStr[len(definitionPrefix):]] = true
		}
	}, root)
	return usedDefinitions
}

// FilterSpecByPaths removes unnecessary paths and definitions used by those paths.
// i.e. if a Path removed by this function, all definitions used by it and not used
// anywhere else will also be removed.
func FilterSpecByPaths(sp *spec.Swagger, keepPathPrefixes []string) {
	*sp = *FilterSpecByPathsWithoutSideEffects(sp, keepPathPrefixes)
}

// FilterSpecByPathsWithoutSideEffects removes unnecessary paths and definitions used by those paths.
// i.e. if a Path removed by this function, all definitions used by it and not used
// anywhere else will also be removed.
// It does not modify the input, but the output shares data structures with the input.
func FilterSpecByPathsWithoutSideEffects(sp *spec.Swagger, keepPathPrefixes []string) *spec.Swagger {
	if sp.Paths == nil {
		return sp
	}

	// Walk all references to find all used definitions. This function
	// want to only deal with unused definitions resulted from filtering paths.
	// Thus a definition will be removed only if it has been used before but
	// it is unused because of a path prune.
	initialUsedDefinitions := usedDefinitionForSpec(sp)

	// First remove unwanted paths
	prefixes := util.NewTrie(keepPathPrefixes)
	ret := *sp
	ret.Paths = &spec.Paths{
		VendorExtensible: sp.Paths.VendorExtensible,
		Paths:            map[string]spec.PathItem{},
	}
	for path, pathItem := range sp.Paths.Paths {
		if !prefixes.HasPrefix(path) {
			continue
		}
		ret.Paths.Paths[path] = pathItem
	}

	// Walk all references to find all definition references.
	usedDefinitions := usedDefinitionForSpec(&ret)

	// Remove unused definitions
	ret.Definitions = spec.Definitions{}
	for k, v := range sp.Definitions {
		if usedDefinitions[k] || !initialUsedDefinitions[k] {
			ret.Definitions[k] = v
		}
	}

	return &ret
}

type rename struct {
	from, to string
}

// renameDefinition renames references, without mutating the input.
// The output might share data structures with the input.
func renameDefinition(s *spec.Swagger, renames map[string]string) *spec.Swagger {
	refRenames := make(map[string]string, len(renames))
	foundOne := false
	for k, v := range renames {
		refRenames[definitionPrefix+k] = definitionPrefix + v
		if _, ok := s.Definitions[k]; ok {
			foundOne = true
		}
	}

	if !foundOne {
		return s
	}

	ret := &spec.Swagger{}
	*ret = *s

	ret = schemamutation.ReplaceReferences(func(ref *spec.Ref) *spec.Ref {
		refName := ref.String()
		if newRef, found := refRenames[refName]; found {
			ret := spec.MustCreateRef(newRef)
			return &ret
		}
		return ref
	}, ret)

	renamedDefinitions := make(spec.Definitions, len(ret.Definitions))
	for k, v := range ret.Definitions {
		if newRef, found := renames[k]; found {
			k = newRef
		}
		renamedDefinitions[k] = v
	}
	ret.Definitions = renamedDefinitions

	return ret
}

// MergeSpecsIgnorePathConflict is the same as MergeSpecs except it will ignore any path
// conflicts by keeping the paths of destination. It will rename definition conflicts.
// The source is not mutated.
func MergeSpecsIgnorePathConflict(dest, source *spec.Swagger) error {
	return mergeSpecs(dest, source, true, true)
}

// MergeSpecsFailOnDefinitionConflict is differ from MergeSpecs as it fails if there is
// a definition conflict.
// The source is not mutated.
func MergeSpecsFailOnDefinitionConflict(dest, source *spec.Swagger) error {
	return mergeSpecs(dest, source, false, false)
}

// MergeSpecs copies paths and definitions from source to dest, rename definitions if needed.
// dest will be mutated, and source will not be changed. It will fail on path conflicts.
// The source is not mutated.
func MergeSpecs(dest, source *spec.Swagger) error {
	return mergeSpecs(dest, source, true, false)
}

// mergeSpecs merges source into dest while resolving conflicts.
// The source is not mutated.
func mergeSpecs(dest, source *spec.Swagger, renameModelConflicts, ignorePathConflicts bool) (err error) {
	// Paths may be empty, due to [ACL constraints](http://goo.gl/8us55a#securityFiltering).
	if source.Paths == nil {
		// When a source spec does not have any path, that means none of the definitions
		// are used thus we should not do anything
		return nil
	}
	if dest.Paths == nil {
		dest.Paths = &spec.Paths{}
	}
	if ignorePathConflicts {
		keepPaths := []string{}
		hasConflictingPath := false
		for k := range source.Paths.Paths {
			if _, found := dest.Paths.Paths[k]; !found {
				keepPaths = append(keepPaths, k)
			} else {
				hasConflictingPath = true
			}
		}
		if len(keepPaths) == 0 {
			// There is nothing to merge. All paths are conflicting.
			return nil
		}
		if hasConflictingPath {
			source = FilterSpecByPathsWithoutSideEffects(source, keepPaths)
		}
	}

	// Check for model conflicts and rename to make definitions conflict-free (modulo different GVKs)
	usedNames := map[string]bool{}
	for k := range dest.Definitions {
		usedNames[k] = true
	}
	renames := map[string]string{}
DEFINITIONLOOP:
	for k, v := range source.Definitions {
		existing, found := dest.Definitions[k]
		if !found || deepEqualDefinitionsModuloGVKs(&existing, &v) {
			// skip for now, we copy them after the rename loop
			continue
		}

		if !renameModelConflicts {
			return fmt.Errorf("model name conflict in merging OpenAPI spec: %s", k)
		}

		// Reuse previously renamed model if one exists
		var newName string
		i := 1
		for found {
			i++
			newName = fmt.Sprintf("%s_v%d", k, i)
			existing, found = dest.Definitions[newName]
			if found && deepEqualDefinitionsModuloGVKs(&existing, &v) {
				renames[k] = newName
				continue DEFINITIONLOOP
			}
		}

		_, foundInSource := source.Definitions[newName]
		for usedNames[newName] || foundInSource {
			i++
			newName = fmt.Sprintf("%s_v%d", k, i)
			_, foundInSource = source.Definitions[newName]
		}
		renames[k] = newName
		usedNames[newName] = true
	}
	source = renameDefinition(source, renames)

	// now without conflict (modulo different GVKs), copy definitions to dest
	for k, v := range source.Definitions {
		if existing, found := dest.Definitions[k]; !found {
			if dest.Definitions == nil {
				dest.Definitions = spec.Definitions{}
			}
			dest.Definitions[k] = v
		} else if merged, changed, err := mergedGVKs(&existing, &v); err != nil {
			return err
		} else if changed {
			existing.Extensions[gvkKey] = merged
		}
	}

	// Check for path conflicts
	for k, v := range source.Paths.Paths {
		if _, found := dest.Paths.Paths[k]; found {
			return fmt.Errorf("unable to merge: duplicated path %s", k)
		}
		// PathItem may be empty, due to [ACL constraints](http://goo.gl/8us55a#securityFiltering).
		if dest.Paths.Paths == nil {
			dest.Paths.Paths = map[string]spec.PathItem{}
		}
		dest.Paths.Paths[k] = v
	}

	return nil
}

// deepEqualDefinitionsModuloGVKs compares s1 and s2, but ignores the x-kubernetes-group-version-kind extension.
func deepEqualDefinitionsModuloGVKs(s1, s2 *spec.Schema) bool {
	if s1 == nil {
		return s2 == nil
	} else if s2 == nil {
		return false
	}
	if !reflect.DeepEqual(s1.Extensions, s2.Extensions) {
		for k, v := range s1.Extensions {
			if k == gvkKey {
				continue
			}
			if !reflect.DeepEqual(v, s2.Extensions[k]) {
				return false
			}
		}
		len1 := len(s1.Extensions)
		len2 := len(s2.Extensions)
		if _, found := s1.Extensions[gvkKey]; found {
			len1--
		}
		if _, found := s2.Extensions[gvkKey]; found {
			len2--
		}
		if len1 != len2 {
			return false
		}

		if s1.Extensions != nil {
			shallowCopy := *s1
			s1 = &shallowCopy
			s1.Extensions = nil
		}
		if s2.Extensions != nil {
			shallowCopy := *s2
			s2 = &shallowCopy
			s2.Extensions = nil
		}
	}

	return reflect.DeepEqual(s1, s2)
}

// mergedGVKs merges the x-kubernetes-group-version-kind slices and returns the result, and whether
// s1's x-kubernetes-group-version-kind slice was changed at all.
func mergedGVKs(s1, s2 *spec.Schema) (interface{}, bool, error) {
	gvk1, found1 := s1.Extensions[gvkKey]
	gvk2, found2 := s2.Extensions[gvkKey]

	if !found1 {
		return gvk2, found2, nil
	}
	if !found2 {
		return gvk1, false, nil
	}

	slice1, ok := gvk1.([]interface{})
	if !ok {
		return nil, false, fmt.Errorf("expected slice of GroupVersionKinds, got: %+v", slice1)
	}
	slice2, ok := gvk2.([]interface{})
	if !ok {
		return nil, false, fmt.Errorf("expected slice of GroupVersionKinds, got: %+v", slice2)
	}

	ret := make([]interface{}, len(slice1), len(slice1)+len(slice2))
	keys := make([]string, 0, len(slice1)+len(slice2))
	copy(ret, slice1)
	seen := make(map[string]bool, len(slice1))
	for _, x := range slice1 {
		gvk, ok := x.(map[string]interface{})
		if !ok {
			return nil, false, fmt.Errorf(`expected {"group": <group>, "kind": <kind>, "version": <version>}, got: %#v`, x)
		}
		k := fmt.Sprintf("%s/%s.%s", gvk["group"], gvk["version"], gvk["kind"])
		keys = append(keys, k)
		seen[k] = true
	}
	changed := false
	for _, x := range slice2 {
		gvk, ok := x.(map[string]interface{})
		if !ok {
			return nil, false, fmt.Errorf(`expected {"group": <group>, "kind": <kind>, "version": <version>}, got: %#v`, x)
		}
		k := fmt.Sprintf("%s/%s.%s", gvk["group"], gvk["version"], gvk["kind"])
		if seen[k] {
			continue
		}
		ret = append(ret, x)
		keys = append(keys, k)
		changed = true
	}

	if changed {
		sort.Sort(byKeys{ret, keys})
	}

	return ret, changed, nil
}

type byKeys struct {
	values []interface{}
	keys   []string
}

func (b byKeys) Len() int {
	return len(b.values)
}

func (b byKeys) Less(i, j int) bool {
	return b.keys[i] < b.keys[j]
}

func (b byKeys) Swap(i, j int) {
	b.values[i], b.values[j] = b.values[j], b.values[i]
	b.keys[i], b.keys[j] = b.keys[j], b.keys[i]
}
