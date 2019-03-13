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
	"strings"

	"github.com/go-openapi/spec"

	"k8s.io/kube-openapi/pkg/util"
)

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

	ret = replaceReferences(func(ref *spec.Ref) *spec.Ref {
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

// mergeSpecs merged source into dest while resolving conflicts.
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
	// Check for model conflicts
	conflicts := false
	for k, v := range source.Definitions {
		v2, found := dest.Definitions[k]
		if found && !reflect.DeepEqual(v, v2) {
			if !renameModelConflicts {
				return fmt.Errorf("model name conflict in merging OpenAPI spec: %s", k)
			}
			conflicts = true
			break
		}
	}

	if conflicts {
		usedNames := map[string]bool{}
		for k := range dest.Definitions {
			usedNames[k] = true
		}
		renames := map[string]string{}

	OUTERLOOP:
		for k, v := range source.Definitions {
			if usedNames[k] {
				v2, found := dest.Definitions[k]
				// Reuse model if they are exactly the same.
				if found && reflect.DeepEqual(v, v2) {
					continue
				}

				// Reuse previously renamed model if one exists
				var newName string
				i := 1
				for found {
					i++
					newName = fmt.Sprintf("%s_v%d", k, i)
					v2, found = dest.Definitions[newName]
					if found && reflect.DeepEqual(v, v2) {
						renames[k] = newName
						continue OUTERLOOP
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
		}
		source = renameDefinition(source, renames)
	}
	for k, v := range source.Definitions {
		if _, found := dest.Definitions[k]; !found {
			if dest.Definitions == nil {
				dest.Definitions = spec.Definitions{}
			}
			dest.Definitions[k] = v
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
