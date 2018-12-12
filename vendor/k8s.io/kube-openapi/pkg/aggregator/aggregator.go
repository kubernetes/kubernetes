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
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/go-openapi/spec"

	"k8s.io/kube-openapi/pkg/util"
)

const (
	definitionPrefix = "#/definitions/"
)

// Run a walkRefCallback method on all references of an OpenAPI spec
type referenceWalker struct {
	// walkRefCallback will be called on each reference and the return value
	// will replace that reference. This will allow the callers to change
	// all/some references of an spec (e.g. useful in renaming definitions).
	walkRefCallback func(ref spec.Ref) spec.Ref

	// The spec to walk through.
	root *spec.Swagger

	// Keep track of visited references
	alreadyVisited map[string]bool
}

func walkOnAllReferences(walkRef func(ref spec.Ref) spec.Ref, sp *spec.Swagger) {
	walker := &referenceWalker{walkRefCallback: walkRef, root: sp, alreadyVisited: map[string]bool{}}
	walker.Start()
}

func (s *referenceWalker) walkRef(ref spec.Ref) spec.Ref {
	refStr := ref.String()
	// References that start with #/definitions/ has a definition
	// inside the same spec file. If that is the case, walk through
	// those definitions too.
	// We do not support external references yet.
	if !s.alreadyVisited[refStr] && strings.HasPrefix(refStr, definitionPrefix) {
		s.alreadyVisited[refStr] = true
		k := refStr[len(definitionPrefix):]
		def := s.root.Definitions[k]
		s.walkSchema(&def)
		// Make sure we don't assign to nil map
		if s.root.Definitions == nil {
			s.root.Definitions = spec.Definitions{}
		}
		s.root.Definitions[k] = def
	}
	return s.walkRefCallback(ref)
}

func (s *referenceWalker) walkSchema(schema *spec.Schema) {
	if schema == nil {
		return
	}
	schema.Ref = s.walkRef(schema.Ref)
	for k, v := range schema.Definitions {
		s.walkSchema(&v)
		schema.Definitions[k] = v
	}
	for k, v := range schema.Properties {
		s.walkSchema(&v)
		schema.Properties[k] = v
	}
	for k, v := range schema.PatternProperties {
		s.walkSchema(&v)
		schema.PatternProperties[k] = v
	}
	for i := range schema.AllOf {
		s.walkSchema(&schema.AllOf[i])
	}
	for i := range schema.AnyOf {
		s.walkSchema(&schema.AnyOf[i])
	}
	for i := range schema.OneOf {
		s.walkSchema(&schema.OneOf[i])
	}
	if schema.Not != nil {
		s.walkSchema(schema.Not)
	}
	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		s.walkSchema(schema.AdditionalProperties.Schema)
	}
	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		s.walkSchema(schema.AdditionalItems.Schema)
	}
	if schema.Items != nil {
		if schema.Items.Schema != nil {
			s.walkSchema(schema.Items.Schema)
		}
		for i := range schema.Items.Schemas {
			s.walkSchema(&schema.Items.Schemas[i])
		}
	}
}

func (s *referenceWalker) walkParams(params []spec.Parameter) {
	if params == nil {
		return
	}
	for _, param := range params {
		param.Ref = s.walkRef(param.Ref)
		s.walkSchema(param.Schema)
		if param.Items != nil {
			param.Items.Ref = s.walkRef(param.Items.Ref)
		}
	}
}

func (s *referenceWalker) walkResponse(resp *spec.Response) {
	if resp == nil {
		return
	}
	resp.Ref = s.walkRef(resp.Ref)
	s.walkSchema(resp.Schema)
}

func (s *referenceWalker) walkOperation(op *spec.Operation) {
	if op == nil {
		return
	}
	s.walkParams(op.Parameters)
	if op.Responses == nil {
		return
	}
	s.walkResponse(op.Responses.Default)
	for _, r := range op.Responses.StatusCodeResponses {
		s.walkResponse(&r)
	}
}

func (s *referenceWalker) Start() {
	if s.root.Paths == nil {
		return
	}
	for _, pathItem := range s.root.Paths.Paths {
		s.walkParams(pathItem.Parameters)
		s.walkOperation(pathItem.Delete)
		s.walkOperation(pathItem.Get)
		s.walkOperation(pathItem.Head)
		s.walkOperation(pathItem.Options)
		s.walkOperation(pathItem.Patch)
		s.walkOperation(pathItem.Post)
		s.walkOperation(pathItem.Put)
	}
}

// usedDefinitionForSpec returns a map with all used definitions in the provided spec as keys and true as values.
func usedDefinitionForSpec(sp *spec.Swagger) map[string]bool {
	usedDefinitions := map[string]bool{}
	walkOnAllReferences(func(ref spec.Ref) spec.Ref {
		if refStr := ref.String(); refStr != "" && strings.HasPrefix(refStr, definitionPrefix) {
			usedDefinitions[refStr[len(definitionPrefix):]] = true
		}
		return ref
	}, sp)
	return usedDefinitions
}

// FilterSpecByPaths removes unnecessary paths and definitions used by those paths.
// i.e. if a Path removed by this function, all definitions used by it and not used
// anywhere else will also be removed.
func FilterSpecByPaths(sp *spec.Swagger, keepPathPrefixes []string) {
	// Walk all references to find all used definitions. This function
	// want to only deal with unused definitions resulted from filtering paths.
	// Thus a definition will be removed only if it has been used before but
	// it is unused because of a path prune.
	initialUsedDefinitions := usedDefinitionForSpec(sp)

	// First remove unwanted paths
	prefixes := util.NewTrie(keepPathPrefixes)
	orgPaths := sp.Paths
	if orgPaths == nil {
		return
	}
	sp.Paths = &spec.Paths{
		VendorExtensible: orgPaths.VendorExtensible,
		Paths:            map[string]spec.PathItem{},
	}
	for path, pathItem := range orgPaths.Paths {
		if !prefixes.HasPrefix(path) {
			continue
		}
		sp.Paths.Paths[path] = pathItem
	}

	// Walk all references to find all definition references.
	usedDefinitions := usedDefinitionForSpec(sp)

	// Remove unused definitions
	orgDefinitions := sp.Definitions
	sp.Definitions = spec.Definitions{}
	for k, v := range orgDefinitions {
		if usedDefinitions[k] || !initialUsedDefinitions[k] {
			sp.Definitions[k] = v
		}
	}
}

func renameDefinition(s *spec.Swagger, old, new string) {
	oldRef := definitionPrefix + old
	newRef := definitionPrefix + new
	walkOnAllReferences(func(ref spec.Ref) spec.Ref {
		if ref.String() == oldRef {
			return spec.MustCreateRef(newRef)
		}
		return ref
	}, s)
	// Make sure we don't assign to nil map
	if s.Definitions == nil {
		s.Definitions = spec.Definitions{}
	}
	s.Definitions[new] = s.Definitions[old]
	delete(s.Definitions, old)
}

// MergeSpecsIgnorePathConflict is the same as MergeSpecs except it will ignore any path
// conflicts by keeping the paths of destination. It will rename definition conflicts.
func MergeSpecsIgnorePathConflict(dest, source *spec.Swagger) error {
	return mergeSpecs(dest, source, true, true)
}

// MergeSpecsFailOnDefinitionConflict is differ from MergeSpecs as it fails if there is
// a definition conflict.
func MergeSpecsFailOnDefinitionConflict(dest, source *spec.Swagger) error {
	return mergeSpecs(dest, source, false, false)
}

// MergeSpecs copies paths and definitions from source to dest, rename definitions if needed.
// dest will be mutated, and source will not be changed. It will fail on path conflicts.
func MergeSpecs(dest, source *spec.Swagger) error {
	return mergeSpecs(dest, source, true, false)
}

func mergeSpecs(dest, source *spec.Swagger, renameModelConflicts, ignorePathConflicts bool) (err error) {
	specCloned := false
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
			source, err = CloneSpec(source)
			if err != nil {
				return err
			}
			specCloned = true
			FilterSpecByPaths(source, keepPaths)
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
		if !specCloned {
			source, err = CloneSpec(source)
			if err != nil {
				return err
			}
		}
		specCloned = true
		usedNames := map[string]bool{}
		for k := range dest.Definitions {
			usedNames[k] = true
		}
		type Rename struct {
			from, to string
		}
		renames := []Rename{}

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
						renames = append(renames, Rename{from: k, to: newName})
						continue OUTERLOOP
					}
				}

				_, foundInSource := source.Definitions[newName]
				for usedNames[newName] || foundInSource {
					i++
					newName = fmt.Sprintf("%s_v%d", k, i)
					_, foundInSource = source.Definitions[newName]
				}
				renames = append(renames, Rename{from: k, to: newName})
				usedNames[newName] = true
			}
		}
		for _, r := range renames {
			renameDefinition(source, r.from, r.to)
		}
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

// CloneSpec clones OpenAPI spec
func CloneSpec(source *spec.Swagger) (*spec.Swagger, error) {
	// TODO(mehdy): Find a faster way to clone an spec
	bytes, err := json.Marshal(source)
	if err != nil {
		return nil, err
	}
	var ret spec.Swagger
	err = json.Unmarshal(bytes, &ret)
	if err != nil {
		return nil, err
	}
	return &ret, nil
}
