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
		def := s.root.Definitions[refStr[len(definitionPrefix):]]
		s.walkSchema(&def)
	}
	return s.walkRefCallback(ref)
}

func (s *referenceWalker) walkSchema(schema *spec.Schema) {
	if schema == nil {
		return
	}
	schema.Ref = s.walkRef(schema.Ref)
	for _, v := range schema.Definitions {
		s.walkSchema(&v)
	}
	for _, v := range schema.Properties {
		s.walkSchema(&v)
	}
	for _, v := range schema.PatternProperties {
		s.walkSchema(&v)
	}
	for _, v := range schema.AllOf {
		s.walkSchema(&v)
	}
	for _, v := range schema.AnyOf {
		s.walkSchema(&v)
	}
	for _, v := range schema.OneOf {
		s.walkSchema(&v)
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
		for _, v := range schema.Items.Schemas {
			s.walkSchema(&v)
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

// FilterSpecByPaths remove unnecessary paths and unused definitions.
func FilterSpecByPaths(sp *spec.Swagger, keepPathPrefixes []string) {
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
	usedDefinitions := map[string]bool{}

	walkOnAllReferences(func(ref spec.Ref) spec.Ref {
		if ref.String() != "" {
			refStr := ref.String()
			if strings.HasPrefix(refStr, definitionPrefix) {
				usedDefinitions[refStr[len(definitionPrefix):]] = true
			}
		}
		return ref
	}, sp)

	// Remove unused definitions
	orgDefinitions := sp.Definitions
	sp.Definitions = spec.Definitions{}
	for k, v := range orgDefinitions {
		if usedDefinitions[k] {
			sp.Definitions[k] = v
		}
	}
}

func renameDefinition(s *spec.Swagger, old, new string) {
	old_ref := definitionPrefix + old
	new_ref := definitionPrefix + new
	walkOnAllReferences(func(ref spec.Ref) spec.Ref {
		if ref.String() == old_ref {
			return spec.MustCreateRef(new_ref)
		}
		return ref
	}, s)
	s.Definitions[new] = s.Definitions[old]
	delete(s.Definitions, old)
}

// Copy paths and definitions from source to dest, rename definitions if needed.
// dest will be mutated, and source will not be changed.
func MergeSpecs(dest, source *spec.Swagger) error {
	sourceCopy, err := CloneSpec(source)
	if err != nil {
		return err
	}
	for k, v := range sourceCopy.Paths.Paths {
		if _, found := dest.Paths.Paths[k]; found {
			return fmt.Errorf("unable to merge: duplicated path %s", k)
		}
		dest.Paths.Paths[k] = v
	}
	usedNames := map[string]bool{}
	for k := range dest.Definitions {
		usedNames[k] = true
	}
	type Rename struct {
		from, to string
	}
	renames := []Rename{}
	for k, v := range sourceCopy.Definitions {
		if usedNames[k] {
			v2, found := dest.Definitions[k]
			// Reuse model iff they are exactly the same.
			if found && reflect.DeepEqual(v, v2) {
				continue
			}
			i := 2
			newName := fmt.Sprintf("%s_v%d", k, i)
			_, foundInSource := sourceCopy.Definitions[newName]
			for usedNames[newName] || foundInSource {
				i += 1
				newName = fmt.Sprintf("%s_v%d", k, i)
				_, foundInSource = sourceCopy.Definitions[newName]
			}
			renames = append(renames, Rename{from: k, to: newName})
			usedNames[newName] = true
		}
	}
	for _, r := range renames {
		renameDefinition(sourceCopy, r.from, r.to)
	}
	for k, v := range sourceCopy.Definitions {
		if _, found := dest.Definitions[k]; !found {
			dest.Definitions[k] = v
		}
	}
	return nil
}

// Clone OpenAPI spec
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
