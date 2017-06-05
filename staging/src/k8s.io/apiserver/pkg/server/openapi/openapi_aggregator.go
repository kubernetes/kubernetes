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

package openapi

import (
	"fmt"
	"strings"

	"github.com/go-openapi/spec"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apiserver/pkg/util/trie"
)

const (
	DEFINITION_PREFIX = "#/definitions/"
)

var cloner = conversion.NewCloner()

// Run a walkRefCallback method on all references of an OpenAPI spec
type walkAllRefs struct {
	// walkRefCallback will be called on each reference and the return value
	// will replace that reference. This will allow the callers to change
	// all/some references of an spec (e.g. useful in renaming definitions).
	walkRefCallback func(ref spec.Ref) spec.Ref

	// The spec to walk through.
	root *spec.Swagger
}

func newWalkAllRefs(walkRef func(ref spec.Ref) spec.Ref, sp *spec.Swagger) *walkAllRefs {
	return &walkAllRefs{
		walkRefCallback: walkRef,
		root:            sp,
	}
}

func (s *walkAllRefs) walkRef(ref spec.Ref) spec.Ref {
	if ref.String() != "" {
		refStr := ref.String()
		// References that start with #/definitions/ has a definition
		// inside the same spec file. If that is the case, walk through
		// those definitions too.
		// We do not support external references yet.
		if strings.HasPrefix(refStr, DEFINITION_PREFIX) {
			def := s.root.Definitions[refStr[len(DEFINITION_PREFIX):]]
			s.walkSchema(&def)
		}
	}
	return s.walkRefCallback(ref)
}

func (s *walkAllRefs) walkSchema(schema *spec.Schema) {
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

func (s *walkAllRefs) walkParams(params []spec.Parameter) {
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

func (s *walkAllRefs) walkResponse(resp *spec.Response) {
	if resp == nil {
		return
	}
	resp.Ref = s.walkRef(resp.Ref)
	s.walkSchema(resp.Schema)
}

func (s *walkAllRefs) walkOperation(op *spec.Operation) {
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

func (s *walkAllRefs) Start() {
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
	prefixes := trie.New(keepPathPrefixes)
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

	newWalkAllRefs(func(ref spec.Ref) spec.Ref {
		if ref.String() != "" {
			refStr := ref.String()
			if strings.HasPrefix(refStr, DEFINITION_PREFIX) {
				usedDefinitions[refStr[len(DEFINITION_PREFIX):]] = true
			}
		}
		return ref
	}, sp).Start()

	// Remove unused definitions
	orgDefinitions := sp.Definitions
	sp.Definitions = spec.Definitions{}
	for k, v := range orgDefinitions {
		if usedDefinitions[k] {
			sp.Definitions[k] = v
		}
	}
}

func equalSchemaMap(s1, s2 map[string]spec.Schema) bool {
	if len(s1) != len(s2) {
		return false
	}
	for k, v := range s1 {
		v2, found := s2[k]
		if !found {
			return false
		}
		if !EqualSchema(&v, &v2) {
			return false
		}
	}
	return true
}

func equalSchemaArray(s1, s2 []spec.Schema) bool {
	if s1 == nil || s2 == nil {
		return s1 == nil && s2 == nil
	}
	if len(s1) != len(s2) {
		return false
	}
	for _, v1 := range s1 {
		found := false
		for _, v2 := range s2 {
			if EqualSchema(&v1, &v2) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	for _, v2 := range s2 {
		found := false
		for _, v1 := range s1 {
			if EqualSchema(&v1, &v2) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func equalSchemaOrBool(s1, s2 *spec.SchemaOrBool) bool {
	if s1 == nil || s2 == nil {
		return s1 == s2
	}
	if s1.Allows != s2.Allows {
		return false
	}
	if !EqualSchema(s1.Schema, s2.Schema) {
		return false
	}
	return true
}

func equalSchemaOrArray(s1, s2 *spec.SchemaOrArray) bool {
	if s1 == nil || s2 == nil {
		return s1 == s2
	}
	if !EqualSchema(s1.Schema, s2.Schema) {
		return false
	}
	if !equalSchemaArray(s1.Schemas, s2.Schemas) {
		return false
	}
	return true
}

func equalStringArray(s1, s2 []string) bool {
	if len(s1) != len(s2) {
		return false
	}
	for _, v1 := range s1 {
		found := false
		for _, v2 := range s2 {
			if v1 == v2 {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	for _, v2 := range s2 {
		found := false
		for _, v1 := range s1 {
			if v1 == v2 {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func equalFloatPointer(s1, s2 *float64) bool {
	if s1 == nil || s2 == nil {
		return s1 == s2
	}
	return *s1 == *s2
}

func equalIntPointer(s1, s2 *int64) bool {
	if s1 == nil || s2 == nil {
		return s1 == s2
	}
	return *s1 == *s2
}

// EqualSchema returns true if models have the same properties and references
// even if they have different documentation.
func EqualSchema(s1, s2 *spec.Schema) bool {
	if s1 == nil || s2 == nil {
		return s1 == s2
	}
	if s1.Ref.String() != s2.Ref.String() {
		return false
	}
	if !equalSchemaMap(s1.Definitions, s2.Definitions) {
		return false
	}
	if !equalSchemaMap(s1.Properties, s2.Properties) {
		fmt.Println("Not equal props")
		return false
	}
	if !equalSchemaMap(s1.PatternProperties, s2.PatternProperties) {
		return false
	}
	if !equalSchemaArray(s1.AllOf, s2.AllOf) {
		return false
	}
	if !equalSchemaArray(s1.AnyOf, s2.AnyOf) {
		return false
	}
	if !equalSchemaArray(s1.OneOf, s2.OneOf) {
		return false
	}
	if !EqualSchema(s1.Not, s2.Not) {
		return false
	}
	if !equalSchemaOrBool(s1.AdditionalProperties, s2.AdditionalProperties) {
		return false
	}
	if !equalSchemaOrBool(s1.AdditionalItems, s2.AdditionalItems) {
		return false
	}
	if !equalSchemaOrArray(s1.Items, s2.Items) {
		return false
	}
	if !equalStringArray(s1.Type, s2.Type) {
		return false
	}
	if s1.Format != s2.Format {
		return false
	}
	if !equalFloatPointer(s1.Minimum, s2.Minimum) {
		return false
	}
	if !equalFloatPointer(s1.Maximum, s2.Maximum) {
		return false
	}
	if s1.ExclusiveMaximum != s2.ExclusiveMaximum {
		return false
	}
	if s1.ExclusiveMinimum != s2.ExclusiveMinimum {
		return false
	}
	if !equalFloatPointer(s1.MultipleOf, s2.MultipleOf) {
		return false
	}
	if !equalIntPointer(s1.MaxLength, s2.MaxLength) {
		return false
	}
	if !equalIntPointer(s1.MinLength, s2.MinLength) {
		return false
	}
	if !equalIntPointer(s1.MaxItems, s2.MaxItems) {
		return false
	}
	if !equalIntPointer(s1.MinItems, s2.MinItems) {
		return false
	}
	if s1.Pattern != s2.Pattern {
		return false
	}
	if s1.UniqueItems != s2.UniqueItems {
		return false
	}
	if !equalIntPointer(s1.MaxProperties, s2.MaxProperties) {
		return false
	}
	if !equalIntPointer(s1.MinProperties, s2.MinProperties) {
		return false
	}
	if !equalStringArray(s1.Required, s2.Required) {
		return false
	}
	return len(s1.Enum) == 0 && len(s2.Enum) == 0 && len(s1.Dependencies) == 0 && len(s2.Dependencies) == 0
}

func renameDefinition(s *spec.Swagger, old, new string) {
	old_ref := DEFINITION_PREFIX + old
	new_ref := DEFINITION_PREFIX + new
	newWalkAllRefs(func(ref spec.Ref) spec.Ref {
		if ref.String() == old_ref {
			return spec.MustCreateRef(new_ref)
		}
		return ref
	}, s).Start()
	s.Definitions[new] = s.Definitions[old]
	delete(s.Definitions, old)
}

// Copy paths and definitions from source to dest, rename definitions if needed.
// dest will be mutated, and source will not be changed.
func MergeSpecs(dest, source *spec.Swagger) error {
	source, err := CloneSpec(source)
	if err != nil {
		return err
	}
	for k, v := range source.Paths.Paths {
		if _, found := dest.Paths.Paths[k]; found {
			return fmt.Errorf("Unable to merge: Duplicated path %s", k)
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
	for k, v := range source.Definitions {
		v2, found := dest.Definitions[k]
		if found || usedNames[k] {
			if found && EqualSchema(&v, &v2) {
				continue
			}
			i := 2
			newName := fmt.Sprintf("%s_v%d", k, i)
			for usedNames[newName] {
				i += 1
				newName = fmt.Sprintf("%s_v%d", k, i)
			}
			renames = append(renames, Rename{from: k, to: newName})
			usedNames[newName] = true
		} else {
			usedNames[k] = true
		}
	}
	for _, r := range renames {
		renameDefinition(source, r.from, r.to)
	}
	for k, v := range source.Definitions {
		if _, found := dest.Definitions[k]; !found {
			dest.Definitions[k] = v
		}
	}
	return nil
}

// Clone OpenAPI spec
func CloneSpec(source *spec.Swagger) (*spec.Swagger, error) {
	if ret, err := cloner.DeepCopy(source); err != nil {
		return nil, err
	} else {
		return ret.(*spec.Swagger), nil
	}
}
