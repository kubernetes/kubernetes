/*
Copyright 2023 The Kubernetes Authors.

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

package resolver

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// PopulateRefs recursively replaces Refs in the schema with the referred one.
// schemaOf is the callback to find the corresponding schema by the ref.
// This function will not mutate the original schema. If the schema needs to be
// mutated, a copy will be returned, otherwise it returns the original schema.
func PopulateRefs(schemaOf func(ref string) (*spec.Schema, bool), rootRef string) (*spec.Schema, error) {
	visitedRefs := sets.New[string]()
	rootSchema, ok := schemaOf(rootRef)
	visitedRefs.Insert(rootRef)
	if !ok {
		return nil, fmt.Errorf("internal error: cannot resolve Ref for root schema %q: %w", rootRef, ErrSchemaNotFound)
	}
	return populateRefs(schemaOf, visitedRefs, rootSchema)
}

func populateRefs(schemaOf func(ref string) (*spec.Schema, bool), visited sets.Set[string], schema *spec.Schema) (*spec.Schema, error) {
	result := *schema
	changed := false

	ref, isRef := refOf(schema)
	if isRef {
		if visited.Has(ref) {
			return &spec.Schema{
				// for circular ref, return an empty object as placeholder
				SchemaProps: spec.SchemaProps{Type: []string{"object"}},
			}, nil
		}
		visited.Insert(ref)
		// restore visited state at the end of the recursion.
		defer func() {
			visited.Delete(ref)
		}()
		// replace the whole schema with the referred one.
		resolved, ok := schemaOf(ref)
		if !ok {
			return nil, fmt.Errorf("internal error: cannot resolve Ref %q: %w", ref, ErrSchemaNotFound)
		}
		result = *resolved
		changed = true
	}
	// schema is an object, populate its properties and additionalProperties
	props := make(map[string]spec.Schema, len(schema.Properties))
	propsChanged := false
	for name, prop := range result.Properties {
		populated, err := populateRefs(schemaOf, visited, &prop)
		if err != nil {
			return nil, err
		}
		if populated != &prop {
			propsChanged = true
		}
		props[name] = *populated
	}
	if propsChanged {
		changed = true
		result.Properties = props
	}
	if result.AdditionalProperties != nil && result.AdditionalProperties.Schema != nil {
		populated, err := populateRefs(schemaOf, visited, result.AdditionalProperties.Schema)
		if err != nil {
			return nil, err
		}
		if populated != result.AdditionalProperties.Schema {
			changed = true
			result.AdditionalProperties.Schema = populated
		}
	}
	// schema is a list, populate its items
	if result.Items != nil && result.Items.Schema != nil {
		populated, err := populateRefs(schemaOf, visited, result.Items.Schema)
		if err != nil {
			return nil, err
		}
		if populated != result.Items.Schema {
			changed = true
			result.Items.Schema = populated
		}
	}
	if changed {
		return &result, nil
	}
	return schema, nil
}

func refOf(schema *spec.Schema) (string, bool) {
	if schema.Ref.GetURL() != nil {
		return schema.Ref.String(), true
	}
	// A Ref may be wrapped in allOf to preserve its description
	// see https://github.com/kubernetes/kubernetes/issues/106387
	// For kube-openapi, allOf is only used for wrapping a Ref.
	for _, allOf := range schema.AllOf {
		if ref, isRef := refOf(&allOf); isRef {
			return ref, isRef
		}
	}
	return "", false
}
