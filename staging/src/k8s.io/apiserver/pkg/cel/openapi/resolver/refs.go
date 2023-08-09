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

	"k8s.io/kube-openapi/pkg/validation/spec"
)

// populateRefs recursively replaces Refs in the schema with the referred one.
// schemaOf is the callback to find the corresponding schema by the ref.
// This function will not mutate the original schema. If the schema needs to be
// mutated, a copy will be returned, otherwise it returns the original schema.
func populateRefs(schemaOf func(ref string) (*spec.Schema, bool), schema *spec.Schema) (*spec.Schema, error) {
	result := *schema
	changed := false

	ref, isRef := refOf(schema)
	if isRef {
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
		populated, err := populateRefs(schemaOf, &prop)
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
		populated, err := populateRefs(schemaOf, result.AdditionalProperties.Schema)
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
		populated, err := populateRefs(schemaOf, result.Items.Schema)
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
