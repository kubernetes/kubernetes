/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
)

// NewStructural converts an OpenAPI v3 schema into a structural schema. A pre-validated JSONSchemaProps will
// not fail on NewStructural. This means that we require that:
//
// - items is not an array of schemas
// - the following fields are not set:
//   - id
//   - schema
//   - $ref
//   - patternProperties
//   - dependencies
//   - additionalItems
//   - definitions.
//
// The follow fields are not preserved:
// - externalDocs
// - example.
func NewStructural(s *apiextensions.JSONSchemaProps) (*Structural, error) {
	if s == nil {
		return nil, nil
	}

	if err := validateUnsupportedFields(s); err != nil {
		return nil, err
	}

	vv, err := newValueValidation(s)
	if err != nil {
		return nil, err
	}

	g, err := newGenerics(s)
	if err != nil {
		return nil, err
	}

	x, err := newExtensions(s)
	if err != nil {
		return nil, err
	}

	ss := &Structural{
		Generic:         *g,
		Extensions:      *x,
		ValueValidation: vv,
	}

	if s.Items != nil {
		if len(s.Items.JSONSchemas) > 0 {
			// we validate that it is not an array
			return nil, fmt.Errorf("OpenAPIV3Schema 'items' must be a schema, but is an array")
		}
		item, err := NewStructural(s.Items.Schema)
		if err != nil {
			return nil, err
		}
		ss.Items = item
	}

	if len(s.Properties) > 0 {
		ss.Properties = make(map[string]Structural, len(s.Properties))
		for k, x := range s.Properties {
			fld, err := NewStructural(&x)
			if err != nil {
				return nil, err
			}
			ss.Properties[k] = *fld
		}
	}

	return ss, nil
}

func newGenerics(s *apiextensions.JSONSchemaProps) (*Generic, error) {
	if s == nil {
		return nil, nil
	}
	g := &Generic{
		Type:        s.Type,
		Description: s.Description,
		Title:       s.Title,
		Nullable:    s.Nullable,
	}
	if s.Default != nil {
		g.Default = JSON{interface{}(*s.Default)}
	}

	if s.AdditionalProperties != nil {
		if s.AdditionalProperties.Schema != nil {
			ss, err := NewStructural(s.AdditionalProperties.Schema)
			if err != nil {
				return nil, err
			}
			g.AdditionalProperties = &StructuralOrBool{Structural: ss, Bool: true}
		} else {
			g.AdditionalProperties = &StructuralOrBool{Bool: s.AdditionalProperties.Allows}
		}
	}

	return g, nil
}

func newValueValidation(s *apiextensions.JSONSchemaProps) (*ValueValidation, error) {
	if s == nil {
		return nil, nil
	}
	not, err := newNestedValueValidation(s.Not)
	if err != nil {
		return nil, err
	}
	v := &ValueValidation{
		Format:           s.Format,
		Maximum:          s.Maximum,
		ExclusiveMaximum: s.ExclusiveMaximum,
		Minimum:          s.Minimum,
		ExclusiveMinimum: s.ExclusiveMinimum,
		MaxLength:        s.MaxLength,
		MinLength:        s.MinLength,
		Pattern:          s.Pattern,
		MaxItems:         s.MaxItems,
		MinItems:         s.MinItems,
		UniqueItems:      s.UniqueItems,
		MultipleOf:       s.MultipleOf,
		MaxProperties:    s.MaxProperties,
		MinProperties:    s.MinProperties,
		Required:         s.Required,
		Not:              not,
	}

	for _, e := range s.Enum {
		v.Enum = append(v.Enum, JSON{e})
	}

	for _, x := range s.AllOf {
		clause, err := newNestedValueValidation(&x)
		if err != nil {
			return nil, err
		}
		v.AllOf = append(v.AllOf, *clause)
	}

	for _, x := range s.AnyOf {
		clause, err := newNestedValueValidation(&x)
		if err != nil {
			return nil, err
		}
		v.AnyOf = append(v.AnyOf, *clause)
	}

	for _, x := range s.OneOf {
		clause, err := newNestedValueValidation(&x)
		if err != nil {
			return nil, err
		}
		v.OneOf = append(v.OneOf, *clause)
	}

	return v, nil
}

func newNestedValueValidation(s *apiextensions.JSONSchemaProps) (*NestedValueValidation, error) {
	if s == nil {
		return nil, nil
	}

	if err := validateUnsupportedFields(s); err != nil {
		return nil, err
	}

	vv, err := newValueValidation(s)
	if err != nil {
		return nil, err
	}

	g, err := newGenerics(s)
	if err != nil {
		return nil, err
	}

	x, err := newExtensions(s)
	if err != nil {
		return nil, err
	}

	v := &NestedValueValidation{
		ValueValidation:     *vv,
		ForbiddenGenerics:   *g,
		ForbiddenExtensions: *x,
	}

	if s.Items != nil {
		if len(s.Items.JSONSchemas) > 0 {
			// we validate that it is not an array
			return nil, fmt.Errorf("OpenAPIV3Schema 'items' must be a schema, but is an array")
		}
		nvv, err := newNestedValueValidation(s.Items.Schema)
		if err != nil {
			return nil, err
		}
		v.Items = nvv
	}
	if s.Properties != nil {
		v.Properties = make(map[string]NestedValueValidation, len(s.Properties))
		for k, x := range s.Properties {
			nvv, err := newNestedValueValidation(&x)
			if err != nil {
				return nil, err
			}
			v.Properties[k] = *nvv
		}
	}

	return v, nil
}

func newExtensions(s *apiextensions.JSONSchemaProps) (*Extensions, error) {
	if s == nil {
		return nil, nil
	}

	ret := &Extensions{
		XEmbeddedResource: s.XEmbeddedResource,
		XIntOrString:      s.XIntOrString,
		XListMapKeys:      s.XListMapKeys,
		XListType:         s.XListType,
	}

	if s.XPreserveUnknownFields != nil {
		if !*s.XPreserveUnknownFields {
			return nil, fmt.Errorf("internal error: 'x-kubernetes-preserve-unknown-fields' must be true or undefined")
		}
		ret.XPreserveUnknownFields = true
	}

	return ret, nil
}

// validateUnsupportedFields checks that those fields rejected by validation are actually unset.
func validateUnsupportedFields(s *apiextensions.JSONSchemaProps) error {
	if len(s.ID) > 0 {
		return fmt.Errorf("OpenAPIV3Schema 'id' is not supported")
	}
	if len(s.Schema) > 0 {
		return fmt.Errorf("OpenAPIV3Schema 'schema' is not supported")
	}
	if s.Ref != nil && len(*s.Ref) > 0 {
		return fmt.Errorf("OpenAPIV3Schema '$ref' is not supported")
	}
	if len(s.PatternProperties) > 0 {
		return fmt.Errorf("OpenAPIV3Schema 'patternProperties' is not supported")
	}
	if len(s.Dependencies) > 0 {
		return fmt.Errorf("OpenAPIV3Schema 'dependencies' is not supported")
	}
	if s.AdditionalItems != nil {
		return fmt.Errorf("OpenAPIV3Schema 'additionalItems' is not supported")
	}
	if len(s.Definitions) > 0 {
		return fmt.Errorf("OpenAPIV3Schema 'definitions' is not supported")
	}

	return nil
}
