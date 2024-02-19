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

// Visitor recursively walks through a structural schema.
type Visitor struct {
	// Structural is called on each Structural node in the schema, before recursing into
	// the subtrees. It is allowed to mutate s. Return true if something has been changed.
	// +optional
	Structural func(s *Structural) bool
	// NestedValueValidation is called on each NestedValueValidation node in the schema,
	// before recursing into subtrees. It is allowed to mutate vv. Return true if something
	// has been changed.
	// +optional
	NestedValueValidation func(vv *NestedValueValidation) bool
}

// Visit recursively walks through the structural schema and calls the given callbacks
// at each node of those types.
func (m *Visitor) Visit(s *Structural) {
	m.visitStructural(s)
}

func (m *Visitor) visitStructural(s *Structural) bool {
	ret := false
	if m.Structural != nil {
		ret = m.Structural(s)
	}

	if s.Items != nil {
		m.visitStructural(s.Items)
	}
	for k, v := range s.Properties {
		if changed := m.visitStructural(&v); changed {
			ret = true
			s.Properties[k] = v
		}
	}
	if s.Generic.AdditionalProperties != nil && s.Generic.AdditionalProperties.Structural != nil {
		m.visitStructural(s.Generic.AdditionalProperties.Structural)
	}
	if s.ValueValidation != nil {
		for i := range s.ValueValidation.AllOf {
			m.visitNestedValueValidation(&s.ValueValidation.AllOf[i])
		}
		for i := range s.ValueValidation.AnyOf {
			m.visitNestedValueValidation(&s.ValueValidation.AnyOf[i])
		}
		for i := range s.ValueValidation.OneOf {
			m.visitNestedValueValidation(&s.ValueValidation.OneOf[i])
		}
		if s.ValueValidation.Not != nil {
			m.visitNestedValueValidation(s.ValueValidation.Not)
		}
	}

	return ret
}

func (m *Visitor) visitNestedValueValidation(vv *NestedValueValidation) bool {
	ret := false
	if m.NestedValueValidation != nil {
		ret = m.NestedValueValidation(vv)
	}

	if vv.Items != nil {
		m.visitNestedValueValidation(vv.Items)
	}
	for k, v := range vv.Properties {
		if changed := m.visitNestedValueValidation(&v); changed {
			ret = true
			vv.Properties[k] = v
		}
	}
	if vv.ForbiddenGenerics.AdditionalProperties != nil && vv.ForbiddenGenerics.AdditionalProperties.Structural != nil {
		m.visitStructural(vv.ForbiddenGenerics.AdditionalProperties.Structural)
	}
	for i := range vv.ValueValidation.AllOf {
		m.visitNestedValueValidation(&vv.ValueValidation.AllOf[i])
	}
	for i := range vv.ValueValidation.AnyOf {
		m.visitNestedValueValidation(&vv.ValueValidation.AnyOf[i])
	}
	for i := range vv.ValueValidation.OneOf {
		m.visitNestedValueValidation(&vv.ValueValidation.OneOf[i])
	}
	if vv.ValueValidation.Not != nil {
		m.visitNestedValueValidation(vv.ValueValidation.Not)
	}

	return ret
}
