/*
Copyright 2018 The Kubernetes Authors.

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

import "sync"

// Schema is a list of named types.
//
// Schema types are indexed in a map before the first search so this type
// should be considered immutable.
type Schema struct {
	Types []TypeDef `yaml:"types,omitempty"`

	once sync.Once
	m    map[string]TypeDef
}

// A TypeSpecifier references a particular type in a schema.
type TypeSpecifier struct {
	Type   TypeRef `yaml:"type,omitempty"`
	Schema Schema  `yaml:"schema,omitempty"`
}

// TypeDef represents a named type in a schema.
type TypeDef struct {
	// Top level types should be named. Every type must have a unique name.
	Name string `yaml:"name,omitempty"`

	Atom `yaml:"atom,omitempty,inline"`
}

// TypeRef either refers to a named type or declares an inlined type.
type TypeRef struct {
	// Either the name or one member of Atom should be set.
	NamedType *string `yaml:"namedType,omitempty"`
	Inlined   Atom    `yaml:",inline,omitempty"`
}

// Atom represents the smallest possible pieces of the type system.
// Each set field in the Atom represents a possible type for the object.
// If none of the fields are set, any object will fail validation against the atom.
type Atom struct {
	*Scalar `yaml:"scalar,omitempty"`
	*List   `yaml:"list,omitempty"`
	*Map    `yaml:"map,omitempty"`
}

// Scalar (AKA "primitive") represents a type which has a single value which is
// either numeric, string, or boolean.
//
// TODO: split numeric into float/int? Something even more fine-grained?
type Scalar string

const (
	Numeric = Scalar("numeric")
	String  = Scalar("string")
	Boolean = Scalar("boolean")
)

// ElementRelationship is an enum of the different possible relationships
// between the elements of container types (maps, lists).
type ElementRelationship string

const (
	// Associative only applies to lists (see the documentation there).
	Associative = ElementRelationship("associative")
	// Atomic makes container types (lists, maps) behave
	// as scalars / leaf fields
	Atomic = ElementRelationship("atomic")
	// Separable means the items of the container type have no particular
	// relationship (default behavior for maps).
	Separable = ElementRelationship("separable")
)

// Map is a key-value pair. Its default semantics are the same as an
// associative list, but:
// * It is serialized differently:
//     map:  {"k": {"value": "v"}}
//     list: [{"key": "k", "value": "v"}]
// * Keys must be string typed.
// * Keys can't have multiple components.
//
// Optionally, maps may be atomic (for example, imagine representing an RGB
// color value--it doesn't make sense to have different actors own the R and G
// values).
//
// Maps may also represent a type which is composed of a number of different fields.
// Each field has a name and a type.
//
// Fields are indexed in a map before the first search so this type
// should be considered immutable.
type Map struct {
	// Each struct field appears exactly once in this list. The order in
	// this list defines the canonical field ordering.
	Fields []StructField `yaml:"fields,omitempty"`

	// A Union is a grouping of fields with special rules. It may refer to
	// one or more fields in the above list. A given field from the above
	// list may be referenced in exactly 0 or 1 places in the below list.
	// One can have multiple unions in the same struct, but the fields can't
	// overlap between unions.
	Unions []Union `yaml:"unions,omitempty"`

	// ElementType is the type of the structs's unknown fields.
	ElementType TypeRef `yaml:"elementType,omitempty"`

	// ElementRelationship states the relationship between the map's items.
	// * `separable` (or unset) implies that each element is 100% independent.
	// * `atomic` implies that all elements depend on each other, and this
	//   is effectively a scalar / leaf field; it doesn't make sense for
	//   separate actors to set the elements. Example: an RGB color struct;
	//   it would never make sense to "own" only one component of the
	//   color.
	// The default behavior for maps is `separable`; it's permitted to
	// leave this unset to get the default behavior.
	ElementRelationship ElementRelationship `yaml:"elementRelationship,omitempty"`

	once sync.Once
	m    map[string]StructField
}

// FindField is a convenience function that returns the referenced StructField,
// if it exists, or (nil, false) if it doesn't.
func (m *Map) FindField(name string) (StructField, bool) {
	m.once.Do(func() {
		m.m = make(map[string]StructField, len(m.Fields))
		for _, field := range m.Fields {
			m.m[field.Name] = field
		}
	})
	sf, ok := m.m[name]
	return sf, ok
}

// UnionFields are mapping between the fields that are part of the union and
// their discriminated value. The discriminated value has to be set, and
// should not conflict with other discriminated value in the list.
type UnionField struct {
	// FieldName is the name of the field that is part of the union. This
	// is the serialized form of the field.
	FieldName string `yaml:"fieldName"`
	// Discriminatorvalue is the value of the discriminator to
	// select that field. If the union doesn't have a discriminator,
	// this field is ignored.
	DiscriminatorValue string `yaml:"discriminatorValue"`
}

// Union, or oneof, means that only one of multiple fields of a structure can be
// set at a time. Setting the discriminator helps clearing oher fields:
// - If discriminator changed to non-nil, and a new field has been added
// that doesn't match, an error is returned,
// - If discriminator hasn't changed and two fields or more are set, an
// error is returned,
// - If discriminator changed to non-nil, all other fields but the
// discriminated one will be cleared,
// - Otherwise, If only one field is left, update discriminator to that value.
type Union struct {
	// Discriminator, if present, is the name of the field that
	// discriminates fields in the union. The mapping between the value of
	// the discriminator and the field is done by using the Fields list
	// below.
	Discriminator *string `yaml:"discriminator,omitempty"`

	// DeduceInvalidDiscriminator indicates if the discriminator
	// should be updated automatically based on the fields set. This
	// typically defaults to false since we don't want to deduce by
	// default (the behavior exists to maintain compatibility on
	// existing types and shouldn't be used for new types).
	DeduceInvalidDiscriminator bool `yaml:"deduceInvalidDiscriminator,omitempty"`

	// This is the list of fields that belong to this union. All the
	// fields present in here have to be part of the parent
	// structure. Discriminator (if oneOf has one), is NOT included in
	// this list. The value for field is how we map the name of the field
	// to actual value for discriminator.
	Fields []UnionField `yaml:"fields,omitempty"`
}

// StructField pairs a field name with a field type.
type StructField struct {
	// Name is the field name.
	Name string `yaml:"name,omitempty"`
	// Type is the field type.
	Type TypeRef `yaml:"type,omitempty"`
}

// List represents a type which contains a zero or more elements, all of the
// same subtype. Lists may be either associative: each element is more or less
// independent and could be managed by separate entities in the system; or
// atomic, where the elements are heavily dependent on each other: it is not
// sensible to change one element without considering the ramifications on all
// the other elements.
type List struct {
	// ElementType is the type of the list's elements.
	ElementType TypeRef `yaml:"elementType,omitempty"`

	// ElementRelationship states the relationship between the list's elements
	// and must have one of these values:
	// * `atomic`: the list is treated as a single entity, like a scalar.
	// * `associative`:
	//   - If the list element is a scalar, the list is treated as a set.
	//   - If the list element is a map, the list is treated as a map.
	// There is no default for this value for lists; all schemas must
	// explicitly state the element relationship for all lists.
	ElementRelationship ElementRelationship `yaml:"elementRelationship,omitempty"`

	// Iff ElementRelationship is `associative`, and the element type is
	// map, then Keys must have non-zero length, and it lists the fields
	// of the element's map type which are to be used as the keys of the
	// list.
	//
	// TODO: change this to "non-atomic struct" above and make the code reflect this.
	//
	// Each key must refer to a single field name (no nesting, not JSONPath).
	Keys []string `yaml:"keys,omitempty"`
}

// FindNamedType is a convenience function that returns the referenced TypeDef,
// if it exists, or (nil, false) if it doesn't.
func (s *Schema) FindNamedType(name string) (TypeDef, bool) {
	s.once.Do(func() {
		s.m = make(map[string]TypeDef, len(s.Types))
		for _, t := range s.Types {
			s.m[t.Name] = t
		}
	})
	t, ok := s.m[name]
	return t, ok
}

// Resolve is a convenience function which returns the atom referenced, whether
// it is inline or named. Returns (Atom{}, false) if the type can't be resolved.
//
// This allows callers to not care about the difference between a (possibly
// inlined) reference and a definition.
func (s *Schema) Resolve(tr TypeRef) (Atom, bool) {
	if tr.NamedType != nil {
		t, ok := s.FindNamedType(*tr.NamedType)
		if !ok {
			return Atom{}, false
		}
		return t.Atom, true
	}
	return tr.Inlined, true
}
