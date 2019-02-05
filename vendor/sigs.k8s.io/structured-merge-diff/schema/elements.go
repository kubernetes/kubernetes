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

// Schema is a list of named types.
type Schema struct {
	Types []TypeDef `yaml:"types,omitempty"`
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
type Atom struct {
	// Exactly one of the below must be set.
	*Scalar  `yaml:"scalar,omitempty"`
	*Struct  `yaml:"struct,omitempty"`
	*List    `yaml:"list,omitempty"`
	*Map     `yaml:"map,omitempty"`
	*Untyped `yaml:"untyped,omitempty"`
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
// between the elements of container types (maps, lists, structs, untyped).
type ElementRelationship string

const (
	// Associative only applies to lists (see the documentation there).
	Associative = ElementRelationship("associative")
	// Atomic makes container types (lists, maps, structs, untyped) behave
	// as scalars / leaf fields (which is the default for untyped data).
	Atomic = ElementRelationship("atomic")
	// Separable means the items of the container type have no particular
	// relationship (default behavior for maps and structs).
	Separable = ElementRelationship("separable")
)

// Struct represents a type which is composed of a number of different fields.
// Each field has a name and a type.
//
// TODO: in the future, we will add one-of groups (sometimes called unions).
type Struct struct {
	// Each struct field appears exactly once in this list. The order in
	// this list defines the canonical field ordering.
	Fields []StructField `yaml:"fields,omitempty"`

	// TODO: Implement unions, either this way or by inlining.
	// Unions are groupings of fields with special rules. They may refer to
	// one or more fields in the above list. A given field from the above
	// list may be referenced in exactly 0 or 1 places in the below list.
	// Unions []Union `yaml:"unions,omitempty"`

	// ElementRelationship states the relationship between the struct's items.
	// * `separable` (or unset) implies that each element is 100% independent.
	// * `atomic` implies that all elements depend on each other, and this
	//   is effectively a scalar / leaf field; it doesn't make sense for
	//   separate actors to set the elements. Example: an RGB color struct;
	//   it would never make sense to "own" only one component of the
	//   color.
	// The default behavior for structs is `separable`; it's permitted to
	// leave this unset to get the default behavior.
	ElementRelationship ElementRelationship `yaml:"elementRelationship,omitempty"`
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
	//   - If the list element is a struct, the list is treated as a map.
	//   - The list element must not be a map or a list itself.
	// There is no default for this value for lists; all schemas must
	// explicitly state the element relationship for all lists.
	ElementRelationship ElementRelationship `yaml:"elementRelationship,omitempty"`

	// Iff ElementRelationship is `associative`, and the element type is
	// struct, then Keys must have non-zero length, and it lists the fields
	// of the element's struct type which are to be used as the keys of the
	// list.
	//
	// TODO: change this to "non-atomic struct" above and make the code reflect this.
	//
	// Each key must refer to a single field name (no nesting, not JSONPath).
	Keys []string `yaml:"keys,omitempty"`
}

// Map is a key-value pair. Its default semantics are the same as an
// associative list, but:
// * It is serialized differently:
//     map:  {"k": {"value": "v"}}
//     list: [{"key": "k", "value": "v"}]
// * Keys must be string typed.
// * Keys can't have multiple components.
//
// Although serialized the same, maps are different from structs in that each
// map item must have the same type.
//
// Optionally, maps may be atomic (for example, imagine representing an RGB
// color value--it doesn't make sense to have different actors own the R and G
// values).
type Map struct {
	// ElementType is the type of the list's elements.
	ElementType TypeRef `yaml:"elementType,omitempty"`

	// ElementRelationship states the relationship between the map's items.
	// * `separable` implies that each element is 100% independent.
	// * `atomic` implies that all elements depend on each other, and this
	//   is effectively a scalar / leaf field; it doesn't make sense for
	//   separate actors to set the elements.
	//   TODO: find a simple example.
	// The default behavior for maps is `separable`; it's permitted to
	// leave this unset to get the default behavior.
	ElementRelationship ElementRelationship `yaml:"elementRelationship,omitempty"`
}

// Untyped represents types that allow arbitrary content. (Think: plugin
// objects.)
type Untyped struct {
	// ElementRelationship states the relationship between the items, if
	// container-typed data happens to be present here.
	// * `atomic` implies that all elements depend on each other, and this
	//   is effectively a scalar / leaf field; it doesn't make sense for
	//   separate actors to set the elements.
	// TODO: support "guess" (guesses at associative list keys)
	// TODO: support "lookup" (calls a lookup function to figure out the
	//       schema based on the data)
	// The default behavior for untyped data is `atomic`; it's permitted to
	// leave this unset to get the default behavior.
	ElementRelationship ElementRelationship `yaml:"elementRelationship,omitempty"`
}

// FindNamedType is a convenience function that returns the referenced TypeDef,
// if it exists, or (nil, false) if it doesn't.
func (s Schema) FindNamedType(name string) (TypeDef, bool) {
	for _, t := range s.Types {
		if t.Name == name {
			return t, true
		}
	}
	return TypeDef{}, false
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
