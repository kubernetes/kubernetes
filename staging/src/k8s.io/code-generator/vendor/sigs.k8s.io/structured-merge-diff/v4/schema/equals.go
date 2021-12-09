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

import "reflect"

// Equals returns true iff the two Schemas are equal.
func (a *Schema) Equals(b *Schema) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}

	if len(a.Types) != len(b.Types) {
		return false
	}
	for i := range a.Types {
		if !a.Types[i].Equals(&b.Types[i]) {
			return false
		}
	}
	return true
}

// Equals returns true iff the two TypeRefs are equal.
//
// Note that two typerefs that have an equivalent type but where one is
// inlined and the other is named, are not considered equal.
func (a *TypeRef) Equals(b *TypeRef) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if (a.NamedType == nil) != (b.NamedType == nil) {
		return false
	}
	if a.NamedType != nil {
		if *a.NamedType != *b.NamedType {
			return false
		}
		//return true
	}
	return a.Inlined.Equals(&b.Inlined)
}

// Equals returns true iff the two TypeDefs are equal.
func (a *TypeDef) Equals(b *TypeDef) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a.Name != b.Name {
		return false
	}
	return a.Atom.Equals(&b.Atom)
}

// Equals returns true iff the two Atoms are equal.
func (a *Atom) Equals(b *Atom) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if (a.Scalar == nil) != (b.Scalar == nil) {
		return false
	}
	if (a.List == nil) != (b.List == nil) {
		return false
	}
	if (a.Map == nil) != (b.Map == nil) {
		return false
	}
	switch {
	case a.Scalar != nil:
		return *a.Scalar == *b.Scalar
	case a.List != nil:
		return a.List.Equals(b.List)
	case a.Map != nil:
		return a.Map.Equals(b.Map)
	}
	return true
}

// Equals returns true iff the two Maps are equal.
func (a *Map) Equals(b *Map) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if !a.ElementType.Equals(&b.ElementType) {
		return false
	}
	if a.ElementRelationship != b.ElementRelationship {
		return false
	}
	if len(a.Fields) != len(b.Fields) {
		return false
	}
	for i := range a.Fields {
		if !a.Fields[i].Equals(&b.Fields[i]) {
			return false
		}
	}
	if len(a.Unions) != len(b.Unions) {
		return false
	}
	for i := range a.Unions {
		if !a.Unions[i].Equals(&b.Unions[i]) {
			return false
		}
	}
	return true
}

// Equals returns true iff the two Unions are equal.
func (a *Union) Equals(b *Union) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if (a.Discriminator == nil) != (b.Discriminator == nil) {
		return false
	}
	if a.Discriminator != nil {
		if *a.Discriminator != *b.Discriminator {
			return false
		}
	}
	if a.DeduceInvalidDiscriminator != b.DeduceInvalidDiscriminator {
		return false
	}
	if len(a.Fields) != len(b.Fields) {
		return false
	}
	for i := range a.Fields {
		if !a.Fields[i].Equals(&b.Fields[i]) {
			return false
		}
	}
	return true
}

// Equals returns true iff the two UnionFields are equal.
func (a *UnionField) Equals(b *UnionField) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a.FieldName != b.FieldName {
		return false
	}
	if a.DiscriminatorValue != b.DiscriminatorValue {
		return false
	}
	return true
}

// Equals returns true iff the two StructFields are equal.
func (a *StructField) Equals(b *StructField) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a.Name != b.Name {
		return false
	}
	if !reflect.DeepEqual(a.Default, b.Default) {
		return false
	}
	return a.Type.Equals(&b.Type)
}

// Equals returns true iff the two Lists are equal.
func (a *List) Equals(b *List) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if !a.ElementType.Equals(&b.ElementType) {
		return false
	}
	if a.ElementRelationship != b.ElementRelationship {
		return false
	}
	if len(a.Keys) != len(b.Keys) {
		return false
	}
	for i := range a.Keys {
		if a.Keys[i] != b.Keys[i] {
			return false
		}
	}
	return true
}
