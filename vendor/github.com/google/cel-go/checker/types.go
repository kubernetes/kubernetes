// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package checker

import (
	"github.com/google/cel-go/common/types"
)

// isDyn returns true if the input t is either type DYN or a well-known ANY message.
func isDyn(t *types.Type) bool {
	// Note: object type values that are well-known and map to a DYN value in practice
	// are sanitized prior to being added to the environment.
	switch t.Kind() {
	case types.DynKind, types.AnyKind:
		return true
	default:
		return false
	}
}

// isDynOrError returns true if the input is either an Error, DYN, or well-known ANY message.
func isDynOrError(t *types.Type) bool {
	return isError(t) || isDyn(t)
}

func isError(t *types.Type) bool {
	return t.Kind() == types.ErrorKind
}

func isOptional(t *types.Type) bool {
	if t.Kind() == types.OpaqueKind {
		return t.TypeName() == "optional"
	}
	return false
}

func maybeUnwrapOptional(t *types.Type) (*types.Type, bool) {
	if isOptional(t) {
		return t.Parameters()[0], true
	}
	return t, false
}

// isEqualOrLessSpecific checks whether one type is equal or less specific than the other one.
// A type is less specific if it matches the other type using the DYN type.
func isEqualOrLessSpecific(t1, t2 *types.Type) bool {
	kind1, kind2 := t1.Kind(), t2.Kind()
	// The first type is less specific.
	if isDyn(t1) || kind1 == types.TypeParamKind {
		return true
	}
	// The first type is not less specific.
	if isDyn(t2) || kind2 == types.TypeParamKind {
		return false
	}
	// Types must be of the same kind to be equal.
	if kind1 != kind2 {
		return false
	}

	// With limited exceptions for ANY and JSON values, the types must agree and be equivalent in
	// order to return true.
	switch kind1 {
	case types.OpaqueKind:
		if t1.TypeName() != t2.TypeName() ||
			len(t1.Parameters()) != len(t2.Parameters()) {
			return false
		}
		for i, p1 := range t1.Parameters() {
			if !isEqualOrLessSpecific(p1, t2.Parameters()[i]) {
				return false
			}
		}
		return true
	case types.ListKind:
		return isEqualOrLessSpecific(t1.Parameters()[0], t2.Parameters()[0])
	case types.MapKind:
		return isEqualOrLessSpecific(t1.Parameters()[0], t2.Parameters()[0]) &&
			isEqualOrLessSpecific(t1.Parameters()[1], t2.Parameters()[1])
	case types.TypeKind:
		return true
	default:
		return t1.IsExactType(t2)
	}
}

// / internalIsAssignable returns true if t1 is assignable to t2.
func internalIsAssignable(m *mapping, t1, t2 *types.Type) bool {
	// Process type parameters.
	kind1, kind2 := t1.Kind(), t2.Kind()
	if kind2 == types.TypeParamKind {
		// If t2 is a valid type substitution for t1, return true.
		valid, t2HasSub := isValidTypeSubstitution(m, t1, t2)
		if valid {
			return true
		}
		// If t2 is not a valid type sub for t1, and already has a known substitution return false
		// since it is not possible for t1 to be a substitution for t2.
		if !valid && t2HasSub {
			return false
		}
		// Otherwise, fall through to check whether t1 is a possible substitution for t2.
	}
	if kind1 == types.TypeParamKind {
		// Return whether t1 is a valid substitution for t2. If not, do no additional checks as the
		// possible type substitutions have been searched in both directions.
		valid, _ := isValidTypeSubstitution(m, t2, t1)
		return valid
	}

	// Next check for wildcard types.
	if isDynOrError(t1) || isDynOrError(t2) {
		return true
	}
	// Preserve the nullness checks of the legacy type-checker.
	if kind1 == types.NullTypeKind {
		return internalIsAssignableNull(t2)
	}
	if kind2 == types.NullTypeKind {
		return internalIsAssignableNull(t1)
	}

	// Test for when the types do not need to agree, but are more specific than dyn.
	switch kind1 {
	case types.BoolKind, types.BytesKind, types.DoubleKind, types.IntKind, types.StringKind, types.UintKind,
		types.AnyKind, types.DurationKind, types.TimestampKind,
		types.StructKind:
		return t1.IsAssignableType(t2)
	case types.TypeKind:
		return kind2 == types.TypeKind
	case types.OpaqueKind, types.ListKind, types.MapKind:
		return t1.Kind() == t2.Kind() && t1.TypeName() == t2.TypeName() &&
			internalIsAssignableList(m, t1.Parameters(), t2.Parameters())
	default:
		return false
	}
}

// isValidTypeSubstitution returns whether t2 (or its type substitution) is a valid type
// substitution for t1, and whether t2 has a type substitution in mapping m.
//
// The type t2 is a valid substitution for t1 if any of the following statements is true
// - t2 has a type substitution (t2sub) equal to t1
// - t2 has a type substitution (t2sub) assignable to t1
// - t2 does not occur within t1.
func isValidTypeSubstitution(m *mapping, t1, t2 *types.Type) (valid, hasSub bool) {
	// Early return if the t1 and t2 are the same instance.
	kind1, kind2 := t1.Kind(), t2.Kind()
	if kind1 == kind2 && t1.IsExactType(t2) {
		return true, true
	}
	if t2Sub, found := m.find(t2); found {
		// Early return if t1 and t2Sub are the same instance as otherwise the mapping
		// might mark a type as being a subtitution for itself.
		if kind1 == t2Sub.Kind() && t1.IsExactType(t2Sub) {
			return true, true
		}
		// If the types are compatible, pick the more general type and return true
		if internalIsAssignable(m, t1, t2Sub) {
			t2New := mostGeneral(t1, t2Sub)
			// only update the type reference map if the target type does not occur within it.
			if notReferencedIn(m, t2, t2New) {
				m.add(t2, t2New)
			}
			// acknowledge the type agreement, and that the substitution is already tracked.
			return true, true
		}
		return false, true
	}
	if notReferencedIn(m, t2, t1) {
		m.add(t2, t1)
		return true, false
	}
	return false, false
}

// internalIsAssignableList returns true if the element types at each index in the list are
// assignable from l1[i] to l2[i]. The list lengths must also agree for the lists to be
// assignable.
func internalIsAssignableList(m *mapping, l1, l2 []*types.Type) bool {
	if len(l1) != len(l2) {
		return false
	}
	for i, t1 := range l1 {
		if !internalIsAssignable(m, t1, l2[i]) {
			return false
		}
	}
	return true
}

// internalIsAssignableNull returns true if the type is nullable.
func internalIsAssignableNull(t *types.Type) bool {
	return isLegacyNullable(t) || t.IsAssignableType(types.NullType)
}

// isLegacyNullable preserves the null-ness compatibility of the original type-checker implementation.
func isLegacyNullable(t *types.Type) bool {
	switch t.Kind() {
	case types.OpaqueKind, types.StructKind, types.AnyKind, types.DurationKind, types.TimestampKind:
		return true
	}
	return false
}

// isAssignable returns an updated type substitution mapping if t1 is assignable to t2.
func isAssignable(m *mapping, t1, t2 *types.Type) *mapping {
	mCopy := m.copy()
	if internalIsAssignable(mCopy, t1, t2) {
		return mCopy
	}
	return nil
}

// isAssignableList returns an updated type substitution mapping if l1 is assignable to l2.
func isAssignableList(m *mapping, l1, l2 []*types.Type) *mapping {
	mCopy := m.copy()
	if internalIsAssignableList(mCopy, l1, l2) {
		return mCopy
	}
	return nil
}

// mostGeneral returns the more general of two types which are known to unify.
func mostGeneral(t1, t2 *types.Type) *types.Type {
	if isEqualOrLessSpecific(t1, t2) {
		return t1
	}
	return t2
}

// notReferencedIn checks whether the type doesn't appear directly or transitively within the other
// type. This is a standard requirement for type unification, commonly referred to as the "occurs
// check".
func notReferencedIn(m *mapping, t, withinType *types.Type) bool {
	if t.IsExactType(withinType) {
		return false
	}
	withinKind := withinType.Kind()
	switch withinKind {
	case types.TypeParamKind:
		wtSub, found := m.find(withinType)
		if !found {
			return true
		}
		return notReferencedIn(m, t, wtSub)
	case types.OpaqueKind, types.ListKind, types.MapKind:
		for _, pt := range withinType.Parameters() {
			if !notReferencedIn(m, t, pt) {
				return false
			}
		}
		return true
	default:
		return true
	}
}

// substitute replaces all direct and indirect occurrences of bound type parameters. Unbound type
// parameters are replaced by DYN if typeParamToDyn is true.
func substitute(m *mapping, t *types.Type, typeParamToDyn bool) *types.Type {
	if tSub, found := m.find(t); found {
		return substitute(m, tSub, typeParamToDyn)
	}
	kind := t.Kind()
	if typeParamToDyn && kind == types.TypeParamKind {
		return types.DynType
	}
	switch kind {
	case types.OpaqueKind:
		return types.NewOpaqueType(t.TypeName(), substituteParams(m, t.Parameters(), typeParamToDyn)...)
	case types.ListKind:
		return types.NewListType(substitute(m, t.Parameters()[0], typeParamToDyn))
	case types.MapKind:
		return types.NewMapType(substitute(m, t.Parameters()[0], typeParamToDyn),
			substitute(m, t.Parameters()[1], typeParamToDyn))
	case types.TypeKind:
		if len(t.Parameters()) > 0 {
			return types.NewTypeTypeWithParam(substitute(m, t.Parameters()[0], typeParamToDyn))
		}
		return t
	default:
		return t
	}
}

func substituteParams(m *mapping, typeParams []*types.Type, typeParamToDyn bool) []*types.Type {
	subParams := make([]*types.Type, len(typeParams))
	for i, tp := range typeParams {
		subParams[i] = substitute(m, tp, typeParamToDyn)
	}
	return subParams
}

func newFunctionType(resultType *types.Type, argTypes ...*types.Type) *types.Type {
	return types.NewOpaqueType("function", append([]*types.Type{resultType}, argTypes...)...)
}
