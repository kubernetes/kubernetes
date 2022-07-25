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
	"fmt"
	"strings"

	"github.com/google/cel-go/checker/decls"

	"google.golang.org/protobuf/proto"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

const (
	kindUnknown = iota + 1
	kindError
	kindFunction
	kindDyn
	kindPrimitive
	kindWellKnown
	kindWrapper
	kindNull
	kindAbstract
	kindType
	kindList
	kindMap
	kindObject
	kindTypeParam
)

// FormatCheckedType converts a type message into a string representation.
func FormatCheckedType(t *exprpb.Type) string {
	switch kindOf(t) {
	case kindDyn:
		return "dyn"
	case kindFunction:
		return formatFunction(t.GetFunction().GetResultType(),
			t.GetFunction().GetArgTypes(),
			false)
	case kindList:
		return fmt.Sprintf("list(%s)", FormatCheckedType(t.GetListType().ElemType))
	case kindObject:
		return t.GetMessageType()
	case kindMap:
		return fmt.Sprintf("map(%s, %s)",
			FormatCheckedType(t.GetMapType().KeyType),
			FormatCheckedType(t.GetMapType().ValueType))
	case kindNull:
		return "null"
	case kindPrimitive:
		switch t.GetPrimitive() {
		case exprpb.Type_UINT64:
			return "uint"
		case exprpb.Type_INT64:
			return "int"
		}
		return strings.Trim(strings.ToLower(t.GetPrimitive().String()), " ")
	case kindType:
		if t.GetType() == nil {
			return "type"
		}
		return fmt.Sprintf("type(%s)", FormatCheckedType(t.GetType()))
	case kindWellKnown:
		switch t.GetWellKnown() {
		case exprpb.Type_ANY:
			return "any"
		case exprpb.Type_DURATION:
			return "duration"
		case exprpb.Type_TIMESTAMP:
			return "timestamp"
		}
	case kindWrapper:
		return fmt.Sprintf("wrapper(%s)",
			FormatCheckedType(decls.NewPrimitiveType(t.GetWrapper())))
	case kindError:
		return "!error!"
	case kindTypeParam:
		return t.GetTypeParam()
	}
	return t.String()
}

// isDyn returns true if the input t is either type DYN or a well-known ANY message.
func isDyn(t *exprpb.Type) bool {
	// Note: object type values that are well-known and map to a DYN value in practice
	// are sanitized prior to being added to the environment.
	switch kindOf(t) {
	case kindDyn:
		return true
	case kindWellKnown:
		return t.GetWellKnown() == exprpb.Type_ANY
	default:
		return false
	}
}

// isDynOrError returns true if the input is either an Error, DYN, or well-known ANY message.
func isDynOrError(t *exprpb.Type) bool {
	switch kindOf(t) {
	case kindError:
		return true
	default:
		return isDyn(t)
	}
}

// isEqualOrLessSpecific checks whether one type is equal or less specific than the other one.
// A type is less specific if it matches the other type using the DYN type.
func isEqualOrLessSpecific(t1 *exprpb.Type, t2 *exprpb.Type) bool {
	kind1, kind2 := kindOf(t1), kindOf(t2)
	// The first type is less specific.
	if isDyn(t1) || kind1 == kindTypeParam {
		return true
	}
	// The first type is not less specific.
	if isDyn(t2) || kind2 == kindTypeParam {
		return false
	}
	// Types must be of the same kind to be equal.
	if kind1 != kind2 {
		return false
	}

	// With limited exceptions for ANY and JSON values, the types must agree and be equivalent in
	// order to return true.
	switch kind1 {
	case kindAbstract:
		a1 := t1.GetAbstractType()
		a2 := t2.GetAbstractType()
		if a1.GetName() != a2.GetName() ||
			len(a1.GetParameterTypes()) != len(a2.GetParameterTypes()) {
			return false
		}
		for i, p1 := range a1.GetParameterTypes() {
			if !isEqualOrLessSpecific(p1, a2.GetParameterTypes()[i]) {
				return false
			}
		}
		return true
	case kindList:
		return isEqualOrLessSpecific(t1.GetListType().ElemType, t2.GetListType().ElemType)
	case kindMap:
		m1 := t1.GetMapType()
		m2 := t2.GetMapType()
		return isEqualOrLessSpecific(m1.KeyType, m2.KeyType) &&
			isEqualOrLessSpecific(m1.ValueType, m2.ValueType)
	case kindType:
		return true
	default:
		return proto.Equal(t1, t2)
	}
}

/// internalIsAssignable returns true if t1 is assignable to t2.
func internalIsAssignable(m *mapping, t1 *exprpb.Type, t2 *exprpb.Type) bool {
	// Process type parameters.
	kind1, kind2 := kindOf(t1), kindOf(t2)
	if kind2 == kindTypeParam {
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
	if kind1 == kindTypeParam {
		// Return whether t1 is a valid substitution for t2. If not, do no additional checks as the
		// possible type substitutions have been searched in both directions.
		valid, _ := isValidTypeSubstitution(m, t2, t1)
		return valid
	}

	// Next check for wildcard types.
	if isDynOrError(t1) || isDynOrError(t2) {
		return true
	}

	// Test for when the types do not need to agree, but are more specific than dyn.
	switch kind1 {
	case kindNull:
		return internalIsAssignableNull(t2)
	case kindPrimitive:
		return internalIsAssignablePrimitive(t1.GetPrimitive(), t2)
	case kindWrapper:
		return internalIsAssignable(m, decls.NewPrimitiveType(t1.GetWrapper()), t2)
	default:
		if kind1 != kind2 {
			return false
		}
	}

	// Test for when the types must agree.
	switch kind1 {
	// ERROR, TYPE_PARAM, and DYN handled above.
	case kindAbstract:
		return internalIsAssignableAbstractType(m, t1.GetAbstractType(), t2.GetAbstractType())
	case kindFunction:
		return internalIsAssignableFunction(m, t1.GetFunction(), t2.GetFunction())
	case kindList:
		return internalIsAssignable(m, t1.GetListType().GetElemType(), t2.GetListType().GetElemType())
	case kindMap:
		return internalIsAssignableMap(m, t1.GetMapType(), t2.GetMapType())
	case kindObject:
		return t1.GetMessageType() == t2.GetMessageType()
	case kindType:
		// A type is a type is a type, any additional parameterization of the
		// type cannot affect method resolution or assignability.
		return true
	case kindWellKnown:
		return t1.GetWellKnown() == t2.GetWellKnown()
	default:
		return false
	}
}

// isValidTypeSubstitution returns whether t2 (or its type substitution) is a valid type
// substitution for t1, and whether t2 has a type substitution in mapping m.
//
// The type t2 is a valid substitution for t1 if any of the following statements is true
// - t2 has a type substitition (t2sub) equal to t1
// - t2 has a type substitution (t2sub) assignable to t1
// - t2 does not occur within t1.
func isValidTypeSubstitution(m *mapping, t1, t2 *exprpb.Type) (valid, hasSub bool) {
	// Early return if the t1 and t2 are the same instance.
	kind1, kind2 := kindOf(t1), kindOf(t2)
	if kind1 == kind2 && (t1 == t2 || proto.Equal(t1, t2)) {
		return true, true
	}
	if t2Sub, found := m.find(t2); found {
		// Early return if t1 and t2Sub are the same instance as otherwise the mapping
		// might mark a type as being a subtitution for itself.
		if kind1 == kindOf(t2Sub) && (t1 == t2Sub || proto.Equal(t1, t2Sub)) {
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

// internalIsAssignableAbstractType returns true if the abstract type names agree and all type
// parameters are assignable.
func internalIsAssignableAbstractType(m *mapping,
	a1 *exprpb.Type_AbstractType,
	a2 *exprpb.Type_AbstractType) bool {
	return a1.GetName() == a2.GetName() &&
		internalIsAssignableList(m, a1.GetParameterTypes(), a2.GetParameterTypes())
}

// internalIsAssignableFunction returns true if the function return type and arg types are
// assignable.
func internalIsAssignableFunction(m *mapping,
	f1 *exprpb.Type_FunctionType,
	f2 *exprpb.Type_FunctionType) bool {
	f1ArgTypes := flattenFunctionTypes(f1)
	f2ArgTypes := flattenFunctionTypes(f2)
	if internalIsAssignableList(m, f1ArgTypes, f2ArgTypes) {
		return true
	}
	return false
}

// internalIsAssignableList returns true if the element types at each index in the list are
// assignable from l1[i] to l2[i]. The list lengths must also agree for the lists to be
// assignable.
func internalIsAssignableList(m *mapping, l1 []*exprpb.Type, l2 []*exprpb.Type) bool {
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

// internalIsAssignableMap returns true if map m1 may be assigned to map m2.
func internalIsAssignableMap(m *mapping, m1 *exprpb.Type_MapType, m2 *exprpb.Type_MapType) bool {
	if internalIsAssignableList(m,
		[]*exprpb.Type{m1.GetKeyType(), m1.GetValueType()},
		[]*exprpb.Type{m2.GetKeyType(), m2.GetValueType()}) {
		return true
	}
	return false
}

// internalIsAssignableNull returns true if the type is nullable.
func internalIsAssignableNull(t *exprpb.Type) bool {
	switch kindOf(t) {
	case kindAbstract, kindObject, kindNull, kindWellKnown, kindWrapper:
		return true
	default:
		return false
	}
}

// internalIsAssignablePrimitive returns true if the target type is the same or if it is a wrapper
// for the primitive type.
func internalIsAssignablePrimitive(p exprpb.Type_PrimitiveType, target *exprpb.Type) bool {
	switch kindOf(target) {
	case kindPrimitive:
		return p == target.GetPrimitive()
	case kindWrapper:
		return p == target.GetWrapper()
	default:
		return false
	}
}

// isAssignable returns an updated type substitution mapping if t1 is assignable to t2.
func isAssignable(m *mapping, t1 *exprpb.Type, t2 *exprpb.Type) *mapping {
	mCopy := m.copy()
	if internalIsAssignable(mCopy, t1, t2) {
		return mCopy
	}
	return nil
}

// isAssignableList returns an updated type substitution mapping if l1 is assignable to l2.
func isAssignableList(m *mapping, l1 []*exprpb.Type, l2 []*exprpb.Type) *mapping {
	mCopy := m.copy()
	if internalIsAssignableList(mCopy, l1, l2) {
		return mCopy
	}
	return nil
}

// kindOf returns the kind of the type as defined in the checked.proto.
func kindOf(t *exprpb.Type) int {
	if t == nil || t.TypeKind == nil {
		return kindUnknown
	}
	switch t.TypeKind.(type) {
	case *exprpb.Type_Error:
		return kindError
	case *exprpb.Type_Function:
		return kindFunction
	case *exprpb.Type_Dyn:
		return kindDyn
	case *exprpb.Type_Primitive:
		return kindPrimitive
	case *exprpb.Type_WellKnown:
		return kindWellKnown
	case *exprpb.Type_Wrapper:
		return kindWrapper
	case *exprpb.Type_Null:
		return kindNull
	case *exprpb.Type_Type:
		return kindType
	case *exprpb.Type_ListType_:
		return kindList
	case *exprpb.Type_MapType_:
		return kindMap
	case *exprpb.Type_MessageType:
		return kindObject
	case *exprpb.Type_TypeParam:
		return kindTypeParam
	case *exprpb.Type_AbstractType_:
		return kindAbstract
	}
	return kindUnknown
}

// mostGeneral returns the more general of two types which are known to unify.
func mostGeneral(t1 *exprpb.Type, t2 *exprpb.Type) *exprpb.Type {
	if isEqualOrLessSpecific(t1, t2) {
		return t1
	}
	return t2
}

// notReferencedIn checks whether the type doesn't appear directly or transitively within the other
// type. This is a standard requirement for type unification, commonly referred to as the "occurs
// check".
func notReferencedIn(m *mapping, t *exprpb.Type, withinType *exprpb.Type) bool {
	if proto.Equal(t, withinType) {
		return false
	}
	withinKind := kindOf(withinType)
	switch withinKind {
	case kindTypeParam:
		wtSub, found := m.find(withinType)
		if !found {
			return true
		}
		return notReferencedIn(m, t, wtSub)
	case kindAbstract:
		for _, pt := range withinType.GetAbstractType().GetParameterTypes() {
			if !notReferencedIn(m, t, pt) {
				return false
			}
		}
		return true
	case kindList:
		return notReferencedIn(m, t, withinType.GetListType().ElemType)
	case kindMap:
		mt := withinType.GetMapType()
		return notReferencedIn(m, t, mt.KeyType) && notReferencedIn(m, t, mt.ValueType)
	case kindWrapper:
		return notReferencedIn(m, t, decls.NewPrimitiveType(withinType.GetWrapper()))
	default:
		return true
	}
}

// substitute replaces all direct and indirect occurrences of bound type parameters. Unbound type
// parameters are replaced by DYN if typeParamToDyn is true.
func substitute(m *mapping, t *exprpb.Type, typeParamToDyn bool) *exprpb.Type {
	if tSub, found := m.find(t); found {
		return substitute(m, tSub, typeParamToDyn)
	}
	kind := kindOf(t)
	if typeParamToDyn && kind == kindTypeParam {
		return decls.Dyn
	}
	switch kind {
	case kindAbstract:
		at := t.GetAbstractType()
		params := make([]*exprpb.Type, len(at.GetParameterTypes()))
		for i, p := range at.GetParameterTypes() {
			params[i] = substitute(m, p, typeParamToDyn)
		}
		return decls.NewAbstractType(at.GetName(), params...)
	case kindFunction:
		fn := t.GetFunction()
		rt := substitute(m, fn.ResultType, typeParamToDyn)
		args := make([]*exprpb.Type, len(fn.ArgTypes))
		for i, a := range fn.ArgTypes {
			args[i] = substitute(m, a, typeParamToDyn)
		}
		return decls.NewFunctionType(rt, args...)
	case kindList:
		return decls.NewListType(substitute(m, t.GetListType().ElemType, typeParamToDyn))
	case kindMap:
		mt := t.GetMapType()
		return decls.NewMapType(substitute(m, mt.KeyType, typeParamToDyn),
			substitute(m, mt.ValueType, typeParamToDyn))
	case kindType:
		if t.GetType() != nil {
			return decls.NewTypeType(substitute(m, t.GetType(), typeParamToDyn))
		}
		return t
	default:
		return t
	}
}

func typeKey(t *exprpb.Type) string {
	return FormatCheckedType(t)
}

// flattenFunctionTypes takes a function with arg types T1, T2, ..., TN and result type TR
// and returns a slice containing {T1, T2, ..., TN, TR}.
func flattenFunctionTypes(f *exprpb.Type_FunctionType) []*exprpb.Type {
	argTypes := f.GetArgTypes()
	if len(argTypes) == 0 {
		return []*exprpb.Type{f.GetResultType()}
	}
	flattend := make([]*exprpb.Type, len(argTypes)+1, len(argTypes)+1)
	for i, at := range argTypes {
		flattend[i] = at
	}
	flattend[len(argTypes)] = f.GetResultType()
	return flattend
}
