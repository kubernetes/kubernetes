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

// Package decls provides helpers for creating variable and function declarations.
package decls

import (
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	emptypb "google.golang.org/protobuf/types/known/emptypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

var (
	// Error type used to communicate issues during type-checking.
	Error = &exprpb.Type{
		TypeKind: &exprpb.Type_Error{
			Error: &emptypb.Empty{}}}

	// Dyn is a top-type used to represent any value.
	Dyn = &exprpb.Type{
		TypeKind: &exprpb.Type_Dyn{
			Dyn: &emptypb.Empty{}}}
)

// Commonly used types.
var (
	Bool   = NewPrimitiveType(exprpb.Type_BOOL)
	Bytes  = NewPrimitiveType(exprpb.Type_BYTES)
	Double = NewPrimitiveType(exprpb.Type_DOUBLE)
	Int    = NewPrimitiveType(exprpb.Type_INT64)
	Null   = &exprpb.Type{
		TypeKind: &exprpb.Type_Null{
			Null: structpb.NullValue_NULL_VALUE}}
	String = NewPrimitiveType(exprpb.Type_STRING)
	Uint   = NewPrimitiveType(exprpb.Type_UINT64)
)

// Well-known types.
// TODO: Replace with an abstract type registry.
var (
	Any       = NewWellKnownType(exprpb.Type_ANY)
	Duration  = NewWellKnownType(exprpb.Type_DURATION)
	Timestamp = NewWellKnownType(exprpb.Type_TIMESTAMP)
)

// NewAbstractType creates an abstract type declaration which references a proto
// message name and may also include type parameters.
func NewAbstractType(name string, paramTypes ...*exprpb.Type) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_AbstractType_{
			AbstractType: &exprpb.Type_AbstractType{
				Name:           name,
				ParameterTypes: paramTypes}}}
}

// NewOptionalType constructs an abstract type indicating that the parameterized type
// may be contained within the object.
func NewOptionalType(paramType *exprpb.Type) *exprpb.Type {
	return NewAbstractType("optional_type", paramType)
}

// NewFunctionType creates a function invocation contract, typically only used
// by type-checking steps after overload resolution.
func NewFunctionType(resultType *exprpb.Type,
	argTypes ...*exprpb.Type) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Function{
			Function: &exprpb.Type_FunctionType{
				ResultType: resultType,
				ArgTypes:   argTypes}}}
}

// NewFunction creates a named function declaration with one or more overloads.
func NewFunction(name string,
	overloads ...*exprpb.Decl_FunctionDecl_Overload) *exprpb.Decl {
	return &exprpb.Decl{
		Name: name,
		DeclKind: &exprpb.Decl_Function{
			Function: &exprpb.Decl_FunctionDecl{
				Overloads: overloads}}}
}

// NewIdent creates a named identifier declaration with an optional literal
// value.
//
// Literal values are typically only associated with enum identifiers.
//
// Deprecated: Use NewVar or NewConst instead.
func NewIdent(name string, t *exprpb.Type, v *exprpb.Constant) *exprpb.Decl {
	return &exprpb.Decl{
		Name: name,
		DeclKind: &exprpb.Decl_Ident{
			Ident: &exprpb.Decl_IdentDecl{
				Type:  t,
				Value: v}}}
}

// NewConst creates a constant identifier with a CEL constant literal value.
func NewConst(name string, t *exprpb.Type, v *exprpb.Constant) *exprpb.Decl {
	return NewIdent(name, t, v)
}

// NewVar creates a variable identifier.
func NewVar(name string, t *exprpb.Type) *exprpb.Decl {
	return NewIdent(name, t, nil)
}

// NewInstanceOverload creates a instance function overload contract.
// First element of argTypes is instance.
func NewInstanceOverload(id string, argTypes []*exprpb.Type,
	resultType *exprpb.Type) *exprpb.Decl_FunctionDecl_Overload {
	return &exprpb.Decl_FunctionDecl_Overload{
		OverloadId:         id,
		ResultType:         resultType,
		Params:             argTypes,
		IsInstanceFunction: true}
}

// NewListType generates a new list with elements of a certain type.
func NewListType(elem *exprpb.Type) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_ListType_{
			ListType: &exprpb.Type_ListType{
				ElemType: elem}}}
}

// NewMapType generates a new map with typed keys and values.
func NewMapType(key *exprpb.Type, value *exprpb.Type) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_MapType_{
			MapType: &exprpb.Type_MapType{
				KeyType:   key,
				ValueType: value}}}
}

// NewObjectType creates an object type for a qualified type name.
func NewObjectType(typeName string) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_MessageType{
			MessageType: typeName}}
}

// NewOverload creates a function overload declaration which contains a unique
// overload id as well as the expected argument and result types. Overloads
// must be aggregated within a Function declaration.
func NewOverload(id string, argTypes []*exprpb.Type,
	resultType *exprpb.Type) *exprpb.Decl_FunctionDecl_Overload {
	return &exprpb.Decl_FunctionDecl_Overload{
		OverloadId:         id,
		ResultType:         resultType,
		Params:             argTypes,
		IsInstanceFunction: false}
}

// NewParameterizedInstanceOverload creates a parametric function instance overload type.
func NewParameterizedInstanceOverload(id string,
	argTypes []*exprpb.Type,
	resultType *exprpb.Type,
	typeParams []string) *exprpb.Decl_FunctionDecl_Overload {
	return &exprpb.Decl_FunctionDecl_Overload{
		OverloadId:         id,
		ResultType:         resultType,
		Params:             argTypes,
		TypeParams:         typeParams,
		IsInstanceFunction: true}
}

// NewParameterizedOverload creates a parametric function overload type.
func NewParameterizedOverload(id string,
	argTypes []*exprpb.Type,
	resultType *exprpb.Type,
	typeParams []string) *exprpb.Decl_FunctionDecl_Overload {
	return &exprpb.Decl_FunctionDecl_Overload{
		OverloadId:         id,
		ResultType:         resultType,
		Params:             argTypes,
		TypeParams:         typeParams,
		IsInstanceFunction: false}
}

// NewPrimitiveType creates a type for a primitive value. See the var declarations
// for Int, Uint, etc.
func NewPrimitiveType(primitive exprpb.Type_PrimitiveType) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Primitive{
			Primitive: primitive}}
}

// NewTypeType creates a new type designating a type.
func NewTypeType(nested *exprpb.Type) *exprpb.Type {
	if nested == nil {
		// must set the nested field for a valid oneof option
		nested = &exprpb.Type{}
	}
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Type{
			Type: nested}}
}

// NewTypeParamType creates a type corresponding to a named, contextual parameter.
func NewTypeParamType(name string) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_TypeParam{
			TypeParam: name}}
}

// NewWellKnownType creates a type corresponding to a protobuf well-known type
// value.
func NewWellKnownType(wellKnown exprpb.Type_WellKnownType) *exprpb.Type {
	return &exprpb.Type{
		TypeKind: &exprpb.Type_WellKnown{
			WellKnown: wellKnown}}
}

// NewWrapperType creates a wrapped primitive type instance. Wrapped types
// are roughly equivalent to a nullable, or optionally valued type.
func NewWrapperType(wrapped *exprpb.Type) *exprpb.Type {
	primitive := wrapped.GetPrimitive()
	if primitive == exprpb.Type_PRIMITIVE_TYPE_UNSPECIFIED {
		// TODO: return an error
		panic("Wrapped type must be a primitive")
	}
	return &exprpb.Type{
		TypeKind: &exprpb.Type_Wrapper{
			Wrapper: primitive}}
}
