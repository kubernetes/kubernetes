/*
Copyright 2017 The Kubernetes Authors.

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

package parse

import (
	"fmt"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

// Contains functions for casting openapi interfaces to their underlying types

// getSchemaType returns the string type of the schema - e.g. array, primitive, map, kind, reference
func getSchemaType(schema openapi.Schema) string {
	if schema == nil {
		return ""
	}
	visitor := &baseSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Kind
}

// getKind converts schema to an *openapi.Kind object
func getKind(schema openapi.Schema) (*openapi.Kind, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &kindSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

// getArray converts schema to an *openapi.Array object
func getArray(schema openapi.Schema) (*openapi.Array, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &arraySchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

// getMap converts schema to an *openapi.Map object
func getMap(schema openapi.Schema) (*openapi.Map, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &mapSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

// getPrimitive converts schema to an *openapi.Primitive object
func getPrimitive(schema openapi.Schema) (*openapi.Primitive, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &primitiveSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

type baseSchemaVisitor struct {
	Err  error
	Kind string
}

// VisitArray implements openapi
func (v *baseSchemaVisitor) VisitArray(array *openapi.Array) {
	v.Kind = "array"
	v.Err = fmt.Errorf("Array type not expected")
}

// MergeMap implements openapi
func (v *baseSchemaVisitor) VisitMap(*openapi.Map) {
	v.Kind = "map"
	v.Err = fmt.Errorf("Map type not expected")
}

// MergePrimitive implements openapi
func (v *baseSchemaVisitor) VisitPrimitive(*openapi.Primitive) {
	v.Kind = "primitive"
	v.Err = fmt.Errorf("Primitive type not expected")
}

// VisitKind implements openapi
func (v *baseSchemaVisitor) VisitKind(*openapi.Kind) {
	v.Kind = "kind"
	v.Err = fmt.Errorf("Kind type not expected")
}

// VisitReference implements openapi
func (v *baseSchemaVisitor) VisitReference(reference openapi.Reference) {
	v.Kind = "reference"
	v.Err = fmt.Errorf("Reference type not expected")
}

type kindSchemaVisitor struct {
	baseSchemaVisitor
	Result *openapi.Kind
}

// VisitKind implements openapi
func (v *kindSchemaVisitor) VisitKind(result *openapi.Kind) {
	v.Result = result
	v.Kind = "kind"
}

// VisitReference implements openapi
func (v *kindSchemaVisitor) VisitReference(reference openapi.Reference) {
	reference.SubSchema().Accept(v)
}

type mapSchemaVisitor struct {
	baseSchemaVisitor
	Result *openapi.Map
}

// MergeMap implements openapi
func (v *mapSchemaVisitor) VisitMap(result *openapi.Map) {
	v.Result = result
	v.Kind = "map"
}

// VisitReference implements openapi
func (v *mapSchemaVisitor) VisitReference(reference openapi.Reference) {
	reference.SubSchema().Accept(v)
}

type arraySchemaVisitor struct {
	baseSchemaVisitor
	Result *openapi.Array
}

// VisitArray implements openapi
func (v *arraySchemaVisitor) VisitArray(result *openapi.Array) {
	v.Result = result
	v.Kind = "array"
}

// MergePrimitive implements openapi
func (v *arraySchemaVisitor) VisitReference(reference openapi.Reference) {
	reference.SubSchema().Accept(v)
}

type primitiveSchemaVisitor struct {
	baseSchemaVisitor
	Result *openapi.Primitive
}

// MergePrimitive implements openapi
func (v *primitiveSchemaVisitor) VisitPrimitive(result *openapi.Primitive) {
	v.Result = result
	v.Kind = "primitive"
}

// VisitReference implements openapi
func (v *primitiveSchemaVisitor) VisitReference(reference openapi.Reference) {
	reference.SubSchema().Accept(v)
}
