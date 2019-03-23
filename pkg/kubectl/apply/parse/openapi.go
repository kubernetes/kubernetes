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
	"strings"

	"k8s.io/kube-openapi/pkg/util/proto"
)

// Contains functions for casting openapi interfaces to their underlying types

// getSchemaType returns the string type of the schema - e.g. array, primitive, map, kind, reference
func getSchemaType(schema proto.Schema) string {
	if schema == nil {
		return ""
	}
	visitor := &baseSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Kind
}

// getKind converts schema to an *proto.Kind object
func getKind(schema proto.Schema) (*proto.Kind, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &kindSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

// getArray converts schema to an *proto.Array object
func getArray(schema proto.Schema) (*proto.Array, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &arraySchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

// getMap converts schema to an *proto.Map object
func getMap(schema proto.Schema) (*proto.Map, error) {
	if schema == nil {
		return nil, nil
	}
	visitor := &mapSchemaVisitor{}
	schema.Accept(visitor)
	return visitor.Result, visitor.Err
}

// getPrimitive converts schema to an *proto.Primitive object
func getPrimitive(schema proto.Schema) (*proto.Primitive, error) {
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
func (v *baseSchemaVisitor) VisitArray(array *proto.Array) {
	v.Kind = "array"
	v.Err = fmt.Errorf("Array type not expected")
}

// MergeMap implements openapi
func (v *baseSchemaVisitor) VisitMap(*proto.Map) {
	v.Kind = "map"
	v.Err = fmt.Errorf("Map type not expected")
}

// MergePrimitive implements openapi
func (v *baseSchemaVisitor) VisitPrimitive(*proto.Primitive) {
	v.Kind = "primitive"
	v.Err = fmt.Errorf("Primitive type not expected")
}

// VisitKind implements openapi
func (v *baseSchemaVisitor) VisitKind(*proto.Kind) {
	v.Kind = "kind"
	v.Err = fmt.Errorf("Kind type not expected")
}

// VisitReference implements openapi
func (v *baseSchemaVisitor) VisitReference(reference proto.Reference) {
	v.Kind = "reference"
	v.Err = fmt.Errorf("Reference type not expected")
}

type kindSchemaVisitor struct {
	baseSchemaVisitor
	Result *proto.Kind
}

// VisitKind implements openapi
func (v *kindSchemaVisitor) VisitKind(result *proto.Kind) {
	v.Result = result
	v.Kind = "kind"
}

// VisitReference implements openapi
func (v *kindSchemaVisitor) VisitReference(reference proto.Reference) {
	reference.SubSchema().Accept(v)
	if v.Err == nil {
		v.Err = copyExtensions(reference.GetPath().String(), reference.GetExtensions(), v.Result.Extensions)
	}
}

func copyExtensions(field string, from, to map[string]interface{}) error {
	// Copy extensions from field to type for references
	for key, val := range from {
		if curr, found := to[key]; found {
			// Don't allow the same extension to be defined both on the field and on the type
			return fmt.Errorf("Cannot override value for extension %s on field %s from %v to %v",
				key, field, curr, val)
		}
		to[key] = val
	}
	return nil
}

type mapSchemaVisitor struct {
	baseSchemaVisitor
	Result *proto.Map
}

// MergeMap implements openapi
func (v *mapSchemaVisitor) VisitMap(result *proto.Map) {
	v.Result = result
	v.Kind = "map"
}

// VisitReference implements openapi
func (v *mapSchemaVisitor) VisitReference(reference proto.Reference) {
	reference.SubSchema().Accept(v)
	if v.Err == nil {
		v.Err = copyExtensions(reference.GetPath().String(), reference.GetExtensions(), v.Result.Extensions)
	}
}

type arraySchemaVisitor struct {
	baseSchemaVisitor
	Result *proto.Array
}

// VisitArray implements openapi
func (v *arraySchemaVisitor) VisitArray(result *proto.Array) {
	v.Result = result
	v.Kind = "array"
	v.Err = copySubElementPatchStrategy(result.Path.String(), result.GetExtensions(), result.SubType.GetExtensions())
}

// copyPatchStrategy copies the strategies to subelements to the subtype
// e.g. PodTemplate.Volumes is a []Volume with "x-kubernetes-patch-strategy": "merge,retainKeys"
// the "retainKeys" strategy applies to merging Volumes, and must be copied to the sub element
func copySubElementPatchStrategy(field string, from, to map[string]interface{}) error {
	// Check if the parent has a patch strategy extension
	if ext, found := from["x-kubernetes-patch-strategy"]; found {
		strategy, ok := ext.(string)
		if !ok {
			return fmt.Errorf("Expected string value for x-kubernetes-patch-strategy on %s, was %T",
				field, ext)
		}
		// Check of the parent patch strategy has a sub patch strategy, and if so copy to the sub type
		if strings.Contains(strategy, ",") {
			strategies := strings.Split(strategy, ",")
			if len(strategies) != 2 {
				// Only 1 sub strategy is supported
				return fmt.Errorf(
					"Expected between 0 and 2 elements for x-kubernetes-patch-merge-strategy by got %v",
					strategies)
			}
			to["x-kubernetes-patch-strategy"] = strategies[1]
		}
	}
	return nil
}

// MergePrimitive implements openapi
func (v *arraySchemaVisitor) VisitReference(reference proto.Reference) {
	reference.SubSchema().Accept(v)
	if v.Err == nil {
		v.Err = copyExtensions(reference.GetPath().String(), reference.GetExtensions(), v.Result.Extensions)
	}
}

type primitiveSchemaVisitor struct {
	baseSchemaVisitor
	Result *proto.Primitive
}

// MergePrimitive implements openapi
func (v *primitiveSchemaVisitor) VisitPrimitive(result *proto.Primitive) {
	v.Result = result
	v.Kind = "primitive"
}

// VisitReference implements openapi
func (v *primitiveSchemaVisitor) VisitReference(reference proto.Reference) {
	reference.SubSchema().Accept(v)
	if v.Err == nil {
		v.Err = copyExtensions(reference.GetPath().String(), reference.GetExtensions(), v.Result.Extensions)
	}
}
