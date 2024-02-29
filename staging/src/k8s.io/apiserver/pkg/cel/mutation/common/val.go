/*
Copyright 2024 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// ObjectVal is the CEL Val for an object that is constructed via the object
// construction syntax.
type ObjectVal struct {
	typeRef TypeRef
	fields  map[string]ref.Val
}

// NewObjectVal creates an ObjectVal by its TypeRef and its fields.
func NewObjectVal(typeRef TypeRef, fields map[string]ref.Val) *ObjectVal {
	return &ObjectVal{
		typeRef: typeRef,
		fields:  fields,
	}
}

var _ ref.Val = (*ObjectVal)(nil)
var _ traits.Zeroer = (*ObjectVal)(nil)

// ConvertToNative converts the object to map[string]any.
// All nested lists are converted into []any native type.
//
// It returns an error if the target type is not map[string]any,
// or any recursive conversion fails.
func (v *ObjectVal) ConvertToNative(typeDesc reflect.Type) (any, error) {
	var result map[string]any
	if typeDesc != reflect.TypeOf(result) {
		return nil, fmt.Errorf("unable to convert to %v", typeDesc)
	}
	result = make(map[string]any, len(v.fields))
	for k, v := range v.fields {
		converted, err := convertField(v)
		if err != nil {
			return nil, fmt.Errorf("fail to convert field %q: %w", k, err)
		}
		result[k] = converted
	}
	return result, nil
}

// ConvertToType supports type conversions between CEL value types supported by the expression language.
func (v *ObjectVal) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case v.typeRef:
		return v
	case types.TypeType:
		return v.typeRef.CELType()
	}
	return types.NewErr("unsupported conversion into %v", typeValue)
}

// Equal returns true if the `other` value has the same type and content as the implementing struct.
func (v *ObjectVal) Equal(other ref.Val) ref.Val {
	if rhs, ok := other.(*ObjectVal); ok {
		return types.Bool(reflect.DeepEqual(v.fields, rhs.fields))
	}
	return types.Bool(false)
}

// Type returns the TypeValue of the value.
func (v *ObjectVal) Type() ref.Type {
	return v.typeRef.CELType()
}

// Value returns its value as a map[string]any.
func (v *ObjectVal) Value() any {
	var result any
	var object map[string]any
	result, err := v.ConvertToNative(reflect.TypeOf(object))
	if err != nil {
		return types.WrapErr(err)
	}
	return result
}

// IsZeroValue indicates whether the object is the zero value for the type.
// For the ObjectVal, it is zero value if and only if the fields map is empty.
func (v *ObjectVal) IsZeroValue() bool {
	return len(v.fields) == 0
}

// convertField converts a referred ref.Val to its expected type.
// For objects, the expected type is map[string]any
// For lists, the expected type is []any
// For maps, the expected type is map[string]any
// For anything else, it is converted via value.Value()
//
// It will return an error if the request type is a map but the key
// is not a string.
func convertField(value ref.Val) (any, error) {
	// special handling for lists, where the elements are converted with Value() instead of ConvertToNative
	// to allow them to become native value of any type.
	if listOfVal, ok := value.Value().([]ref.Val); ok {
		var result []any
		for _, v := range listOfVal {
			result = append(result, v.Value())
		}
		return result, nil
	}
	// unstructured maps, as seen in annotations
	// map keys must be strings
	if mapOfVal, ok := value.Value().(map[ref.Val]ref.Val); ok {
		result := make(map[string]any)
		for k, v := range mapOfVal {
			stringKey, ok := k.Value().(string)
			if !ok {
				return nil, fmt.Errorf("map key %q is of type %t, not string", k, k)
			}
			result[stringKey] = v.Value()
		}
		return result, nil
	}
	return value.Value(), nil
}
