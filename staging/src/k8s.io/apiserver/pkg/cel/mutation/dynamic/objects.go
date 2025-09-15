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

package dynamic

import (
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"google.golang.org/protobuf/types/known/structpb"
)

// ObjectType is the implementation of the Object type for use when compiling
// CEL expressions without schema information about the object.
// This is to provide CEL expressions with access to Object{} types constructors.
type ObjectType struct {
	objectType *types.Type
}

func (o *ObjectType) HasTrait(trait int) bool {
	return o.objectType.HasTrait(trait)
}

// TypeName returns the name of this ObjectType.
func (o *ObjectType) TypeName() string {
	return o.objectType.TypeName()
}

// Val returns an instance given the fields.
func (o *ObjectType) Val(fields map[string]ref.Val) ref.Val {
	return NewObjectVal(o.objectType, fields)
}

func (o *ObjectType) Type() *types.Type {
	return o.objectType
}

// Field looks up the field by name.
// This is the unstructured version that allows any name as the field name.
// The returned field is of DynType type.
func (o *ObjectType) Field(name string) (*types.FieldType, bool) {
	return &types.FieldType{
		// for unstructured, we do not check for its type,
		// use DynType for all fields.
		Type: types.DynType,
		IsSet: func(target any) bool {
			if m, ok := target.(map[string]any); ok {
				_, isSet := m[name]
				return isSet
			}
			return false
		},
		GetFrom: func(target any) (any, error) {
			if m, ok := target.(map[string]any); ok {
				return m[name], nil
			}
			return nil, fmt.Errorf("cannot get field %q", name)
		},
	}, true
}

func (o *ObjectType) FieldNames() ([]string, bool) {
	return nil, true // Field names are not known for dynamic types. All field names are allowed.
}

// NewObjectType creates a ObjectType by the given field name.
func NewObjectType(name string) *ObjectType {
	return &ObjectType{
		objectType: types.NewObjectType(name),
	}
}

// ObjectVal is the CEL Val for an object that is constructed via the Object{} in
// CEL expressions without schema information about the object.
type ObjectVal struct {
	objectType *types.Type
	fields     map[string]ref.Val
}

// NewObjectVal creates an ObjectVal by its ResolvedType and its fields.
func NewObjectVal(objectType *types.Type, fields map[string]ref.Val) *ObjectVal {
	return &ObjectVal{
		objectType: objectType,
		fields:     fields,
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
	result := make(map[string]any, len(v.fields))
	for k, v := range v.fields {
		converted, err := convertField(v)
		if err != nil {
			return nil, fmt.Errorf("fail to convert field %q: %w", k, err)
		}
		result[k] = converted
	}
	if typeDesc == reflect.TypeOf(result) {
		return result, nil
	}
	// CEL's builtin data literal values all support conversion to structpb.Value, which
	// can then be serialized to JSON. This is convenient for CEL expressions that return
	// an arbitrary JSON value, such as our MutatingAdmissionPolicy JSON Patch valueExpression
	// field, so we support the conversion here, for Object data literals, as well.
	if typeDesc == reflect.TypeOf(&structpb.Value{}) {
		return structpb.NewStruct(result)
	}
	return nil, fmt.Errorf("unable to convert to %v", typeDesc)
}

// ConvertToType supports type conversions between CEL value types supported by the expression language.
func (v *ObjectVal) ConvertToType(typeValue ref.Type) ref.Val {
	if v.objectType.TypeName() == typeValue.TypeName() {
		return v
	}
	if typeValue == types.TypeType {
		return types.NewTypeTypeWithParam(v.objectType)
	}
	return types.NewErr("unsupported conversion into %v", typeValue)
}

// Equal returns true if the `other` value has the same type and content as the implementing struct.
func (v *ObjectVal) Equal(other ref.Val) ref.Val {
	if rhs, ok := other.(*ObjectVal); ok {
		if v.objectType.Equal(rhs.objectType) != types.True {
			return types.False
		}
		return types.Bool(reflect.DeepEqual(v.fields, rhs.fields))
	}
	return types.False
}

// Type returns the TypeValue of the value.
func (v *ObjectVal) Type() ref.Type {
	return types.NewObjectType(v.objectType.TypeName())
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

// CheckTypeNamesMatchFieldPathNames transitively checks the CEL object type names of this ObjectVal. Returns all
// found type name mismatch errors.
// Children ObjectVal types under <field> or this ObjectVal
// must have type names of the form "<ObjectVal.TypeName>.<field>", children of that type must have type names of the
// form "<ObjectVal.TypeName>.<field>.<field>" and so on.
// Intermediate maps and lists are unnamed and ignored.
func (v *ObjectVal) CheckTypeNamesMatchFieldPathNames() error {
	return errors.Join(typeCheck(v, []string{v.Type().TypeName()})...)

}

func typeCheck(v ref.Val, typeNamePath []string) []error {
	var errs []error
	if ov, ok := v.(*ObjectVal); ok {
		tn := ov.objectType.TypeName()
		if strings.Join(typeNamePath, ".") != tn {
			errs = append(errs, fmt.Errorf("unexpected type name %q, expected %q, which matches field name path from root Object type", tn, strings.Join(typeNamePath, ".")))
		}
		for k, f := range ov.fields {
			errs = append(errs, typeCheck(f, append(typeNamePath, k))...)
		}
	}
	value := v.Value()
	if listOfVal, ok := value.([]ref.Val); ok {
		for _, v := range listOfVal {
			errs = append(errs, typeCheck(v, typeNamePath)...)
		}
	}

	if mapOfVal, ok := value.(map[ref.Val]ref.Val); ok {
		for _, v := range mapOfVal {
			errs = append(errs, typeCheck(v, typeNamePath)...)
		}
	}
	return errs
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
		result := make(map[string]any, len(mapOfVal))
		for k, v := range mapOfVal {
			stringKey, ok := k.Value().(string)
			if !ok {
				return nil, fmt.Errorf("map key %q is of type %T, not string", k, k)
			}
			result[stringKey] = v.Value()
		}
		return result, nil
	}
	return value.Value(), nil
}
