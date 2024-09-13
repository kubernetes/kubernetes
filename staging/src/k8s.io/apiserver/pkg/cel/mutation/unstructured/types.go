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

package unstructured

import (
	"fmt"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"reflect"

	"k8s.io/apiserver/pkg/cel/mutation/common"
)

// ObjectType is the implementation of the Object type for an unstructured object.
// This is used to compile CEL expressions that can construct Object{} types
// when the schema is not known or available.
type ObjectType struct {
	objectType *types.Type
}

func (r *ObjectType) HasTrait(trait int) bool {
	return r.objectType.HasTrait(trait)
}

// TypeName returns the name of this ObjectType.
func (r *ObjectType) TypeName() string {
	return r.objectType.TypeName()
}

// Val returns an instance given the fields.
func (r *ObjectType) Val(fields map[string]ref.Val) ref.Val {
	return common.NewObjectVal(r, fields)
}

func (r *ObjectType) Type() *types.Type {
	return r.objectType
}

func (r *ObjectType) TypeType() *types.Type {
	return types.NewTypeTypeWithParam(r.objectType)
}

// Field looks up the field by name.
// This is the unstructured version that allows any name as the field name.
// The returned field is of DynType type.
func (r *ObjectType) Field(name string) (*types.FieldType, bool) {
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

// NewTypeRef creates a ObjectType by the given field name.
func NewTypeRef(name string) *ObjectType {
	return &ObjectType{
		objectType: types.NewObjectType(name),
	}
}

var JSONPatchCELType = types.NewObjectType("JSONPatch")

// JSONPatchType and JSONPatchVal are defined entirely from scratch here because it
// has a dynamic value field.  If this could be defined with an OpenAPI schema,
// we could have used DeclType and UnstructuredToVal here instead.

// JSONPatchType provides a CEL type for "JSONPatch" operations.
type JSONPatchType struct{}

func (r *JSONPatchType) HasTrait(trait int) bool {
	return JSONPatchCELType.HasTrait(trait)
}

// TypeName returns the name of this ObjectType.
func (r *JSONPatchType) TypeName() string {
	return JSONPatchCELType.TypeName()
}

// Val returns an instance given the fields.
func (r *JSONPatchType) Val(fields map[string]ref.Val) ref.Val {
	result := &JSONPatchVal{}
	for name, value := range fields {
		switch name {
		case "op":
			if s, ok := value.Value().(string); ok {
				result.Op = s
			} else {
				return types.NewErr("unexpected type %T for JSONPatchType 'op' field", value.Value())
			}
		case "path":
			if s, ok := value.Value().(string); ok {
				result.Path = s
			} else {
				return types.NewErr("unexpected type %T for JSONPatchType 'path' field", value.Value())
			}
		case "from":
			if s, ok := value.Value().(string); ok {
				result.From = s
			} else {
				return types.NewErr("unexpected type %T for JSONPatchType 'from' field", value.Value())
			}
		case "value":
			result.Val = value
		default:
			return types.NewErr("unexpected JSONPatchType field: %s", name)
		}
	}
	return result
}

func (r *JSONPatchType) Type() *types.Type {
	return JSONPatchCELType
}

func (r *JSONPatchType) TypeType() *types.Type {
	return types.NewTypeTypeWithParam(JSONPatchCELType)
}

func (r *JSONPatchType) Field(name string) (*types.FieldType, bool) {
	var fieldType *types.Type
	switch name {
	case "op", "from", "path":
		fieldType = cel.StringType
	case "value":
		fieldType = types.DynType
	}
	return &types.FieldType{
		Type: fieldType,
	}, true
}

// JSONPatchVal is the ref.Val for a JSONPatch.
type JSONPatchVal struct {
	Op, From, Path string
	Val            ref.Val
}

func (p *JSONPatchVal) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if typeDesc == reflect.TypeOf(&JSONPatchVal{}) {
		return p, nil
	}
	return nil, fmt.Errorf("cannot convert to native type: %v", typeDesc)
}

func (p *JSONPatchVal) ConvertToType(typeValue ref.Type) ref.Val {
	if typeValue == JSONPatchCELType {
		return p
	} else if typeValue == types.TypeType {
		return types.NewTypeTypeWithParam(JSONPatchCELType)
	}
	return types.NewErr("Unsupported type: %s", typeValue.TypeName())
}

func (p *JSONPatchVal) Equal(other ref.Val) ref.Val {
	if o, ok := other.(*JSONPatchVal); ok && p != nil && o != nil {
		if *p == *o {
			return types.True
		}
	}
	return types.False
}

func (p *JSONPatchVal) Get(index ref.Val) ref.Val {
	if name, ok := index.Value().(string); ok {
		switch name {
		case "op":
			return types.String(p.Op)
		case "path":
			return types.String(p.Path)
		case "from":
			return types.String(p.From)
		case "value":
			return p.Val
		default:

		}
	}
	return types.NewErr("unsupported indexer: %s", index)
}

func (p *JSONPatchVal) IsSet(field ref.Val) ref.Val {
	if name, ok := field.Value().(string); ok {
		switch name {
		case "op":
			if len(p.Op) > 0 {
				return types.True
			}
			return types.False
		case "path":
			if len(p.Path) > 0 {
				return types.True
			}
			return types.False
		case "from":
			if len(p.From) > 0 {
				return types.True
			}
			return types.False
		case "value":
			if p.Val != nil {
				return types.True
			}
			return types.False
		}
	}
	return types.NewErr("unsupported field: %s", field)
}

func (p *JSONPatchVal) Type() ref.Type {
	return JSONPatchCELType
}

func (p *JSONPatchVal) Value() any {
	return p
}

var _ ref.Val = &JSONPatchVal{}
