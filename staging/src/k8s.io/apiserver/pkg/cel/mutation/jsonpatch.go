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

package mutation

import (
	"fmt"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"reflect"
)

var jsonPatchType = types.NewObjectType(JSONPatchTypeName)

var (
	jsonPatchOp    = "op"
	jsonPatchPath  = "path"
	jsonPatchFrom  = "from"
	jsonPatchValue = "value"
)

// JSONPatchType and JSONPatchVal are defined entirely from scratch here because JSONPatchVal
// has a dynamic 'value' field which can not be defined with an OpenAPI schema,
// preventing us from using DeclType and UnstructuredToVal.

// JSONPatchType provides a CEL type for "JSONPatch" operations.
type JSONPatchType struct{}

func (r *JSONPatchType) HasTrait(trait int) bool {
	return jsonPatchType.HasTrait(trait)
}

// TypeName returns the name of this ObjectType.
func (r *JSONPatchType) TypeName() string {
	return jsonPatchType.TypeName()
}

// Val returns an instance given the fields.
func (r *JSONPatchType) Val(fields map[string]ref.Val) ref.Val {
	result := &JSONPatchVal{}
	for name, value := range fields {
		switch name {
		case jsonPatchOp:
			if s, ok := value.Value().(string); ok {
				result.Op = s
			} else {
				return types.NewErr("unexpected type %T for JSONPatchType 'op' field", value.Value())
			}
		case jsonPatchPath:
			if s, ok := value.Value().(string); ok {
				result.Path = s
			} else {
				return types.NewErr("unexpected type %T for JSONPatchType 'path' field", value.Value())
			}
		case jsonPatchFrom:
			if s, ok := value.Value().(string); ok {
				result.From = s
			} else {
				return types.NewErr("unexpected type %T for JSONPatchType 'from' field", value.Value())
			}
		case jsonPatchValue:
			result.Val = value
		default:
			return types.NewErr("unexpected JSONPatchType field: %s", name)
		}
	}
	return result
}

func (r *JSONPatchType) Type() *types.Type {
	return jsonPatchType
}

func (r *JSONPatchType) Field(name string) (*types.FieldType, bool) {
	var fieldType *types.Type
	switch name {
	case jsonPatchOp, jsonPatchFrom, jsonPatchPath:
		fieldType = cel.StringType
	case jsonPatchValue:
		fieldType = types.DynType
	}
	return &types.FieldType{
		Type: fieldType,
	}, true
}

func (r *JSONPatchType) FieldNames() ([]string, bool) {
	return []string{jsonPatchOp, jsonPatchFrom, jsonPatchPath, jsonPatchValue}, true
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
	if typeValue == jsonPatchType {
		return p
	} else if typeValue == types.TypeType {
		return types.NewTypeTypeWithParam(jsonPatchType)
	}
	return types.NewErr("unsupported type: %s", typeValue.TypeName())
}

func (p *JSONPatchVal) Equal(other ref.Val) ref.Val {
	if o, ok := other.(*JSONPatchVal); ok && p != nil && o != nil {
		if p.Op != o.Op || p.From != o.From || p.Path != o.Path {
			return types.False
		}
		if (p.Val == nil) != (o.Val == nil) {
			return types.False
		}
		if p.Val == nil {
			return types.True
		}
		return p.Val.Equal(o.Val)
	}
	return types.False
}

func (p *JSONPatchVal) Get(index ref.Val) ref.Val {
	if name, ok := index.Value().(string); ok {
		switch name {
		case jsonPatchOp:
			return types.String(p.Op)
		case jsonPatchPath:
			return types.String(p.Path)
		case jsonPatchFrom:
			return types.String(p.From)
		case jsonPatchValue:
			return p.Val
		default:

		}
	}
	return types.NewErr("unsupported indexer: %s", index)
}

func (p *JSONPatchVal) IsSet(field ref.Val) ref.Val {
	if name, ok := field.Value().(string); ok {
		switch name {
		case jsonPatchOp:
			return types.Bool(len(p.Op) > 0)
		case jsonPatchPath:
			return types.Bool(len(p.Path) > 0)
		case jsonPatchFrom:
			return types.Bool(len(p.From) > 0)
		case jsonPatchValue:
			return types.Bool(p.Val != nil)
		}
	}
	return types.NewErr("unsupported field: %s", field)
}

func (p *JSONPatchVal) Type() ref.Type {
	return jsonPatchType
}

func (p *JSONPatchVal) Value() any {
	return p
}

var _ ref.Val = &JSONPatchVal{}
