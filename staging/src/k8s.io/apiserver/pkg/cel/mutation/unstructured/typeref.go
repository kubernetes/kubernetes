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
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apiserver/pkg/cel/mutation/common"
)

// TypeRef is the implementation of TypeRef for an unstructured object.
// This is especially usefully when the schema is not known or available.
type TypeRef struct {
	celObjectType *types.Type
	celTypeType   *types.Type
}

func (r *TypeRef) HasTrait(trait int) bool {
	return common.ObjectTraits|trait != 0
}

// TypeName returns the name of this TypeRef.
func (r *TypeRef) TypeName() string {
	return r.celObjectType.TypeName()
}

// Val returns an instance given the fields.
func (r *TypeRef) Val(fields map[string]ref.Val) ref.Val {
	return common.NewObjectVal(r, fields)
}

// CELType returns the type. The returned type is of TypeType type.
func (r *TypeRef) CELType() *types.Type {
	return r.celTypeType
}

// Field looks up the field by name.
// This is the unstructured version that allows any name as the field name.
// The returned field is of DynType type.
func (r *TypeRef) Field(name string) (*types.FieldType, bool) {
	return NewFieldType(name), true
}

// NewTypeRef creates a TypeRef by the given field name.
func NewTypeRef(name string) *TypeRef {
	objectType := types.NewObjectType(name, common.ObjectTraits)
	return &TypeRef{
		celObjectType: objectType,
		celTypeType:   types.NewTypeTypeWithParam(objectType),
	}
}
