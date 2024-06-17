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
	"strings"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	"k8s.io/apiserver/pkg/cel/mutation/common"
)

// mockTypeResolver is a mock implementation of TypeResolver that
// allows the object to contain any field.
type mockTypeResolver struct {
}

// mockTypeRef is a mock implementation of TypeRef that
// contains any field.
type mockTypeRef struct {
	objectType *types.Type
	resolver   common.TypeResolver
}

func newMockTypeRef(resolver common.TypeResolver, name string) *mockTypeRef {
	objectType := types.NewObjectType(name, common.ObjectTraits)
	return &mockTypeRef{
		objectType: objectType,
		resolver:   resolver,
	}
}

func (m *mockTypeRef) HasTrait(trait int) bool {
	return common.ObjectTraits|trait != 0
}

func (m *mockTypeRef) TypeName() string {
	return m.objectType.TypeName()
}

func (m *mockTypeRef) CELType() *types.Type {
	return types.NewTypeTypeWithParam(m.objectType)
}

func (m *mockTypeRef) Field(name string) (*types.FieldType, bool) {
	return &types.FieldType{
		Type: types.DynType,
		IsSet: func(target any) bool {
			return true
		},
		GetFrom: func(target any) (any, error) {
			return nil, nil
		},
	}, true
}

func (m *mockTypeRef) Val(fields map[string]ref.Val) ref.Val {
	return common.NewObjectVal(m, fields)
}

func (m *mockTypeResolver) Resolve(name string) (common.TypeRef, bool) {
	if strings.HasPrefix(name, common.RootTypeReferenceName) {
		return newMockTypeRef(m, name), true
	}
	return nil, false
}

// mockTypeResolverForOptional behaves the same as mockTypeResolver
// except returning a mockTypeRefForOptional instead of mockTypeRef
type mockTypeResolverForOptional struct {
	*mockTypeResolver
}

// mockTypeRefForOptional behaves the same as the underlying TypeRef
// except treating "nonExisting" field as non-existing.
// This is used for optional tests.
type mockTypeRefForOptional struct {
	common.TypeRef
}

// Field returns a mock FieldType, or false if the field should not exist.
func (m *mockTypeRefForOptional) Field(name string) (*types.FieldType, bool) {
	if name == "nonExisting" {
		return nil, false
	}
	return m.TypeRef.Field(name)
}

func (m *mockTypeResolverForOptional) Resolve(name string) (common.TypeRef, bool) {
	r, ok := m.mockTypeResolver.Resolve(name)
	if ok {
		return &mockTypeRefForOptional{TypeRef: r}, ok
	}
	return nil, false
}
