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
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// TypeResolver resolves a type by a given name.
type TypeResolver interface {
	// Resolve resolves the type by its name.
	// This function returns false if the name does not refer to a known object type.
	Resolve(name string) (ResolvedType, bool)
}

// ResolvedType refers an object type that can be looked up for its fields.
type ResolvedType interface {
	ref.Type

	Type() *types.Type

	// Field finds the field by the field name, or false if the field is not known.
	// This function directly return a FieldType that is known to CEL to be more customizable.
	Field(name string) (*types.FieldType, bool)

	// FieldNames returns the field names associated with the type, if the type
	// is found.
	FieldNames() ([]string, bool)

	// Val creates an instance for the ResolvedType, given its fields and their values.
	Val(fields map[string]ref.Val) ref.Val
}

// ResolverTypeProvider delegates type resolution first to the TypeResolver and then
// to the underlying types.Provider for types not resolved by the TypeResolver.
type ResolverTypeProvider struct {
	typeResolver           TypeResolver
	underlyingTypeProvider types.Provider
}

var _ types.Provider = (*ResolverTypeProvider)(nil)

// FindStructType returns the Type give a qualified type name, by looking it up with
// the DynamicTypeResolver and translating it to CEL Type.
// If the type is not known to the DynamicTypeResolver, the lookup falls back to the underlying
// ResolverTypeProvider instead.
func (p *ResolverTypeProvider) FindStructType(structType string) (*types.Type, bool) {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return types.NewTypeTypeWithParam(t.Type()), true
	}
	return p.underlyingTypeProvider.FindStructType(structType)
}

// FindStructFieldNames returns the field names associated with the type, if the type
// is found.
func (p *ResolverTypeProvider) FindStructFieldNames(structType string) ([]string, bool) {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return t.FieldNames()
	}
	return p.underlyingTypeProvider.FindStructFieldNames(structType)
}

// FindStructFieldType returns the field type for a checked type value.
// Returns false if the field could not be found.
func (p *ResolverTypeProvider) FindStructFieldType(structType, fieldName string) (*types.FieldType, bool) {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return t.Field(fieldName)
	}
	return p.underlyingTypeProvider.FindStructFieldType(structType, fieldName)
}

// NewValue creates a new type value from a qualified name and map of fields.
func (p *ResolverTypeProvider) NewValue(structType string, fields map[string]ref.Val) ref.Val {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return t.Val(fields)
	}
	return p.underlyingTypeProvider.NewValue(structType, fields)
}

func (p *ResolverTypeProvider) EnumValue(enumName string) ref.Val {
	return p.underlyingTypeProvider.EnumValue(enumName)
}

func (p *ResolverTypeProvider) FindIdent(identName string) (ref.Val, bool) {
	return p.underlyingTypeProvider.FindIdent(identName)
}

// ResolverEnvOption creates the ResolverTypeProvider with a given DynamicTypeResolver,
// and also returns the CEL ResolverEnvOption to apply it to the env.
func ResolverEnvOption(resolver TypeResolver) cel.EnvOption {
	_, envOpt := NewResolverTypeProviderAndEnvOption(resolver)
	return envOpt
}

// NewResolverTypeProviderAndEnvOption creates the ResolverTypeProvider with a given DynamicTypeResolver,
// and also returns the CEL ResolverEnvOption to apply it to the env.
func NewResolverTypeProviderAndEnvOption(resolver TypeResolver) (*ResolverTypeProvider, cel.EnvOption) {
	tp := &ResolverTypeProvider{typeResolver: resolver}
	var envOption cel.EnvOption = func(e *cel.Env) (*cel.Env, error) {
		// wrap the existing type provider (acquired from the env)
		// and set new type provider for the env.
		tp.underlyingTypeProvider = e.CELTypeProvider()
		typeProviderOption := cel.CustomTypeProvider(tp)
		return typeProviderOption(e)
	}
	return tp, envOption
}
