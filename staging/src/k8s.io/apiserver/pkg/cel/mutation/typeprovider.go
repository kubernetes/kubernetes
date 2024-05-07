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
	"k8s.io/apiserver/pkg/cel/mutation/common"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// TypeProvider is a specialized CEL type provider that understands
// the Object type alias that is used to construct an Apply configuration for
// a mutation operation.
type TypeProvider struct {
	typeResolver           common.TypeResolver
	underlyingTypeProvider types.Provider
}

var _ types.Provider = (*TypeProvider)(nil)

// EnumValue returns the numeric value of the given enum value name.
// This TypeProvider does not have special handling for EnumValue and thus directly delegate
// to its underlying type provider.
func (p *TypeProvider) EnumValue(enumName string) ref.Val {
	return p.underlyingTypeProvider.EnumValue(enumName)
}

// FindIdent takes a qualified identifier name and returns a ref.ObjectVal if one exists.
// This TypeProvider does not have special handling for FindIdent and thus directly delegate
// to its underlying type provider.
func (p *TypeProvider) FindIdent(identName string) (ref.Val, bool) {
	return p.underlyingTypeProvider.FindIdent(identName)
}

// FindStructType returns the Type give a qualified type name, by looking it up with
// the TypeResolver and translating it to CEL Type.
// If the type is not known to the TypeResolver, the lookup falls back to the underlying
// TypeProvider instead.
func (p *TypeProvider) FindStructType(structType string) (*types.Type, bool) {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return t.CELType(), true
	}
	return p.underlyingTypeProvider.FindStructType(structType)
}

// FindStructFieldType returns the field type for a checked type value.
// Returns false if the field could not be found.
func (p *TypeProvider) FindStructFieldType(structType, fieldName string) (*types.FieldType, bool) {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return t.Field(fieldName)
	}
	return p.underlyingTypeProvider.FindStructFieldType(structType, fieldName)
}

// NewValue creates a new type value from a qualified name and map of fields.
func (p *TypeProvider) NewValue(structType string, fields map[string]ref.Val) ref.Val {
	t, ok := p.typeResolver.Resolve(structType)
	if ok {
		return t.Val(fields)
	}
	return p.underlyingTypeProvider.NewValue(structType, fields)
}

// NewTypeProviderAndEnvOption creates the TypeProvider with a given TypeResolver,
// and also returns the CEL EnvOption to apply it to the env.
func NewTypeProviderAndEnvOption(resolver common.TypeResolver) (*TypeProvider, cel.EnvOption) {
	tp := &TypeProvider{typeResolver: resolver}
	var envOption cel.EnvOption = func(e *cel.Env) (*cel.Env, error) {
		// wrap the existing type provider (acquired from the env)
		// and set new type provider for the env.
		tp.underlyingTypeProvider = e.CELTypeProvider()
		typeProviderOption := cel.CustomTypeProvider(tp)
		return typeProviderOption(e)
	}
	return tp, envOption
}
