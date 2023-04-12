/*
Copyright 2023 The Kubernetes Authors.

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

package cel

import (
	"github.com/google/cel-go/common/types/ref"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

var _ ref.TypeProvider = (*CompositedTypeProvider)(nil)
var _ ref.TypeAdapter = (*CompositedTypeAdapter)(nil)

// CompositedTypeProvider is the provider that tries each of the underlying
// providers in order, and returns result of the first successful attempt.
type CompositedTypeProvider struct {
	// Providers contains the underlying type providers.
	// If Providers is empty, the CompositedTypeProvider becomes no-op provider.
	Providers []ref.TypeProvider
}

// EnumValue finds out the numeric value of the given enum name.
// The result comes from first provider that returns non-nil.
func (c *CompositedTypeProvider) EnumValue(enumName string) ref.Val {
	for _, p := range c.Providers {
		val := p.EnumValue(enumName)
		if val != nil {
			return val
		}
	}
	return nil
}

// FindIdent takes a qualified identifier name and returns a Value if one
// exists. The result comes from first provider that returns non-nil.
func (c *CompositedTypeProvider) FindIdent(identName string) (ref.Val, bool) {
	for _, p := range c.Providers {
		val, ok := p.FindIdent(identName)
		if ok {
			return val, ok
		}
	}
	return nil, false
}

// FindType finds the Type given a qualified type name, or return false
// if none of the providers finds the type.
// If any of the providers find the type, the first provider that returns true
// will be the result.
func (c *CompositedTypeProvider) FindType(typeName string) (*exprpb.Type, bool) {
	for _, p := range c.Providers {
		typ, ok := p.FindType(typeName)
		if ok {
			return typ, ok
		}
	}
	return nil, false
}

// FindFieldType returns the field type for a checked type value. Returns
// false if none of the providers can find the type.
// If multiple providers can find the field, the result is taken from
// the first that does.
func (c *CompositedTypeProvider) FindFieldType(messageType string, fieldName string) (*ref.FieldType, bool) {
	for _, p := range c.Providers {
		ft, ok := p.FindFieldType(messageType, fieldName)
		if ok {
			return ft, ok
		}
	}
	return nil, false
}

// NewValue creates a new type value from a qualified name and map of field
// name to value.
// If multiple providers can create the new type, the first that returns
// non-nil will decide the result.
func (c *CompositedTypeProvider) NewValue(typeName string, fields map[string]ref.Val) ref.Val {
	for _, p := range c.Providers {
		v := p.NewValue(typeName, fields)
		if v != nil {
			return v
		}
	}
	return nil
}

// CompositedTypeAdapter is the adapter that tries each of the underlying
// type adapter in order until the first successfully conversion.
type CompositedTypeAdapter struct {
	// Adapters contains underlying type adapters.
	// If Adapters is empty, the CompositedTypeAdapter becomes a no-op adapter.
	Adapters []ref.TypeAdapter
}

// NativeToValue takes the value and convert it into a ref.Val
// The result comes from the first TypeAdapter that returns non-nil.
func (c *CompositedTypeAdapter) NativeToValue(value interface{}) ref.Val {
	for _, a := range c.Adapters {
		v := a.NativeToValue(value)
		if v != nil {
			return v
		}
	}
	return nil
}
