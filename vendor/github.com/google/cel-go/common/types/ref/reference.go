// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ref contains the reference interfaces used throughout the types
// components.
package ref

import (
	"reflect"
)

// Type interface indicate the name of a given type.
type Type interface {
	// HasTrait returns whether the type has a given trait associated with it.
	//
	// See common/types/traits/traits.go for a list of supported traits.
	HasTrait(trait int) bool

	// TypeName returns the qualified type name of the type.
	//
	// The type name is also used as the type's identifier name at type-check
	// and interpretation time.
	TypeName() string
}

// Val interface defines the functions supported by all expression values.
// Val implementations may specialize the behavior of the value through the
// addition of traits.
type Val interface {
	// ConvertToNative converts the Value to a native Go struct according to the
	// reflected type description, or error if the conversion is not feasible.
	ConvertToNative(typeDesc reflect.Type) (interface{}, error)

	// ConvertToType supports type conversions between value types supported by
	// the expression language.
	ConvertToType(typeValue Type) Val

	// Equal returns true if the `other` value has the same type and content as
	// the implementing struct.
	Equal(other Val) Val

	// Type returns the TypeValue of the value.
	Type() Type

	// Value returns the raw value of the instance which may not be directly
	// compatible with the expression language types.
	Value() interface{}
}
