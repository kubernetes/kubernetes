// Copyright 2022 Google LLC
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

package types

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types/ref"
)

var (
	// OptionalType indicates the runtime type of an optional value.
	OptionalType = NewTypeValue("optional")

	// OptionalNone is a sentinel value which is used to indicate an empty optional value.
	OptionalNone = &Optional{}
)

// OptionalOf returns an optional value which wraps a concrete CEL value.
func OptionalOf(value ref.Val) *Optional {
	return &Optional{value: value}
}

// Optional value which points to a value if non-empty.
type Optional struct {
	value ref.Val
}

// HasValue returns true if the optional has a value.
func (o *Optional) HasValue() bool {
	return o.value != nil
}

// GetValue returns the wrapped value contained in the optional.
func (o *Optional) GetValue() ref.Val {
	if !o.HasValue() {
		return NewErr("optional.none() dereference")
	}
	return o.value
}

// ConvertToNative implements the ref.Val interface method.
func (o *Optional) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if !o.HasValue() {
		return nil, errors.New("optional.none() dereference")
	}
	return o.value.ConvertToNative(typeDesc)
}

// ConvertToType implements the ref.Val interface method.
func (o *Optional) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case OptionalType:
		return o
	case TypeType:
		return OptionalType
	}
	return NewErr("type conversion error from '%s' to '%s'", OptionalType, typeVal)
}

// Equal determines whether the values contained by two optional values are equal.
func (o *Optional) Equal(other ref.Val) ref.Val {
	otherOpt, isOpt := other.(*Optional)
	if !isOpt {
		return False
	}
	if !o.HasValue() {
		return Bool(!otherOpt.HasValue())
	}
	if !otherOpt.HasValue() {
		return False
	}
	return o.value.Equal(otherOpt.value)
}

func (o *Optional) String() string {
	if o.HasValue() {
		return fmt.Sprintf("optional(%v)", o.GetValue())
	}
	return "optional.none()"
}

// Type implements the ref.Val interface method.
func (o *Optional) Type() ref.Type {
	return OptionalType
}

// Value returns the underlying 'Value()' of the wrapped value, if present.
func (o *Optional) Value() any {
	if o.value == nil {
		return nil
	}
	return o.value.Value()
}
