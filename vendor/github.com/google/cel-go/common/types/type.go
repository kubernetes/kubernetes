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

package types

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

var (
	// TypeType is the type of a TypeValue.
	TypeType = NewTypeValue("type")
)

// TypeValue is an instance of a Value that describes a value's type.
type TypeValue struct {
	name      string
	traitMask int
}

// NewTypeValue returns *TypeValue which is both a ref.Type and ref.Val.
func NewTypeValue(name string, traits ...int) *TypeValue {
	traitMask := 0
	for _, trait := range traits {
		traitMask |= trait
	}
	return &TypeValue{
		name:      name,
		traitMask: traitMask}
}

// NewObjectTypeValue returns a *TypeValue based on the input name, which is
// annotated with the traits relevant to all objects.
func NewObjectTypeValue(name string) *TypeValue {
	return NewTypeValue(name,
		traits.FieldTesterType,
		traits.IndexerType)
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (t *TypeValue) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// TODO: replace the internal type representation with a proto-value.
	return nil, fmt.Errorf("type conversion not supported for 'type'")
}

// ConvertToType implements ref.Val.ConvertToType.
func (t *TypeValue) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case TypeType:
		return TypeType
	case StringType:
		return String(t.TypeName())
	}
	return NewErr("type conversion error from '%s' to '%s'", TypeType, typeVal)
}

// Equal implements ref.Val.Equal.
func (t *TypeValue) Equal(other ref.Val) ref.Val {
	otherType, ok := other.(ref.Type)
	return Bool(ok && t.TypeName() == otherType.TypeName())
}

// HasTrait indicates whether the type supports the given trait.
// Trait codes are defined in the traits package, e.g. see traits.AdderType.
func (t *TypeValue) HasTrait(trait int) bool {
	return trait&t.traitMask == trait
}

// String implements fmt.Stringer.
func (t *TypeValue) String() string {
	return t.name
}

// Type implements ref.Val.Type.
func (t *TypeValue) Type() ref.Type {
	return TypeType
}

// TypeName gives the type's name as a string.
func (t *TypeValue) TypeName() string {
	return t.name
}

// Value implements ref.Val.Value.
func (t *TypeValue) Value() any {
	return t.name
}
