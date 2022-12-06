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
	"google.golang.org/protobuf/proto"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// Null type implementation.
type Null structpb.NullValue

var (
	// NullType singleton.
	NullType = NewTypeValue("null_type")
	// NullValue singleton.
	NullValue = Null(structpb.NullValue_NULL_VALUE)

	jsonNullType = reflect.TypeOf(structpb.NullValue_NULL_VALUE)
)

// ConvertToNative implements ref.Val.ConvertToNative.
func (n Null) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	switch typeDesc.Kind() {
	case reflect.Int32:
		return reflect.ValueOf(n).Convert(typeDesc).Interface(), nil
	case reflect.Ptr:
		switch typeDesc {
		case anyValueType:
			// Convert to a JSON-null before packing to an Any field since the enum value for JSON
			// null cannot be packed directly.
			pb, err := n.ConvertToNative(jsonValueType)
			if err != nil {
				return nil, err
			}
			return anypb.New(pb.(proto.Message))
		case jsonValueType:
			return structpb.NewNullValue(), nil
		}
	case reflect.Interface:
		nv := n.Value()
		if reflect.TypeOf(nv).Implements(typeDesc) {
			return nv, nil
		}
		if reflect.TypeOf(n).Implements(typeDesc) {
			return n, nil
		}
	}
	// If the type conversion isn't supported return an error.
	return nil, fmt.Errorf("type conversion error from '%v' to '%v'", NullType, typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (n Null) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case StringType:
		return String("null")
	case NullType:
		return n
	case TypeType:
		return NullType
	}
	return NewErr("type conversion error from '%s' to '%s'", NullType, typeVal)
}

// Equal implements ref.Val.Equal.
func (n Null) Equal(other ref.Val) ref.Val {
	return Bool(NullType == other.Type())
}

// Type implements ref.Val.Type.
func (n Null) Type() ref.Type {
	return NullType
}

// Value implements ref.Val.Value.
func (n Null) Value() interface{} {
	return structpb.NullValue_NULL_VALUE
}
