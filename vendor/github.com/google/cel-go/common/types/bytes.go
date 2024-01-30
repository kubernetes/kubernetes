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
	"bytes"
	"encoding/base64"
	"fmt"
	"reflect"
	"unicode/utf8"

	"github.com/google/cel-go/common/types/ref"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
)

// Bytes type that implements ref.Val and supports add, compare, and size
// operations.
type Bytes []byte

var (
	// byteWrapperType golang reflected type for protobuf bytes wrapper type.
	byteWrapperType = reflect.TypeOf(&wrapperspb.BytesValue{})
)

// Add implements traits.Adder interface method by concatenating byte sequences.
func (b Bytes) Add(other ref.Val) ref.Val {
	otherBytes, ok := other.(Bytes)
	if !ok {
		return ValOrErr(other, "no such overload")
	}
	return append(b, otherBytes...)
}

// Compare implements traits.Comparer interface method by lexicographic ordering.
func (b Bytes) Compare(other ref.Val) ref.Val {
	otherBytes, ok := other.(Bytes)
	if !ok {
		return ValOrErr(other, "no such overload")
	}
	return Int(bytes.Compare(b, otherBytes))
}

// ConvertToNative implements the ref.Val interface method.
func (b Bytes) ConvertToNative(typeDesc reflect.Type) (any, error) {
	switch typeDesc.Kind() {
	case reflect.Array, reflect.Slice:
		return reflect.ValueOf(b).Convert(typeDesc).Interface(), nil
	case reflect.Ptr:
		switch typeDesc {
		case anyValueType:
			// Primitives must be wrapped before being set on an Any field.
			return anypb.New(wrapperspb.Bytes([]byte(b)))
		case byteWrapperType:
			// Convert the bytes to a wrapperspb.BytesValue.
			return wrapperspb.Bytes([]byte(b)), nil
		case jsonValueType:
			// CEL follows the proto3 to JSON conversion by encoding bytes to a string via base64.
			// The encoding below matches the golang 'encoding/json' behavior during marshaling,
			// which uses base64.StdEncoding.
			str := base64.StdEncoding.EncodeToString([]byte(b))
			return structpb.NewStringValue(str), nil
		}
	case reflect.Interface:
		bv := b.Value()
		if reflect.TypeOf(bv).Implements(typeDesc) {
			return bv, nil
		}
		if reflect.TypeOf(b).Implements(typeDesc) {
			return b, nil
		}
	}
	return nil, fmt.Errorf("type conversion error from Bytes to '%v'", typeDesc)
}

// ConvertToType implements the ref.Val interface method.
func (b Bytes) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case StringType:
		if !utf8.Valid(b) {
			return NewErr("invalid UTF-8 in bytes, cannot convert to string")
		}
		return String(b)
	case BytesType:
		return b
	case TypeType:
		return BytesType
	}
	return NewErr("type conversion error from '%s' to '%s'", BytesType, typeVal)
}

// Equal implements the ref.Val interface method.
func (b Bytes) Equal(other ref.Val) ref.Val {
	otherBytes, ok := other.(Bytes)
	return Bool(ok && bytes.Equal(b, otherBytes))
}

// IsZeroValue returns true if the byte array is empty.
func (b Bytes) IsZeroValue() bool {
	return len(b) == 0
}

// Size implements the traits.Sizer interface method.
func (b Bytes) Size() ref.Val {
	return Int(len(b))
}

// Type implements the ref.Val interface method.
func (b Bytes) Type() ref.Type {
	return BytesType
}

// Value implements the ref.Val interface method.
func (b Bytes) Value() any {
	return []byte(b)
}
