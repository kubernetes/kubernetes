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
	"math"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/google/cel-go/common/types/ref"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
)

// Int type that implements ref.Val as well as comparison and math operators.
type Int int64

// Int constants used for comparison results.
const (
	// IntZero is the zero-value for Int
	IntZero   = Int(0)
	IntOne    = Int(1)
	IntNegOne = Int(-1)
)

var (
	// int32WrapperType reflected type for protobuf int32 wrapper type.
	int32WrapperType = reflect.TypeOf(&wrapperspb.Int32Value{})

	// int64WrapperType reflected type for protobuf int64 wrapper type.
	int64WrapperType = reflect.TypeOf(&wrapperspb.Int64Value{})
)

// Add implements traits.Adder.Add.
func (i Int) Add(other ref.Val) ref.Val {
	otherInt, ok := other.(Int)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	val, err := addInt64Checked(int64(i), int64(otherInt))
	if err != nil {
		return WrapErr(err)
	}
	return Int(val)
}

// Compare implements traits.Comparer.Compare.
func (i Int) Compare(other ref.Val) ref.Val {
	switch ov := other.(type) {
	case Double:
		if math.IsNaN(float64(ov)) {
			return NewErr("NaN values cannot be ordered")
		}
		return compareIntDouble(i, ov)
	case Int:
		return compareInt(i, ov)
	case Uint:
		return compareIntUint(i, ov)
	default:
		return MaybeNoSuchOverloadErr(other)
	}
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (i Int) ConvertToNative(typeDesc reflect.Type) (any, error) {
	switch typeDesc.Kind() {
	case reflect.Int, reflect.Int32:
		// Enums are also mapped as int32 derivations.
		// Note, the code doesn't convert to the enum value directly since this is not known, but
		// the net effect with respect to proto-assignment is handled correctly by the reflection
		// Convert method.
		v, err := int64ToInt32Checked(int64(i))
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(v).Convert(typeDesc).Interface(), nil
	case reflect.Int8:
		v, err := int64ToInt8Checked(int64(i))
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(v).Convert(typeDesc).Interface(), nil
	case reflect.Int16:
		v, err := int64ToInt16Checked(int64(i))
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(v).Convert(typeDesc).Interface(), nil
	case reflect.Int64:
		return reflect.ValueOf(i).Convert(typeDesc).Interface(), nil
	case reflect.Ptr:
		switch typeDesc {
		case anyValueType:
			// Primitives must be wrapped before being set on an Any field.
			return anypb.New(wrapperspb.Int64(int64(i)))
		case int32WrapperType:
			// Convert the value to a wrapperspb.Int32Value, error on overflow.
			v, err := int64ToInt32Checked(int64(i))
			if err != nil {
				return nil, err
			}
			return wrapperspb.Int32(v), nil
		case int64WrapperType:
			// Convert the value to a wrapperspb.Int64Value.
			return wrapperspb.Int64(int64(i)), nil
		case jsonValueType:
			// The proto-to-JSON conversion rules would convert all 64-bit integer values to JSON
			// decimal strings. Because CEL ints might come from the automatic widening of 32-bit
			// values in protos, the JSON type is chosen dynamically based on the value.
			//
			// - Integers -2^53-1 < n < 2^53-1 are encoded as JSON numbers.
			// - Integers outside this range are encoded as JSON strings.
			//
			// The integer to float range represents the largest interval where such a conversion
			// can round-trip accurately. Thus, conversions from a 32-bit source can expect a JSON
			// number as with protobuf. Those consuming JSON from a 64-bit source must be able to
			// handle either a JSON number or a JSON decimal string. To handle these cases safely
			// the string values must be explicitly converted to int() within a CEL expression;
			// however, it is best to simply stay within the JSON number range when building JSON
			// objects in CEL.
			if i.isJSONSafe() {
				return structpb.NewNumberValue(float64(i)), nil
			}
			// Proto3 to JSON conversion requires string-formatted int64 values
			// since the conversion to floating point would result in truncation.
			return structpb.NewStringValue(strconv.FormatInt(int64(i), 10)), nil
		}
		switch typeDesc.Elem().Kind() {
		case reflect.Int32:
			// Convert the value to a wrapperspb.Int32Value, error on overflow.
			v, err := int64ToInt32Checked(int64(i))
			if err != nil {
				return nil, err
			}
			p := reflect.New(typeDesc.Elem())
			p.Elem().Set(reflect.ValueOf(v).Convert(typeDesc.Elem()))
			return p.Interface(), nil
		case reflect.Int64:
			v := int64(i)
			p := reflect.New(typeDesc.Elem())
			p.Elem().Set(reflect.ValueOf(v).Convert(typeDesc.Elem()))
			return p.Interface(), nil
		}
	case reflect.Interface:
		iv := i.Value()
		if reflect.TypeOf(iv).Implements(typeDesc) {
			return iv, nil
		}
		if reflect.TypeOf(i).Implements(typeDesc) {
			return i, nil
		}
	}
	return nil, fmt.Errorf("unsupported type conversion from 'int' to %v", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (i Int) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case IntType:
		return i
	case UintType:
		u, err := int64ToUint64Checked(int64(i))
		if err != nil {
			return WrapErr(err)
		}
		return Uint(u)
	case DoubleType:
		return Double(i)
	case StringType:
		return String(fmt.Sprintf("%d", int64(i)))
	case TimestampType:
		// The maximum positive value that can be passed to time.Unix is math.MaxInt64 minus the number
		// of seconds between year 1 and year 1970. See comments on unixToInternal.
		if int64(i) < minUnixTime || int64(i) > maxUnixTime {
			return celErrTimestampOverflow
		}
		return timestampOf(time.Unix(int64(i), 0).UTC())
	case TypeType:
		return IntType
	}
	return NewErr("type conversion error from '%s' to '%s'", IntType, typeVal)
}

// Divide implements traits.Divider.Divide.
func (i Int) Divide(other ref.Val) ref.Val {
	otherInt, ok := other.(Int)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	val, err := divideInt64Checked(int64(i), int64(otherInt))
	if err != nil {
		return WrapErr(err)
	}
	return Int(val)
}

// Equal implements ref.Val.Equal.
func (i Int) Equal(other ref.Val) ref.Val {
	switch ov := other.(type) {
	case Double:
		if math.IsNaN(float64(ov)) {
			return False
		}
		return Bool(compareIntDouble(i, ov) == 0)
	case Int:
		return Bool(i == ov)
	case Uint:
		return Bool(compareIntUint(i, ov) == 0)
	default:
		return False
	}
}

// IsZeroValue returns true if integer is equal to 0
func (i Int) IsZeroValue() bool {
	return i == IntZero
}

// Modulo implements traits.Modder.Modulo.
func (i Int) Modulo(other ref.Val) ref.Val {
	otherInt, ok := other.(Int)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	val, err := moduloInt64Checked(int64(i), int64(otherInt))
	if err != nil {
		return WrapErr(err)
	}
	return Int(val)
}

// Multiply implements traits.Multiplier.Multiply.
func (i Int) Multiply(other ref.Val) ref.Val {
	otherInt, ok := other.(Int)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	val, err := multiplyInt64Checked(int64(i), int64(otherInt))
	if err != nil {
		return WrapErr(err)
	}
	return Int(val)
}

// Negate implements traits.Negater.Negate.
func (i Int) Negate() ref.Val {
	val, err := negateInt64Checked(int64(i))
	if err != nil {
		return WrapErr(err)
	}
	return Int(val)
}

// Subtract implements traits.Subtractor.Subtract.
func (i Int) Subtract(subtrahend ref.Val) ref.Val {
	subtraInt, ok := subtrahend.(Int)
	if !ok {
		return MaybeNoSuchOverloadErr(subtrahend)
	}
	val, err := subtractInt64Checked(int64(i), int64(subtraInt))
	if err != nil {
		return WrapErr(err)
	}
	return Int(val)
}

// Type implements ref.Val.Type.
func (i Int) Type() ref.Type {
	return IntType
}

// Value implements ref.Val.Value.
func (i Int) Value() any {
	return int64(i)
}

func (i Int) format(sb *strings.Builder) {
	sb.WriteString(strconv.FormatInt(int64(i), 10))
}

// isJSONSafe indicates whether the int is safely representable as a floating point value in JSON.
func (i Int) isJSONSafe() bool {
	return i >= minIntJSON && i <= maxIntJSON
}

const (
	// maxIntJSON is defined as the Number.MAX_SAFE_INTEGER value per EcmaScript 6.
	maxIntJSON = 1<<53 - 1
	// minIntJSON is defined as the Number.MIN_SAFE_INTEGER value per EcmaScript 6.
	minIntJSON = -maxIntJSON
)
