// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"

	"go.opentelemetry.io/otel/internal"
)

//go:generate stringer -type=Type

// Type describes the type of the data Value holds.
type Type int

// Value represents the value part in key-value pairs.
type Value struct {
	vtype    Type
	numeric  uint64
	stringly string
	// TODO Lazy value type?

	array interface{}
}

const (
	// INVALID is used for a Value with no value set.
	INVALID Type = iota
	// BOOL is a boolean Type Value.
	BOOL
	// INT64 is a 64-bit signed integral Type Value.
	INT64
	// FLOAT64 is a 64-bit floating point Type Value.
	FLOAT64
	// STRING is a string Type Value.
	STRING
	// ARRAY is an array Type Value used to store 1-dimensional slices or
	// arrays of bool, int, int32, int64, uint, uint32, uint64, float,
	// float32, float64, or string types.
	ARRAY
)

// BoolValue creates a BOOL Value.
func BoolValue(v bool) Value {
	return Value{
		vtype:   BOOL,
		numeric: internal.BoolToRaw(v),
	}
}

// Int64Value creates an INT64 Value.
func Int64Value(v int64) Value {
	return Value{
		vtype:   INT64,
		numeric: internal.Int64ToRaw(v),
	}
}

// Float64Value creates a FLOAT64 Value.
func Float64Value(v float64) Value {
	return Value{
		vtype:   FLOAT64,
		numeric: internal.Float64ToRaw(v),
	}
}

// StringValue creates a STRING Value.
func StringValue(v string) Value {
	return Value{
		vtype:    STRING,
		stringly: v,
	}
}

// IntValue creates an INT64 Value.
func IntValue(v int) Value {
	return Int64Value(int64(v))
}

// ArrayValue creates an ARRAY value from an array or slice.
// Only arrays or slices of bool, int, int64, float, float64, or string types are allowed.
// Specifically, arrays  and slices can not contain other arrays, slices, structs, or non-standard
// types. If the passed value is not an array or slice of these types an
// INVALID value is returned.
func ArrayValue(v interface{}) Value {
	switch reflect.TypeOf(v).Kind() {
	case reflect.Array, reflect.Slice:
		// get array type regardless of dimensions
		typ := reflect.TypeOf(v).Elem()
		kind := typ.Kind()
		switch kind {
		case reflect.Bool, reflect.Int, reflect.Int64,
			reflect.Float64, reflect.String:
			val := reflect.ValueOf(v)
			length := val.Len()
			frozen := reflect.Indirect(reflect.New(reflect.ArrayOf(length, typ)))
			reflect.Copy(frozen, val)
			return Value{
				vtype: ARRAY,
				array: frozen.Interface(),
			}
		default:
			return Value{vtype: INVALID}
		}
	}
	return Value{vtype: INVALID}
}

// Type returns a type of the Value.
func (v Value) Type() Type {
	return v.vtype
}

// AsBool returns the bool value. Make sure that the Value's type is
// BOOL.
func (v Value) AsBool() bool {
	return internal.RawToBool(v.numeric)
}

// AsInt64 returns the int64 value. Make sure that the Value's type is
// INT64.
func (v Value) AsInt64() int64 {
	return internal.RawToInt64(v.numeric)
}

// AsFloat64 returns the float64 value. Make sure that the Value's
// type is FLOAT64.
func (v Value) AsFloat64() float64 {
	return internal.RawToFloat64(v.numeric)
}

// AsString returns the string value. Make sure that the Value's type
// is STRING.
func (v Value) AsString() string {
	return v.stringly
}

// AsArray returns the array Value as an interface{}.
func (v Value) AsArray() interface{} {
	return v.array
}

type unknownValueType struct{}

// AsInterface returns Value's data as interface{}.
func (v Value) AsInterface() interface{} {
	switch v.Type() {
	case ARRAY:
		return v.AsArray()
	case BOOL:
		return v.AsBool()
	case INT64:
		return v.AsInt64()
	case FLOAT64:
		return v.AsFloat64()
	case STRING:
		return v.stringly
	}
	return unknownValueType{}
}

// Emit returns a string representation of Value's data.
func (v Value) Emit() string {
	switch v.Type() {
	case ARRAY:
		return fmt.Sprint(v.array)
	case BOOL:
		return strconv.FormatBool(v.AsBool())
	case INT64:
		return strconv.FormatInt(v.AsInt64(), 10)
	case FLOAT64:
		return fmt.Sprint(v.AsFloat64())
	case STRING:
		return v.stringly
	default:
		return "unknown"
	}
}

// MarshalJSON returns the JSON encoding of the Value.
func (v Value) MarshalJSON() ([]byte, error) {
	var jsonVal struct {
		Type  string
		Value interface{}
	}
	jsonVal.Type = v.Type().String()
	jsonVal.Value = v.AsInterface()
	return json.Marshal(jsonVal)
}
