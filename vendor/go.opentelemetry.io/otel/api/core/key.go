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

package core

//go:generate stringer -type=ValueType

import (
	"encoding/json"
	"fmt"
	"strconv"
	"unsafe"
)

// Key represents the key part in key-value pairs. It's a string. The
// allowed character set in the key depends on the use of the key.
type Key string

// KeyValue holds a key and value pair.
type KeyValue struct {
	Key   Key
	Value Value
}

// ValueType describes the type of the data Value holds.
type ValueType int

// Value represents the value part in key-value pairs.
type Value struct {
	vtype    ValueType
	numeric  uint64
	stringly string
	// TODO Lazy value type?
}

const (
	INVALID ValueType = iota // No value.
	BOOL                     // Boolean value, use AsBool() to get it.
	INT32                    // 32 bit signed integral value, use AsInt32() to get it.
	INT64                    // 64 bit signed integral value, use AsInt64() to get it.
	UINT32                   // 32 bit unsigned integral value, use AsUint32() to get it.
	UINT64                   // 64 bit unsigned integral value, use AsUint64() to get it.
	FLOAT32                  // 32 bit floating point value, use AsFloat32() to get it.
	FLOAT64                  // 64 bit floating point value, use AsFloat64() to get it.
	STRING                   // String value, use AsString() to get it.
)

// Bool creates a BOOL Value.
func Bool(v bool) Value {
	return Value{
		vtype:   BOOL,
		numeric: boolToRaw(v),
	}
}

// Int64 creates an INT64 Value.
func Int64(v int64) Value {
	return Value{
		vtype:   INT64,
		numeric: int64ToRaw(v),
	}
}

// Uint64 creates a UINT64 Value.
func Uint64(v uint64) Value {
	return Value{
		vtype:   UINT64,
		numeric: uint64ToRaw(v),
	}
}

// Float64 creates a FLOAT64 Value.
func Float64(v float64) Value {
	return Value{
		vtype:   FLOAT64,
		numeric: float64ToRaw(v),
	}
}

// Int32 creates an INT32 Value.
func Int32(v int32) Value {
	return Value{
		vtype:   INT32,
		numeric: int32ToRaw(v),
	}
}

// Uint32 creates a UINT32 Value.
func Uint32(v uint32) Value {
	return Value{
		vtype:   UINT32,
		numeric: uint32ToRaw(v),
	}
}

// Float32 creates a FLOAT32 Value.
func Float32(v float32) Value {
	return Value{
		vtype:   FLOAT32,
		numeric: float32ToRaw(v),
	}
}

// String creates a STRING Value.
func String(v string) Value {
	return Value{
		vtype:    STRING,
		stringly: v,
	}
}

// Int creates either an INT32 or an INT64 Value, depending on whether
// the int type is 32 or 64 bits wide.
func Int(v int) Value {
	if unsafe.Sizeof(v) == 4 {
		return Int32(int32(v))
	}
	return Int64(int64(v))
}

// Uint creates either a UINT32 or a UINT64 Value, depending on
// whether the uint type is 32 or 64 bits wide.
func Uint(v uint) Value {
	if unsafe.Sizeof(v) == 4 {
		return Uint32(uint32(v))
	}
	return Uint64(uint64(v))
}

// Bool creates a KeyValue instance with a BOOL Value.
//
// If creating both key and a bool value at the same time, then
// instead of calling core.Key(name).Bool(value) consider using a
// convenience function provided by the api/key package -
// key.Bool(name, value).
func (k Key) Bool(v bool) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Bool(v),
	}
}

// Int64 creates a KeyValue instance with an INT64 Value.
//
// If creating both key and an int64 value at the same time, then
// instead of calling core.Key(name).Int64(value) consider using a
// convenience function provided by the api/key package -
// key.Int64(name, value).
func (k Key) Int64(v int64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int64(v),
	}
}

// Uint64 creates a KeyValue instance with a UINT64 Value.
//
// If creating both key and a uint64 value at the same time, then
// instead of calling core.Key(name).Uint64(value) consider using a
// convenience function provided by the api/key package -
// key.Uint64(name, value).
func (k Key) Uint64(v uint64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Uint64(v),
	}
}

// Float64 creates a KeyValue instance with a FLOAT64 Value.
//
// If creating both key and a float64 value at the same time, then
// instead of calling core.Key(name).Float64(value) consider using a
// convenience function provided by the api/key package -
// key.Float64(name, value).
func (k Key) Float64(v float64) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float64(v),
	}
}

// Int32 creates a KeyValue instance with an INT32 Value.
//
// If creating both key and an int32 value at the same time, then
// instead of calling core.Key(name).Int32(value) consider using a
// convenience function provided by the api/key package -
// key.Int32(name, value).
func (k Key) Int32(v int32) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int32(v),
	}
}

// Uint32 creates a KeyValue instance with a UINT32 Value.
//
// If creating both key and a uint32 value at the same time, then
// instead of calling core.Key(name).Uint32(value) consider using a
// convenience function provided by the api/key package -
// key.Uint32(name, value).
func (k Key) Uint32(v uint32) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Uint32(v),
	}
}

// Float32 creates a KeyValue instance with a FLOAT32 Value.
//
// If creating both key and a float32 value at the same time, then
// instead of calling core.Key(name).Float32(value) consider using a
// convenience function provided by the api/key package -
// key.Float32(name, value).
func (k Key) Float32(v float32) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Float32(v),
	}
}

// String creates a KeyValue instance with a STRING Value.
//
// If creating both key and a string value at the same time, then
// instead of calling core.Key(name).String(value) consider using a
// convenience function provided by the api/key package -
// key.String(name, value).
func (k Key) String(v string) KeyValue {
	return KeyValue{
		Key:   k,
		Value: String(v),
	}
}

// Int creates a KeyValue instance with either an INT32 or an INT64
// Value, depending on whether the int type is 32 or 64 bits wide.
//
// If creating both key and an int value at the same time, then
// instead of calling core.Key(name).Int(value) consider using a
// convenience function provided by the api/key package -
// key.Int(name, value).
func (k Key) Int(v int) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Int(v),
	}
}

// Uint creates a KeyValue instance with either a UINT32 or a UINT64
// Value, depending on whether the uint type is 32 or 64 bits wide.
//
// If creating both key and a uint value at the same time, then
// instead of calling core.Key(name).Uint(value) consider using a
// convenience function provided by the api/key package -
// key.Uint(name, value).
func (k Key) Uint(v uint) KeyValue {
	return KeyValue{
		Key:   k,
		Value: Uint(v),
	}
}

// Defined returns true for non-empty keys.
func (k Key) Defined() bool {
	return len(k) != 0
}

// Type returns a type of the Value.
func (v Value) Type() ValueType {
	return v.vtype
}

// AsBool returns the bool value. Make sure that the Value's type is
// BOOL.
func (v Value) AsBool() bool {
	return rawToBool(v.numeric)
}

// AsInt32 returns the int32 value. Make sure that the Value's type is
// INT32.
func (v Value) AsInt32() int32 {
	return rawToInt32(v.numeric)
}

// AsInt64 returns the int64 value. Make sure that the Value's type is
// INT64.
func (v Value) AsInt64() int64 {
	return rawToInt64(v.numeric)
}

// AsUint32 returns the uint32 value. Make sure that the Value's type
// is UINT32.
func (v Value) AsUint32() uint32 {
	return rawToUint32(v.numeric)
}

// AsUint64 returns the uint64 value. Make sure that the Value's type is
// UINT64.
func (v Value) AsUint64() uint64 {
	return rawToUint64(v.numeric)
}

// AsFloat32 returns the float32 value. Make sure that the Value's
// type is FLOAT32.
func (v Value) AsFloat32() float32 {
	return rawToFloat32(v.numeric)
}

// AsFloat64 returns the float64 value. Make sure that the Value's
// type is FLOAT64.
func (v Value) AsFloat64() float64 {
	return rawToFloat64(v.numeric)
}

// AsString returns the string value. Make sure that the Value's type
// is STRING.
func (v Value) AsString() string {
	return v.stringly
}

type unknownValueType struct{}

// AsInterface returns Value's data as interface{}.
func (v Value) AsInterface() interface{} {
	switch v.Type() {
	case BOOL:
		return v.AsBool()
	case INT32:
		return v.AsInt32()
	case INT64:
		return v.AsInt64()
	case UINT32:
		return v.AsUint32()
	case UINT64:
		return v.AsUint64()
	case FLOAT32:
		return v.AsFloat32()
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
	case BOOL:
		return strconv.FormatBool(v.AsBool())
	case INT32:
		return strconv.FormatInt(int64(v.AsInt32()), 10)
	case INT64:
		return strconv.FormatInt(v.AsInt64(), 10)
	case UINT32:
		return strconv.FormatUint(uint64(v.AsUint32()), 10)
	case UINT64:
		return strconv.FormatUint(v.AsUint64(), 10)
	case FLOAT32:
		return fmt.Sprint(v.AsFloat32())
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
