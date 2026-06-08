// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"

	attribute "go.opentelemetry.io/otel/attribute/internal"
)

//go:generate stringer -type=Type

// Type describes the type of the data Value holds.
type Type int // nolint: revive  // redefines builtin Type.

// Value represents the value part in key-value pairs.
//
// Note that the zero value is a valid empty value.
type Value struct {
	vtype    Type
	numeric  uint64
	stringly string
	slice    any
}

const (
	// EMPTY is used for a Value with no value set.
	EMPTY Type = iota
	// BOOL is a boolean Type Value.
	BOOL
	// INT64 is a 64-bit signed integral Type Value.
	INT64
	// FLOAT64 is a 64-bit floating point Type Value.
	FLOAT64
	// STRING is a string Type Value.
	STRING
	// BOOLSLICE is a slice of booleans Type Value.
	BOOLSLICE
	// INT64SLICE is a slice of 64-bit signed integral numbers Type Value.
	INT64SLICE
	// FLOAT64SLICE is a slice of 64-bit floating point numbers Type Value.
	FLOAT64SLICE
	// STRINGSLICE is a slice of strings Type Value.
	STRINGSLICE
	// BYTESLICE is a slice of bytes Type Value.
	BYTESLICE
	// SLICE is a slice of Value Type values.
	SLICE
	// INVALID is used for a Value with no value set.
	//
	// Deprecated: Use EMPTY instead as an empty value is a valid value.
	INVALID = EMPTY
)

// BoolValue creates a BOOL Value.
func BoolValue(v bool) Value {
	return Value{
		vtype:   BOOL,
		numeric: boolToRaw(v),
	}
}

// BoolSliceValue creates a BOOLSLICE Value.
func BoolSliceValue(v []bool) Value {
	return Value{vtype: BOOLSLICE, slice: attribute.SliceValue(v)}
}

// IntValue creates an INT64 Value.
func IntValue(v int) Value {
	return Int64Value(int64(v))
}

// IntSliceValue creates an INT64SLICE Value.
func IntSliceValue(v []int) Value {
	val := Value{vtype: INT64SLICE}

	// Avoid the common tiny-slice cases from allocating a new slice.
	switch len(v) {
	case 0:
		val.slice = [0]int64{}
	case 1:
		val.slice = [1]int64{int64(v[0])}
	case 2:
		val.slice = [2]int64{int64(v[0]), int64(v[1])}
	case 3:
		val.slice = [3]int64{int64(v[0]), int64(v[1]), int64(v[2])}
	default:
		// Fallback to a new slice for larger slices.
		cp := make([]int64, len(v))
		for i, val := range v {
			cp[i] = int64(val)
		}
		val.slice = attribute.SliceValue(cp)
	}

	return val
}

// Int64Value creates an INT64 Value.
func Int64Value(v int64) Value {
	return Value{
		vtype:   INT64,
		numeric: int64ToRaw(v),
	}
}

// Int64SliceValue creates an INT64SLICE Value.
func Int64SliceValue(v []int64) Value {
	return Value{vtype: INT64SLICE, slice: attribute.SliceValue(v)}
}

// Float64Value creates a FLOAT64 Value.
func Float64Value(v float64) Value {
	return Value{
		vtype:   FLOAT64,
		numeric: float64ToRaw(v),
	}
}

// Float64SliceValue creates a FLOAT64SLICE Value.
func Float64SliceValue(v []float64) Value {
	return Value{vtype: FLOAT64SLICE, slice: attribute.SliceValue(v)}
}

// StringValue creates a STRING Value.
func StringValue(v string) Value {
	return Value{
		vtype:    STRING,
		stringly: v,
	}
}

// StringSliceValue creates a STRINGSLICE Value.
func StringSliceValue(v []string) Value {
	return Value{vtype: STRINGSLICE, slice: attribute.SliceValue(v)}
}

// ByteSliceValue creates a BYTESLICE Value.
func ByteSliceValue(v []byte) Value {
	return Value{
		vtype:    BYTESLICE,
		stringly: string(v),
	}
}

// SliceValue creates a SLICE Value.
func SliceValue(v ...Value) Value {
	return Value{vtype: SLICE, slice: sliceValue(v)}
}

// Type returns a type of the Value.
func (v Value) Type() Type {
	return v.vtype
}

// AsBool returns the bool value. Make sure that the Value's type is
// BOOL.
func (v Value) AsBool() bool {
	return rawToBool(v.numeric)
}

// AsBoolSlice returns the []bool value. Make sure that the Value's type is
// BOOLSLICE.
func (v Value) AsBoolSlice() []bool {
	if v.vtype != BOOLSLICE {
		return nil
	}
	return v.asBoolSlice()
}

func (v Value) asBoolSlice() []bool {
	return attribute.AsSlice[bool](v.slice)
}

// AsInt64 returns the int64 value. Make sure that the Value's type is
// INT64.
func (v Value) AsInt64() int64 {
	return rawToInt64(v.numeric)
}

// AsInt64Slice returns the []int64 value. Make sure that the Value's type is
// INT64SLICE.
func (v Value) AsInt64Slice() []int64 {
	if v.vtype != INT64SLICE {
		return nil
	}
	return v.asInt64Slice()
}

func (v Value) asInt64Slice() []int64 {
	return attribute.AsSlice[int64](v.slice)
}

// AsFloat64 returns the float64 value. Make sure that the Value's
// type is FLOAT64.
func (v Value) AsFloat64() float64 {
	return rawToFloat64(v.numeric)
}

// AsFloat64Slice returns the []float64 value. Make sure that the Value's type is
// FLOAT64SLICE.
func (v Value) AsFloat64Slice() []float64 {
	if v.vtype != FLOAT64SLICE {
		return nil
	}
	return v.asFloat64Slice()
}

func (v Value) asFloat64Slice() []float64 {
	return attribute.AsSlice[float64](v.slice)
}

// AsString returns the string value. Make sure that the Value's type
// is STRING.
func (v Value) AsString() string {
	return v.stringly
}

// AsStringSlice returns the []string value. Make sure that the Value's type is
// STRINGSLICE.
func (v Value) AsStringSlice() []string {
	if v.vtype != STRINGSLICE {
		return nil
	}
	return v.asStringSlice()
}

func (v Value) asStringSlice() []string {
	return attribute.AsSlice[string](v.slice)
}

// AsSlice returns the []Value value. Make sure that the Value's type is
// SLICE.
func (v Value) AsSlice() []Value {
	if v.vtype != SLICE {
		return nil
	}
	return v.asSlice()
}

func (v Value) asSlice() []Value {
	switch vals := v.slice.(type) {
	case [0]Value:
		return []Value{}
	case [1]Value:
		return []Value{vals[0]}
	case [2]Value:
		return []Value{vals[0], vals[1]}
	case [3]Value:
		return []Value{vals[0], vals[1], vals[2]}
	case [4]Value:
		return []Value{vals[0], vals[1], vals[2], vals[3]}
	case [5]Value:
		return []Value{vals[0], vals[1], vals[2], vals[3], vals[4]}
	default:
		return asValueSliceReflect(v.slice)
	}
}

func asValueSliceReflect(v any) []Value {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || rv.Kind() != reflect.Array || rv.Type().Elem() != reflect.TypeFor[Value]() {
		return nil
	}
	cpy := make([]Value, rv.Len())
	if len(cpy) > 0 {
		_ = reflect.Copy(reflect.ValueOf(cpy), rv)
	}
	return cpy
}

// AsByteSlice returns the bytes value. Make sure that the Value's type
// is BYTESLICE.
func (v Value) AsByteSlice() []byte {
	if v.vtype != BYTESLICE {
		return nil
	}
	return v.asByteSlice()
}

func (v Value) asByteSlice() []byte {
	return []byte(v.stringly)
}

type unknownValueType struct{}

// AsInterface returns Value's data as any.
func (v Value) AsInterface() any {
	switch v.Type() {
	case BOOL:
		return v.AsBool()
	case BOOLSLICE:
		return v.asBoolSlice()
	case INT64:
		return v.AsInt64()
	case INT64SLICE:
		return v.asInt64Slice()
	case FLOAT64:
		return v.AsFloat64()
	case FLOAT64SLICE:
		return v.asFloat64Slice()
	case STRING:
		return v.stringly
	case STRINGSLICE:
		return v.asStringSlice()
	case BYTESLICE:
		return v.asByteSlice()
	case SLICE:
		return v.asSlice()
	case EMPTY:
		return nil
	}
	return unknownValueType{}
}

// String returns a string representation of Value using the
// [OpenTelemetry AnyValue representation for non-OTLP protocols] rules.
//
// Strings are returned as-is without JSON quoting, booleans and integers use
// JSON literals, floating-point values use JSON numbers except that NaN and
// ±Inf are rendered as NaN, Infinity, and -Infinity, byte slices are
// base64-encoded, empty values are the empty string, and slices are encoded as
// JSON arrays. String, byte, and special floating-point values inside arrays
// are encoded as JSON strings, and empty values inside arrays are encoded as
// null.
//
// [OpenTelemetry AnyValue representation for non-OTLP protocols]: https://opentelemetry.io/docs/specs/otel/common/#anyvalue-representation-for-non-otlp-protocols
func (v Value) String() string {
	switch v.Type() {
	case BOOL:
		return strconv.FormatBool(v.AsBool())
	case BOOLSLICE:
		return formatBoolSliceValue(v.slice)
	case INT64:
		return strconv.FormatInt(v.AsInt64(), 10)
	case INT64SLICE:
		return formatInt64SliceValue(v.slice)
	case FLOAT64:
		return formatFloat64(v.AsFloat64())
	case FLOAT64SLICE:
		return formatFloat64SliceValue(v.slice)
	case STRING:
		return v.stringly
	case STRINGSLICE:
		return formatStringSliceValue(v.slice)
	case BYTESLICE:
		return formatByteSlice(v.stringly)
	case SLICE:
		return formatValueSliceValue(v.slice)
	case EMPTY:
		return ""
	default:
		return "unknown"
	}
}

// Emit returns a string representation of Value's data.
//
// Deprecated: Use [Value.String] instead.
func (v Value) Emit() string {
	switch v.Type() {
	case BOOLSLICE:
		return fmt.Sprint(v.asBoolSlice())
	case BOOL:
		return strconv.FormatBool(v.AsBool())
	case INT64SLICE:
		j, err := json.Marshal(v.asInt64Slice())
		if err != nil {
			return fmt.Sprintf("invalid: %v", v.asInt64Slice())
		}
		return string(j)
	case INT64:
		return strconv.FormatInt(v.AsInt64(), 10)
	case FLOAT64SLICE:
		j, err := json.Marshal(v.asFloat64Slice())
		if err != nil {
			return fmt.Sprintf("invalid: %v", v.asFloat64Slice())
		}
		return string(j)
	case FLOAT64:
		return fmt.Sprint(v.AsFloat64())
	case STRINGSLICE:
		j, err := json.Marshal(v.asStringSlice())
		if err != nil {
			return fmt.Sprintf("invalid: %v", v.asStringSlice())
		}
		return string(j)
	case STRING:
		return v.stringly
	case BYTESLICE:
		return formatByteSlice(v.stringly)
	case SLICE:
		return formatValueSliceValue(v.slice)
	case EMPTY:
		return ""
	default:
		return "unknown"
	}
}

const (
	jsonArrayBracketsLen   = len("[]")
	boolArrayElemMaxLen    = len("false")
	int64ArrayElemMaxLen   = len("-9223372036854775808")
	float64ArrayElemMaxLen = len("-1.7976931348623157e+308")
	commaLen               = len(",")
)

func sliceValue(v []Value) any {
	switch len(v) {
	case 0:
		return [0]Value{}
	case 1:
		return [1]Value{v[0]}
	case 2:
		return [2]Value{v[0], v[1]}
	case 3:
		return [3]Value{v[0], v[1], v[2]}
	case 4:
		return [4]Value{v[0], v[1], v[2], v[3]}
	case 5:
		return [5]Value{v[0], v[1], v[2], v[3], v[4]}
	default:
		return sliceValueReflect(v)
	}
}

func sliceValueReflect(v []Value) any {
	cp := reflect.New(reflect.ArrayOf(len(v), reflect.TypeFor[Value]())).Elem()
	reflect.Copy(cp, reflect.ValueOf(v))
	return cp.Interface()
}

func formatBoolSliceValue(v any) string {
	switch vals := v.(type) {
	case [0]bool:
		return "[]"
	case [1]bool:
		return formatBoolSlice(vals[:])
	case [2]bool:
		return formatBoolSlice(vals[:])
	case [3]bool:
		return formatBoolSlice(vals[:])
	default:
		return formatBoolSliceReflect(v)
	}
}

func formatBoolSlice(vals []bool) string {
	var b strings.Builder
	appendBoolSlice(&b, vals)
	return b.String()
}

func formatBoolSliceReflect(v any) string {
	var b strings.Builder
	appendBoolSliceReflect(&b, reflect.ValueOf(v))
	return b.String()
}

func appendBoolSliceValue(dst *strings.Builder, v any) {
	switch vals := v.(type) {
	case [0]bool:
		_, _ = dst.WriteString("[]")
	case [1]bool:
		appendBoolSlice(dst, vals[:])
	case [2]bool:
		appendBoolSlice(dst, vals[:])
	case [3]bool:
		appendBoolSlice(dst, vals[:])
	default:
		appendBoolSliceReflect(dst, reflect.ValueOf(v))
	}
}

func appendBoolSlice(dst *strings.Builder, vals []bool) {
	dst.Grow(jsonArrayBracketsLen + len(vals)*(boolArrayElemMaxLen+commaLen))
	_ = dst.WriteByte('[')
	for i, val := range vals {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		if val {
			_, _ = dst.WriteString("true")
		} else {
			_, _ = dst.WriteString("false")
		}
	}
	_ = dst.WriteByte(']')
}

func appendBoolSliceReflect(dst *strings.Builder, rv reflect.Value) {
	dst.Grow(jsonArrayBracketsLen + rv.Len()*(boolArrayElemMaxLen+commaLen))
	_ = dst.WriteByte('[')
	for i := 0; i < rv.Len(); i++ {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		if rv.Index(i).Bool() {
			_, _ = dst.WriteString("true")
		} else {
			_, _ = dst.WriteString("false")
		}
	}
	_ = dst.WriteByte(']')
}

func formatInt64SliceValue(v any) string {
	switch vals := v.(type) {
	case [0]int64:
		return "[]"
	case [1]int64:
		return formatInt64Slice(vals[:])
	case [2]int64:
		return formatInt64Slice(vals[:])
	case [3]int64:
		return formatInt64Slice(vals[:])
	default:
		return formatInt64SliceReflect(v)
	}
}

func formatInt64Slice(vals []int64) string {
	var b strings.Builder
	appendInt64Slice(&b, vals)
	return b.String()
}

func formatInt64SliceReflect(v any) string {
	var b strings.Builder
	appendInt64SliceReflect(&b, reflect.ValueOf(v))
	return b.String()
}

func appendInt64SliceValue(dst *strings.Builder, v any) {
	switch vals := v.(type) {
	case [0]int64:
		_, _ = dst.WriteString("[]")
	case [1]int64:
		appendInt64Slice(dst, vals[:])
	case [2]int64:
		appendInt64Slice(dst, vals[:])
	case [3]int64:
		appendInt64Slice(dst, vals[:])
	default:
		appendInt64SliceReflect(dst, reflect.ValueOf(v))
	}
}

func appendInt64Slice(dst *strings.Builder, vals []int64) {
	dst.Grow(jsonArrayBracketsLen + len(vals)*(int64ArrayElemMaxLen+commaLen))
	_ = dst.WriteByte('[')

	var buf [int64ArrayElemMaxLen]byte
	for i, val := range vals {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		out := strconv.AppendInt(buf[:0], val, 10)
		_, _ = dst.Write(out)
	}

	_ = dst.WriteByte(']')
}

func appendInt64SliceReflect(dst *strings.Builder, rv reflect.Value) {
	dst.Grow(jsonArrayBracketsLen + rv.Len()*(int64ArrayElemMaxLen+commaLen))
	_ = dst.WriteByte('[')

	var scratch [int64ArrayElemMaxLen]byte
	for i := 0; i < rv.Len(); i++ {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		out := strconv.AppendInt(scratch[:0], rv.Index(i).Int(), 10)
		_, _ = dst.Write(out)
	}

	_ = dst.WriteByte(']')
}

func formatFloat64(v float64) string {
	switch {
	case math.IsNaN(v):
		return "NaN"
	case math.IsInf(v, 1):
		return "Infinity"
	case math.IsInf(v, -1):
		return "-Infinity"
	default:
		return strconv.FormatFloat(v, 'g', -1, 64)
	}
}

func formatFloat64SliceValue(v any) string {
	switch vals := v.(type) {
	case [0]float64:
		return "[]"
	case [1]float64:
		return formatFloat64Slice(vals[:])
	case [2]float64:
		return formatFloat64Slice(vals[:])
	case [3]float64:
		return formatFloat64Slice(vals[:])
	default:
		return formatFloat64SliceReflect(v)
	}
}

func formatFloat64Slice(vals []float64) string {
	var b strings.Builder
	appendFloat64Slice(&b, vals)
	return b.String()
}

func formatFloat64SliceReflect(v any) string {
	var b strings.Builder
	appendFloat64SliceReflect(&b, reflect.ValueOf(v))
	return b.String()
}

func appendFloat64SliceValue(dst *strings.Builder, v any) {
	switch vals := v.(type) {
	case [0]float64:
		_, _ = dst.WriteString("[]")
	case [1]float64:
		appendFloat64Slice(dst, vals[:])
	case [2]float64:
		appendFloat64Slice(dst, vals[:])
	case [3]float64:
		appendFloat64Slice(dst, vals[:])
	default:
		appendFloat64SliceReflect(dst, reflect.ValueOf(v))
	}
}

func appendFloat64Slice(dst *strings.Builder, vals []float64) {
	dst.Grow(jsonArrayBracketsLen + len(vals)*(float64ArrayElemMaxLen+commaLen))
	_ = dst.WriteByte('[')

	var buf [float64ArrayElemMaxLen]byte
	for i, val := range vals {
		if i > 0 {
			_ = dst.WriteByte(',')
		}

		switch {
		case math.IsNaN(val):
			_, _ = dst.WriteString(`"NaN"`)
		case math.IsInf(val, 1):
			_, _ = dst.WriteString(`"Infinity"`)
		case math.IsInf(val, -1):
			_, _ = dst.WriteString(`"-Infinity"`)
		default:
			out := strconv.AppendFloat(buf[:0], val, 'g', -1, 64)
			_, _ = dst.Write(out)
		}
	}

	_ = dst.WriteByte(']')
}

func appendFloat64SliceReflect(dst *strings.Builder, rv reflect.Value) {
	dst.Grow(jsonArrayBracketsLen + rv.Len()*(float64ArrayElemMaxLen+commaLen))
	_ = dst.WriteByte('[')

	var scratch [float64ArrayElemMaxLen]byte
	for i := 0; i < rv.Len(); i++ {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		val := rv.Index(i).Float()
		switch {
		case math.IsNaN(val):
			_, _ = dst.WriteString(`"NaN"`)
		case math.IsInf(val, 1):
			_, _ = dst.WriteString(`"Infinity"`)
		case math.IsInf(val, -1):
			_, _ = dst.WriteString(`"-Infinity"`)
		default:
			out := strconv.AppendFloat(scratch[:0], val, 'g', -1, 64)
			_, _ = dst.Write(out)
		}
	}

	_ = dst.WriteByte(']')
}

func formatStringSliceValue(v any) string {
	switch vals := v.(type) {
	case [0]string:
		return "[]"
	case [1]string:
		return formatStringSlice(vals[:])
	case [2]string:
		return formatStringSlice(vals[:])
	case [3]string:
		return formatStringSlice(vals[:])
	default:
		return formatStringSliceReflect(v)
	}
}

func formatStringSlice(vals []string) string {
	var b strings.Builder
	appendStringSlice(&b, vals)
	return b.String()
}

func formatStringSliceReflect(v any) string {
	var b strings.Builder
	appendStringSliceReflect(&b, reflect.ValueOf(v))
	return b.String()
}

func appendStringSliceValue(dst *strings.Builder, v any) {
	switch vals := v.(type) {
	case [0]string:
		_, _ = dst.WriteString("[]")
	case [1]string:
		appendStringSlice(dst, vals[:])
	case [2]string:
		appendStringSlice(dst, vals[:])
	case [3]string:
		appendStringSlice(dst, vals[:])
	default:
		appendStringSliceReflect(dst, reflect.ValueOf(v))
	}
}

func appendStringSlice(dst *strings.Builder, vals []string) {
	size := jsonArrayBracketsLen
	for _, val := range vals {
		size += len(val) + commaLen + 2 // Account for JSON string quotes and comma.
	}

	dst.Grow(size)
	_ = dst.WriteByte('[')
	for i, val := range vals {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		appendJSONString(dst, val)
	}
	_ = dst.WriteByte(']')
}

func appendStringSliceReflect(dst *strings.Builder, rv reflect.Value) {
	size := jsonArrayBracketsLen
	for i := 0; i < rv.Len(); i++ {
		size += len(rv.Index(i).String()) + commaLen + 2 // Account for JSON string quotes and comma.
	}

	dst.Grow(size)
	_ = dst.WriteByte('[')
	for i := 0; i < rv.Len(); i++ {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		appendJSONString(dst, rv.Index(i).String())
	}
	_ = dst.WriteByte(']')
}

func formatByteSlice(v string) string {
	var b strings.Builder
	appendBase64(&b, v)
	return b.String()
}

func formatValueSliceValue(v any) string {
	switch vals := v.(type) {
	case [0]Value:
		return "[]"
	case [1]Value:
		return formatValueSlice(vals[:])
	case [2]Value:
		return formatValueSlice(vals[:])
	case [3]Value:
		return formatValueSlice(vals[:])
	case [4]Value:
		return formatValueSlice(vals[:])
	case [5]Value:
		return formatValueSlice(vals[:])
	default:
		return formatValueSliceReflect(v)
	}
}

func formatValueSlice(vals []Value) string {
	var b strings.Builder
	appendValueSlice(&b, vals)
	return b.String()
}

func formatValueSliceReflect(v any) string {
	var b strings.Builder
	appendValueSliceReflect(&b, reflect.ValueOf(v))
	return b.String()
}

func appendValueSliceValue(dst *strings.Builder, v any) {
	switch vals := v.(type) {
	case [0]Value:
		_, _ = dst.WriteString("[]")
	case [1]Value:
		appendValueSlice(dst, vals[:])
	case [2]Value:
		appendValueSlice(dst, vals[:])
	case [3]Value:
		appendValueSlice(dst, vals[:])
	case [4]Value:
		appendValueSlice(dst, vals[:])
	case [5]Value:
		appendValueSlice(dst, vals[:])
	default:
		appendValueSliceReflect(dst, reflect.ValueOf(v))
	}
}

func appendValueSlice(dst *strings.Builder, vals []Value) {
	// Estimate 10 bytes per value for small values and commas.
	dst.Grow(jsonArrayBracketsLen + len(vals)*commaLen + len(vals)*10)
	_ = dst.WriteByte('[')
	for i, val := range vals {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		appendJSONValue(dst, val)
	}
	_ = dst.WriteByte(']')
}

func appendValueSliceReflect(dst *strings.Builder, rv reflect.Value) {
	// Estimate 10 bytes per value for small values and commas.
	dst.Grow(jsonArrayBracketsLen + rv.Len()*commaLen + rv.Len()*10)
	_ = dst.WriteByte('[')
	for i := 0; i < rv.Len(); i++ {
		if i > 0 {
			_ = dst.WriteByte(',')
		}
		appendJSONValue(dst, rv.Index(i).Interface().(Value))
	}
	_ = dst.WriteByte(']')
}

func appendJSONValue(dst *strings.Builder, v Value) {
	switch v.Type() {
	case BOOL:
		if v.AsBool() {
			_, _ = dst.WriteString("true")
		} else {
			_, _ = dst.WriteString("false")
		}
	case BOOLSLICE:
		appendBoolSliceValue(dst, v.slice)
	case INT64:
		var buf [int64ArrayElemMaxLen]byte
		out := strconv.AppendInt(buf[:0], v.AsInt64(), 10)
		_, _ = dst.Write(out)
	case INT64SLICE:
		appendInt64SliceValue(dst, v.slice)
	case FLOAT64:
		val := v.AsFloat64()
		switch {
		case math.IsNaN(val):
			appendJSONString(dst, "NaN")
		case math.IsInf(val, 1):
			appendJSONString(dst, "Infinity")
		case math.IsInf(val, -1):
			appendJSONString(dst, "-Infinity")
		default:
			var buf [float64ArrayElemMaxLen]byte
			out := strconv.AppendFloat(buf[:0], val, 'g', -1, 64)
			_, _ = dst.Write(out)
		}
	case FLOAT64SLICE:
		appendFloat64SliceValue(dst, v.slice)
	case STRING:
		appendJSONString(dst, v.stringly)
	case STRINGSLICE:
		appendStringSliceValue(dst, v.slice)
	case BYTESLICE:
		_ = dst.WriteByte('"')
		appendBase64(dst, v.stringly)
		_ = dst.WriteByte('"')
	case SLICE:
		appendValueSliceValue(dst, v.slice)
	case EMPTY:
		_, _ = dst.WriteString("null")
	default:
		appendJSONString(dst, "unknown")
	}
}

// appendJSONString appends s to dst as a JSON string literal.
//
// This is adapted from the Go standard library's encoding/json
// [appendString implementation]. It keeps the same escaping behavior we need
// here, but writes directly into a strings.Builder and intentionally does not
// apply HTML escaping because the OpenTelemetry non-OTLP AnyValue representation
// only requires JSON array string encoding. We inline this instead of using
// encoding/json so slice formatting avoids allocations and reflection.
//
// [appendString implementation]: https://github.com/golang/go/blob/3b5954c6349d31465dca409b45ab6597e0942d9f/src/encoding/json/encode.go#L998-L1064
func appendJSONString(dst *strings.Builder, s string) {
	const hex = "0123456789abcdef" // For escaping bytes to hex.

	_ = dst.WriteByte('"')
	start := 0

	for i := 0; i < len(s); {
		if c := s[i]; c < utf8.RuneSelf {
			if c >= 0x20 && c != '\\' && c != '"' {
				i++
				continue
			}

			if start < i {
				_, _ = dst.WriteString(s[start:i])
			}

			switch c {
			case '\\', '"':
				_ = dst.WriteByte('\\')
				_ = dst.WriteByte(c)
			case '\b':
				_, _ = dst.WriteString(`\b`)
			case '\f':
				_, _ = dst.WriteString(`\f`)
			case '\n':
				_, _ = dst.WriteString(`\n`)
			case '\r':
				_, _ = dst.WriteString(`\r`)
			case '\t':
				_, _ = dst.WriteString(`\t`)
			default:
				_, _ = dst.WriteString(`\u00`)
				_ = dst.WriteByte(hex[c>>4])
				_ = dst.WriteByte(hex[c&0x0f])
			}

			i++
			start = i
			continue
		}

		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			if start < i {
				_, _ = dst.WriteString(s[start:i])
			}
			// Match encoding/json by replacing invalid UTF-8 with U+FFFD.
			_, _ = dst.WriteString(`\ufffd`)
			i++
			start = i
			continue
		}

		if r == '\u2028' || r == '\u2029' {
			if start < i {
				_, _ = dst.WriteString(s[start:i])
			}
			// Escape JSONP-sensitive separators unconditionally, like encoding/json.
			_, _ = dst.WriteString(`\u202`)
			_ = dst.WriteByte(hex[r&0x0f])
			i += size
			start = i
			continue
		}

		i += size
	}

	if start < len(s) {
		_, _ = dst.WriteString(s[start:])
	}
	_ = dst.WriteByte('"')
}

// This is adapted from the Go standard library's encoding/base64
// [Encoding.Encode implementation]. It keeps the same encoding behavior we need
// here, but writes directly into a strings.Builder. We inline this instead of using
// encoding/base64 to avoid allocations.
//
// [Encoding.Encode implementation]: https://github.com/golang/go/blob/3b5954c6349d31465dca409b45ab6597e0942d9f/src/encoding/base64/base64.go#L139-L189
func appendBase64(dst *strings.Builder, s string) {
	const encode = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

	dst.Grow(base64.StdEncoding.EncodedLen(len(s)))

	i := 0
	for ; i+2 < len(s); i += 3 {
		n := uint32(s[i])<<16 | uint32(s[i+1])<<8 | uint32(s[i+2])
		_ = dst.WriteByte(encode[n>>18&0x3f])
		_ = dst.WriteByte(encode[n>>12&0x3f])
		_ = dst.WriteByte(encode[n>>6&0x3f])
		_ = dst.WriteByte(encode[n&0x3f])
	}

	switch len(s) - i {
	case 1:
		n := uint32(s[i]) << 16
		_ = dst.WriteByte(encode[n>>18&0x3f])
		_ = dst.WriteByte(encode[n>>12&0x3f])
		_ = dst.WriteByte('=')
		_ = dst.WriteByte('=')
	case 2:
		n := uint32(s[i])<<16 | uint32(s[i+1])<<8
		_ = dst.WriteByte(encode[n>>18&0x3f])
		_ = dst.WriteByte(encode[n>>12&0x3f])
		_ = dst.WriteByte(encode[n>>6&0x3f])
		_ = dst.WriteByte('=')
	}
}

// MarshalJSON returns the JSON encoding of the Value.
func (v Value) MarshalJSON() ([]byte, error) {
	var jsonVal struct {
		Type  string
		Value any
	}
	jsonVal.Type = v.Type().String()
	jsonVal.Value = v.AsInterface()
	return json.Marshal(jsonVal)
}
