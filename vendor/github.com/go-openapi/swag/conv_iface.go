// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import (
	"time"

	"github.com/go-openapi/swag/conv"
)

// IsFloat64AJSONInteger allows for integers [-2^53, 2^53-1] inclusive.
//
// Deprecated: use [conv.IsFloat64AJSONInteger] instead.
func IsFloat64AJSONInteger(f float64) bool { return conv.IsFloat64AJSONInteger(f) }

// ConvertBool turns a string into a boolean.
//
// Deprecated: use [conv.ConvertBool] instead.
func ConvertBool(str string) (bool, error) { return conv.ConvertBool(str) }

// ConvertFloat32 turns a string into a float32.
//
// Deprecated: use [conv.ConvertFloat32] instead. Alternatively, you may use the generic version [conv.ConvertFloat].
func ConvertFloat32(str string) (float32, error) { return conv.ConvertFloat[float32](str) }

// ConvertFloat64 turns a string into a float64.
//
// Deprecated: use [conv.ConvertFloat64] instead. Alternatively, you may use the generic version [conv.ConvertFloat].
func ConvertFloat64(str string) (float64, error) { return conv.ConvertFloat[float64](str) }

// ConvertInt8 turns a string into an int8.
//
// Deprecated: use [conv.ConvertInt8] instead. Alternatively, you may use the generic version [conv.ConvertInteger].
func ConvertInt8(str string) (int8, error) { return conv.ConvertInteger[int8](str) }

// ConvertInt16 turns a string into an int16.
//
// Deprecated: use [conv.ConvertInt16] instead. Alternatively, you may use the generic version [conv.ConvertInteger].
func ConvertInt16(str string) (int16, error) { return conv.ConvertInteger[int16](str) }

// ConvertInt32 turns a string into an int32.
//
// Deprecated: use [conv.ConvertInt32] instead. Alternatively, you may use the generic version [conv.ConvertInteger].
func ConvertInt32(str string) (int32, error) { return conv.ConvertInteger[int32](str) }

// ConvertInt64 turns a string into an int64.
//
// Deprecated: use [conv.ConvertInt64] instead. Alternatively, you may use the generic version [conv.ConvertInteger].
func ConvertInt64(str string) (int64, error) { return conv.ConvertInteger[int64](str) }

// ConvertUint8 turns a string into an uint8.
//
// Deprecated: use [conv.ConvertUint8] instead. Alternatively, you may use the generic version [conv.ConvertUinteger].
func ConvertUint8(str string) (uint8, error) { return conv.ConvertUinteger[uint8](str) }

// ConvertUint16 turns a string into an uint16.
//
// Deprecated: use [conv.ConvertUint16] instead. Alternatively, you may use the generic version [conv.ConvertUinteger].
func ConvertUint16(str string) (uint16, error) { return conv.ConvertUinteger[uint16](str) }

// ConvertUint32 turns a string into an uint32.
//
// Deprecated: use [conv.ConvertUint32] instead. Alternatively, you may use the generic version [conv.ConvertUinteger].
func ConvertUint32(str string) (uint32, error) { return conv.ConvertUinteger[uint32](str) }

// ConvertUint64 turns a string into an uint64.
//
// Deprecated: use [conv.ConvertUint64] instead. Alternatively, you may use the generic version [conv.ConvertUinteger].
func ConvertUint64(str string) (uint64, error) { return conv.ConvertUinteger[uint64](str) }

// FormatBool turns a boolean into a string.
//
// Deprecated: use [conv.FormatBool] instead.
func FormatBool(value bool) string { return conv.FormatBool(value) }

// FormatFloat32 turns a float32 into a string.
//
// Deprecated: use [conv.FormatFloat] instead.
func FormatFloat32(value float32) string { return conv.FormatFloat(value) }

// FormatFloat64 turns a float64 into a string.
//
// Deprecated: use [conv.FormatFloat] instead.
func FormatFloat64(value float64) string { return conv.FormatFloat(value) }

// FormatInt8 turns an int8 into a string.
//
// Deprecated: use [conv.FormatInteger] instead.
func FormatInt8(value int8) string { return conv.FormatInteger(value) }

// FormatInt16 turns an int16 into a string.
//
// Deprecated: use [conv.FormatInteger] instead.
func FormatInt16(value int16) string { return conv.FormatInteger(value) }

// FormatInt32 turns an int32 into a string
//
// Deprecated: use [conv.FormatInteger] instead.
func FormatInt32(value int32) string { return conv.FormatInteger(value) }

// FormatInt64 turns an int64 into a string.
//
// Deprecated: use [conv.FormatInteger] instead.
func FormatInt64(value int64) string { return conv.FormatInteger(value) }

// FormatUint8 turns an uint8 into a string.
//
// Deprecated: use [conv.FormatUinteger] instead.
func FormatUint8(value uint8) string { return conv.FormatUinteger(value) }

// FormatUint16 turns an uint16 into a string.
//
// Deprecated: use [conv.FormatUinteger] instead.
func FormatUint16(value uint16) string { return conv.FormatUinteger(value) }

// FormatUint32 turns an uint32 into a string.
//
// Deprecated: use [conv.FormatUinteger] instead.
func FormatUint32(value uint32) string { return conv.FormatUinteger(value) }

// FormatUint64 turns an uint64 into a string.
//
// Deprecated: use [conv.FormatUinteger] instead.
func FormatUint64(value uint64) string { return conv.FormatUinteger(value) }

// String turn a pointer to of the string value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func String(v string) *string { return conv.Pointer(v) }

// StringValue turn the value of the string pointer passed in or
// "" if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func StringValue(v *string) string { return conv.Value(v) }

// StringSlice converts a slice of string values into a slice of string pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func StringSlice(src []string) []*string { return conv.PointerSlice(src) }

// StringValueSlice converts a slice of string pointers into a slice of string values.
//
// Deprecated: use [conv.ValueSlice] instead.
func StringValueSlice(src []*string) []string { return conv.ValueSlice(src) }

// StringMap converts a string map of string values into a string map of string pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func StringMap(src map[string]string) map[string]*string { return conv.PointerMap(src) }

// StringValueMap converts a string map of string pointers into a string map of string values.
//
// Deprecated: use [conv.ValueMap] instead.
func StringValueMap(src map[string]*string) map[string]string { return conv.ValueMap(src) }

// Bool turn a pointer to of the bool value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Bool(v bool) *bool { return conv.Pointer(v) }

// BoolValue turn the value of the bool pointer passed in or false if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func BoolValue(v *bool) bool { return conv.Value(v) }

// BoolSlice converts a slice of bool values into a slice of bool pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func BoolSlice(src []bool) []*bool { return conv.PointerSlice(src) }

// BoolValueSlice converts a slice of bool pointers into a slice of bool values.
//
// Deprecated: use [conv.ValueSlice] instead.
func BoolValueSlice(src []*bool) []bool { return conv.ValueSlice(src) }

// BoolMap converts a string map of bool values into a string map of bool pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func BoolMap(src map[string]bool) map[string]*bool { return conv.PointerMap(src) }

// BoolValueMap converts a string map of bool pointers into a string map of bool values.
//
// Deprecated: use [conv.ValueMap] instead.
func BoolValueMap(src map[string]*bool) map[string]bool { return conv.ValueMap(src) }

// Int turn a pointer to of the int value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Int(v int) *int { return conv.Pointer(v) }

// IntValue turn the value of the int pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func IntValue(v *int) int { return conv.Value(v) }

// IntSlice converts a slice of int values into a slice of int pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func IntSlice(src []int) []*int { return conv.PointerSlice(src) }

// IntValueSlice converts a slice of int pointers into a slice of int values.
//
// Deprecated: use [conv.ValueSlice] instead.
func IntValueSlice(src []*int) []int { return conv.ValueSlice(src) }

// IntMap converts a string map of int values into a string map of int pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func IntMap(src map[string]int) map[string]*int { return conv.PointerMap(src) }

// IntValueMap converts a string map of int pointers into a string map of int values.
//
// Deprecated: use [conv.ValueMap] instead.
func IntValueMap(src map[string]*int) map[string]int { return conv.ValueMap(src) }

// Int32 turn a pointer to of the int32 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Int32(v int32) *int32 { return conv.Pointer(v) }

// Int32Value turn the value of the int32 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Int32Value(v *int32) int32 { return conv.Value(v) }

// Int32Slice converts a slice of int32 values into a slice of int32 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Int32Slice(src []int32) []*int32 { return conv.PointerSlice(src) }

// Int32ValueSlice converts a slice of int32 pointers into a slice of int32 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Int32ValueSlice(src []*int32) []int32 { return conv.ValueSlice(src) }

// Int32Map converts a string map of int32 values into a string map of int32 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Int32Map(src map[string]int32) map[string]*int32 { return conv.PointerMap(src) }

// Int32ValueMap converts a string map of int32 pointers into a string map of int32 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Int32ValueMap(src map[string]*int32) map[string]int32 { return conv.ValueMap(src) }

// Int64 turn a pointer to of the int64 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Int64(v int64) *int64 { return conv.Pointer(v) }

// Int64Value turn the value of the int64 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Int64Value(v *int64) int64 { return conv.Value(v) }

// Int64Slice converts a slice of int64 values into a slice of int64 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Int64Slice(src []int64) []*int64 { return conv.PointerSlice(src) }

// Int64ValueSlice converts a slice of int64 pointers into a slice of int64 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Int64ValueSlice(src []*int64) []int64 { return conv.ValueSlice(src) }

// Int64Map converts a string map of int64 values into a string map of int64 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Int64Map(src map[string]int64) map[string]*int64 { return conv.PointerMap(src) }

// Int64ValueMap converts a string map of int64 pointers into a string map of int64 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Int64ValueMap(src map[string]*int64) map[string]int64 { return conv.ValueMap(src) }

// Uint16 turn a pointer to of the uint16 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Uint16(v uint16) *uint16 { return conv.Pointer(v) }

// Uint16Value turn the value of the uint16 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Uint16Value(v *uint16) uint16 { return conv.Value(v) }

// Uint16Slice converts a slice of uint16 values into a slice of uint16 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Uint16Slice(src []uint16) []*uint16 { return conv.PointerSlice(src) }

// Uint16ValueSlice converts a slice of uint16 pointers into a slice of uint16 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Uint16ValueSlice(src []*uint16) []uint16 { return conv.ValueSlice(src) }

// Uint16Map converts a string map of uint16 values into a string map of uint16 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Uint16Map(src map[string]uint16) map[string]*uint16 { return conv.PointerMap(src) }

// Uint16ValueMap converts a string map of uint16 pointers into a string map of uint16 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Uint16ValueMap(src map[string]*uint16) map[string]uint16 { return conv.ValueMap(src) }

// Uint turn a pointer to of the uint value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Uint(v uint) *uint { return conv.Pointer(v) }

// UintValue turn the value of the uint pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func UintValue(v *uint) uint { return conv.Value(v) }

// UintSlice converts a slice of uint values into a slice of uint pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func UintSlice(src []uint) []*uint { return conv.PointerSlice(src) }

// UintValueSlice converts a slice of uint pointers into a slice of uint values.
//
// Deprecated: use [conv.ValueSlice] instead.
func UintValueSlice(src []*uint) []uint { return conv.ValueSlice(src) }

// UintMap converts a string map of uint values into a string map of uint pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func UintMap(src map[string]uint) map[string]*uint { return conv.PointerMap(src) }

// UintValueMap converts a string map of uint pointers into a string map of uint values.
//
// Deprecated: use [conv.ValueMap] instead.
func UintValueMap(src map[string]*uint) map[string]uint { return conv.ValueMap(src) }

// Uint32 turn a pointer to of the uint32 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Uint32(v uint32) *uint32 { return conv.Pointer(v) }

// Uint32Value turn the value of the uint32 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Uint32Value(v *uint32) uint32 { return conv.Value(v) }

// Uint32Slice converts a slice of uint32 values into a slice of uint32 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Uint32Slice(src []uint32) []*uint32 { return conv.PointerSlice(src) }

// Uint32ValueSlice converts a slice of uint32 pointers into a slice of uint32 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Uint32ValueSlice(src []*uint32) []uint32 { return conv.ValueSlice(src) }

// Uint32Map converts a string map of uint32 values into a string map of uint32 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Uint32Map(src map[string]uint32) map[string]*uint32 { return conv.PointerMap(src) }

// Uint32ValueMap converts a string map of uint32 pointers into a string map of uint32 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Uint32ValueMap(src map[string]*uint32) map[string]uint32 { return conv.ValueMap(src) }

// Uint64 turn a pointer to of the uint64 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Uint64(v uint64) *uint64 { return conv.Pointer(v) }

// Uint64Value turn the value of the uint64 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Uint64Value(v *uint64) uint64 { return conv.Value(v) }

// Uint64Slice converts a slice of uint64 values into a slice of uint64 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Uint64Slice(src []uint64) []*uint64 { return conv.PointerSlice(src) }

// Uint64ValueSlice converts a slice of uint64 pointers into a slice of uint64 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Uint64ValueSlice(src []*uint64) []uint64 { return conv.ValueSlice(src) }

// Uint64Map converts a string map of uint64 values into a string map of uint64 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Uint64Map(src map[string]uint64) map[string]*uint64 { return conv.PointerMap(src) }

// Uint64ValueMap converts a string map of uint64 pointers into a string map of uint64 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Uint64ValueMap(src map[string]*uint64) map[string]uint64 { return conv.ValueMap(src) }

// Float32 turn a pointer to of the float32 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Float32(v float32) *float32 { return conv.Pointer(v) }

// Float32Value turn the value of the float32 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Float32Value(v *float32) float32 { return conv.Value(v) }

// Float32Slice converts a slice of float32 values into a slice of float32 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Float32Slice(src []float32) []*float32 { return conv.PointerSlice(src) }

// Float32ValueSlice converts a slice of float32 pointers into a slice of float32 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Float32ValueSlice(src []*float32) []float32 { return conv.ValueSlice(src) }

// Float32Map converts a string map of float32 values into a string map of float32 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Float32Map(src map[string]float32) map[string]*float32 { return conv.PointerMap(src) }

// Float32ValueMap converts a string map of float32 pointers into a string map of float32 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Float32ValueMap(src map[string]*float32) map[string]float32 { return conv.ValueMap(src) }

// Float64 turn a pointer to of the float64 value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Float64(v float64) *float64 { return conv.Pointer(v) }

// Float64Value turn the value of the float64 pointer passed in or 0 if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func Float64Value(v *float64) float64 { return conv.Value(v) }

// Float64Slice converts a slice of float64 values into a slice of float64 pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func Float64Slice(src []float64) []*float64 { return conv.PointerSlice(src) }

// Float64ValueSlice converts a slice of float64 pointers into a slice of float64 values.
//
// Deprecated: use [conv.ValueSlice] instead.
func Float64ValueSlice(src []*float64) []float64 { return conv.ValueSlice(src) }

// Float64Map converts a string map of float64 values into a string map of float64 pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func Float64Map(src map[string]float64) map[string]*float64 { return conv.PointerMap(src) }

// Float64ValueMap converts a string map of float64 pointers into a string map of float64 values.
//
// Deprecated: use [conv.ValueMap] instead.
func Float64ValueMap(src map[string]*float64) map[string]float64 { return conv.ValueMap(src) }

// Time turn a pointer to of the time.Time value passed in.
//
// Deprecated: use [conv.Pointer] instead.
func Time(v time.Time) *time.Time { return conv.Pointer(v) }

// TimeValue turn the value of the time.Time pointer passed in or time.Time{} if the pointer is nil.
//
// Deprecated: use [conv.Value] instead.
func TimeValue(v *time.Time) time.Time { return conv.Value(v) }

// TimeSlice converts a slice of time.Time values into a slice of time.Time pointers.
//
// Deprecated: use [conv.PointerSlice] instead.
func TimeSlice(src []time.Time) []*time.Time { return conv.PointerSlice(src) }

// TimeValueSlice converts a slice of time.Time pointers into a slice of time.Time values
//
// Deprecated: use [conv.ValueSlice] instead.
func TimeValueSlice(src []*time.Time) []time.Time { return conv.ValueSlice(src) }

// TimeMap converts a string map of time.Time values into a string map of time.Time pointers.
//
// Deprecated: use [conv.PointerMap] instead.
func TimeMap(src map[string]time.Time) map[string]*time.Time { return conv.PointerMap(src) }

// TimeValueMap converts a string map of time.Time pointers into a string map of time.Time values.
//
// Deprecated: use [conv.ValueMap] instead.
func TimeValueMap(src map[string]*time.Time) map[string]time.Time { return conv.ValueMap(src) }
