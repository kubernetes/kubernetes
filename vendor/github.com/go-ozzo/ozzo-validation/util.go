// Copyright 2016 Qiang Xue. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package validation

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"reflect"
	"time"
)

var (
	bytesType  = reflect.TypeOf([]byte(nil))
	valuerType = reflect.TypeOf((*driver.Valuer)(nil)).Elem()
)

// EnsureString ensures the given value is a string.
// If the value is a byte slice, it will be typecast into a string.
// An error is returned otherwise.
func EnsureString(value interface{}) (string, error) {
	v := reflect.ValueOf(value)
	if v.Kind() == reflect.String {
		return v.String(), nil
	}
	if v.Type() == bytesType {
		return string(v.Interface().([]byte)), nil
	}
	return "", errors.New("must be either a string or byte slice")
}

// StringOrBytes typecasts a value into a string or byte slice.
// Boolean flags are returned to indicate if the typecasting succeeds or not.
func StringOrBytes(value interface{}) (isString bool, str string, isBytes bool, bs []byte) {
	v := reflect.ValueOf(value)
	if v.Kind() == reflect.String {
		str = v.String()
		isString = true
	} else if v.Kind() == reflect.Slice && v.Type() == bytesType {
		bs = v.Interface().([]byte)
		isBytes = true
	}
	return
}

// LengthOfValue returns the length of a value that is a string, slice, map, or array.
// An error is returned for all other types.
func LengthOfValue(value interface{}) (int, error) {
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.String, reflect.Slice, reflect.Map, reflect.Array:
		return v.Len(), nil
	}
	return 0, fmt.Errorf("cannot get the length of %v", v.Kind())
}

// ToInt converts the given value to an int64.
// An error is returned for all incompatible types.
func ToInt(value interface{}) (int64, error) {
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int(), nil
	}
	return 0, fmt.Errorf("cannot convert %v to int64", v.Kind())
}

// ToUint converts the given value to an uint64.
// An error is returned for all incompatible types.
func ToUint(value interface{}) (uint64, error) {
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint(), nil
	}
	return 0, fmt.Errorf("cannot convert %v to uint64", v.Kind())
}

// ToFloat converts the given value to a float64.
// An error is returned for all incompatible types.
func ToFloat(value interface{}) (float64, error) {
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Float32, reflect.Float64:
		return v.Float(), nil
	}
	return 0, fmt.Errorf("cannot convert %v to float64", v.Kind())
}

// IsEmpty checks if a value is empty or not.
// A value is considered empty if
// - integer, float: zero
// - bool: false
// - string, array: len() == 0
// - slice, map: nil or len() == 0
// - interface, pointer: nil or the referenced value is empty
func IsEmpty(value interface{}) bool {
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.String, reflect.Array, reflect.Map, reflect.Slice:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Invalid:
		return true
	case reflect.Interface, reflect.Ptr:
		if v.IsNil() {
			return true
		}
		return IsEmpty(v.Elem().Interface())
	case reflect.Struct:
		v, ok := value.(time.Time)
		if ok && v.IsZero() {
			return true
		}
	}

	return false
}

// Indirect returns the value that the given interface or pointer references to.
// If the value implements driver.Valuer, it will deal with the value returned by
// the Value() method instead. A boolean value is also returned to indicate if
// the value is nil or not (only applicable to interface, pointer, map, and slice).
// If the value is neither an interface nor a pointer, it will be returned back.
func Indirect(value interface{}) (interface{}, bool) {
	rv := reflect.ValueOf(value)
	kind := rv.Kind()
	switch kind {
	case reflect.Invalid:
		return nil, true
	case reflect.Ptr, reflect.Interface:
		if rv.IsNil() {
			return nil, true
		}
		return Indirect(rv.Elem().Interface())
	case reflect.Slice, reflect.Map, reflect.Func, reflect.Chan:
		if rv.IsNil() {
			return nil, true
		}
	}

	if rv.Type().Implements(valuerType) {
		return indirectValuer(value.(driver.Valuer))
	}

	return value, false
}

func indirectValuer(valuer driver.Valuer) (interface{}, bool) {
	if value, err := valuer.Value(); value != nil && err == nil {
		return Indirect(value)
	}
	return nil, true
}
