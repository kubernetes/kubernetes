// Copyright © 2014 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package cast

import (
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"reflect"
	"strconv"
	"strings"
	"time"
)

var errNegativeNotAllowed = errors.New("unable to cast negative value")

// ToTimeE casts an interface to a time.Time type.
func ToTimeE(i interface{}) (tim time.Time, err error) {
	return ToTimeInDefaultLocationE(i, time.UTC)
}

// ToTimeInDefaultLocationE casts an empty interface to time.Time,
// interpreting inputs without a timezone to be in the given location,
// or the local timezone if nil.
func ToTimeInDefaultLocationE(i interface{}, location *time.Location) (tim time.Time, err error) {
	i = indirect(i)

	switch v := i.(type) {
	case time.Time:
		return v, nil
	case string:
		return StringToDateInDefaultLocation(v, location)
	case int:
		return time.Unix(int64(v), 0), nil
	case int64:
		return time.Unix(v, 0), nil
	case int32:
		return time.Unix(int64(v), 0), nil
	case uint:
		return time.Unix(int64(v), 0), nil
	case uint64:
		return time.Unix(int64(v), 0), nil
	case uint32:
		return time.Unix(int64(v), 0), nil
	default:
		return time.Time{}, fmt.Errorf("unable to cast %#v of type %T to Time", i, i)
	}
}

// ToDurationE casts an interface to a time.Duration type.
func ToDurationE(i interface{}) (d time.Duration, err error) {
	i = indirect(i)

	switch s := i.(type) {
	case time.Duration:
		return s, nil
	case int, int64, int32, int16, int8, uint, uint64, uint32, uint16, uint8:
		d = time.Duration(ToInt64(s))
		return
	case float32, float64:
		d = time.Duration(ToFloat64(s))
		return
	case string:
		if strings.ContainsAny(s, "nsuµmh") {
			d, err = time.ParseDuration(s)
		} else {
			d, err = time.ParseDuration(s + "ns")
		}
		return
	default:
		err = fmt.Errorf("unable to cast %#v of type %T to Duration", i, i)
		return
	}
}

// ToBoolE casts an interface to a bool type.
func ToBoolE(i interface{}) (bool, error) {
	i = indirect(i)

	switch b := i.(type) {
	case bool:
		return b, nil
	case nil:
		return false, nil
	case int:
		if i.(int) != 0 {
			return true, nil
		}
		return false, nil
	case string:
		return strconv.ParseBool(i.(string))
	default:
		return false, fmt.Errorf("unable to cast %#v of type %T to bool", i, i)
	}
}

// ToFloat64E casts an interface to a float64 type.
func ToFloat64E(i interface{}) (float64, error) {
	i = indirect(i)

	switch s := i.(type) {
	case float64:
		return s, nil
	case float32:
		return float64(s), nil
	case int:
		return float64(s), nil
	case int64:
		return float64(s), nil
	case int32:
		return float64(s), nil
	case int16:
		return float64(s), nil
	case int8:
		return float64(s), nil
	case uint:
		return float64(s), nil
	case uint64:
		return float64(s), nil
	case uint32:
		return float64(s), nil
	case uint16:
		return float64(s), nil
	case uint8:
		return float64(s), nil
	case string:
		v, err := strconv.ParseFloat(s, 64)
		if err == nil {
			return v, nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to float64", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to float64", i, i)
	}
}

// ToFloat32E casts an interface to a float32 type.
func ToFloat32E(i interface{}) (float32, error) {
	i = indirect(i)

	switch s := i.(type) {
	case float64:
		return float32(s), nil
	case float32:
		return s, nil
	case int:
		return float32(s), nil
	case int64:
		return float32(s), nil
	case int32:
		return float32(s), nil
	case int16:
		return float32(s), nil
	case int8:
		return float32(s), nil
	case uint:
		return float32(s), nil
	case uint64:
		return float32(s), nil
	case uint32:
		return float32(s), nil
	case uint16:
		return float32(s), nil
	case uint8:
		return float32(s), nil
	case string:
		v, err := strconv.ParseFloat(s, 32)
		if err == nil {
			return float32(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to float32", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to float32", i, i)
	}
}

// ToInt64E casts an interface to an int64 type.
func ToInt64E(i interface{}) (int64, error) {
	i = indirect(i)

	switch s := i.(type) {
	case int:
		return int64(s), nil
	case int64:
		return s, nil
	case int32:
		return int64(s), nil
	case int16:
		return int64(s), nil
	case int8:
		return int64(s), nil
	case uint:
		return int64(s), nil
	case uint64:
		return int64(s), nil
	case uint32:
		return int64(s), nil
	case uint16:
		return int64(s), nil
	case uint8:
		return int64(s), nil
	case float64:
		return int64(s), nil
	case float32:
		return int64(s), nil
	case string:
		v, err := strconv.ParseInt(s, 0, 0)
		if err == nil {
			return v, nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to int64", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to int64", i, i)
	}
}

// ToInt32E casts an interface to an int32 type.
func ToInt32E(i interface{}) (int32, error) {
	i = indirect(i)

	switch s := i.(type) {
	case int:
		return int32(s), nil
	case int64:
		return int32(s), nil
	case int32:
		return s, nil
	case int16:
		return int32(s), nil
	case int8:
		return int32(s), nil
	case uint:
		return int32(s), nil
	case uint64:
		return int32(s), nil
	case uint32:
		return int32(s), nil
	case uint16:
		return int32(s), nil
	case uint8:
		return int32(s), nil
	case float64:
		return int32(s), nil
	case float32:
		return int32(s), nil
	case string:
		v, err := strconv.ParseInt(s, 0, 0)
		if err == nil {
			return int32(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to int32", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to int32", i, i)
	}
}

// ToInt16E casts an interface to an int16 type.
func ToInt16E(i interface{}) (int16, error) {
	i = indirect(i)

	switch s := i.(type) {
	case int:
		return int16(s), nil
	case int64:
		return int16(s), nil
	case int32:
		return int16(s), nil
	case int16:
		return s, nil
	case int8:
		return int16(s), nil
	case uint:
		return int16(s), nil
	case uint64:
		return int16(s), nil
	case uint32:
		return int16(s), nil
	case uint16:
		return int16(s), nil
	case uint8:
		return int16(s), nil
	case float64:
		return int16(s), nil
	case float32:
		return int16(s), nil
	case string:
		v, err := strconv.ParseInt(s, 0, 0)
		if err == nil {
			return int16(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to int16", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to int16", i, i)
	}
}

// ToInt8E casts an interface to an int8 type.
func ToInt8E(i interface{}) (int8, error) {
	i = indirect(i)

	switch s := i.(type) {
	case int:
		return int8(s), nil
	case int64:
		return int8(s), nil
	case int32:
		return int8(s), nil
	case int16:
		return int8(s), nil
	case int8:
		return s, nil
	case uint:
		return int8(s), nil
	case uint64:
		return int8(s), nil
	case uint32:
		return int8(s), nil
	case uint16:
		return int8(s), nil
	case uint8:
		return int8(s), nil
	case float64:
		return int8(s), nil
	case float32:
		return int8(s), nil
	case string:
		v, err := strconv.ParseInt(s, 0, 0)
		if err == nil {
			return int8(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to int8", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to int8", i, i)
	}
}

// ToIntE casts an interface to an int type.
func ToIntE(i interface{}) (int, error) {
	i = indirect(i)

	switch s := i.(type) {
	case int:
		return s, nil
	case int64:
		return int(s), nil
	case int32:
		return int(s), nil
	case int16:
		return int(s), nil
	case int8:
		return int(s), nil
	case uint:
		return int(s), nil
	case uint64:
		return int(s), nil
	case uint32:
		return int(s), nil
	case uint16:
		return int(s), nil
	case uint8:
		return int(s), nil
	case float64:
		return int(s), nil
	case float32:
		return int(s), nil
	case string:
		v, err := strconv.ParseInt(s, 0, 0)
		if err == nil {
			return int(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v of type %T to int", i, i)
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to int", i, i)
	}
}

// ToUintE casts an interface to a uint type.
func ToUintE(i interface{}) (uint, error) {
	i = indirect(i)

	switch s := i.(type) {
	case string:
		v, err := strconv.ParseUint(s, 0, 0)
		if err == nil {
			return uint(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v to uint: %s", i, err)
	case int:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case int64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case int32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case int16:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case int8:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case uint:
		return s, nil
	case uint64:
		return uint(s), nil
	case uint32:
		return uint(s), nil
	case uint16:
		return uint(s), nil
	case uint8:
		return uint(s), nil
	case float64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case float32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint(s), nil
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to uint", i, i)
	}
}

// ToUint64E casts an interface to a uint64 type.
func ToUint64E(i interface{}) (uint64, error) {
	i = indirect(i)

	switch s := i.(type) {
	case string:
		v, err := strconv.ParseUint(s, 0, 64)
		if err == nil {
			return v, nil
		}
		return 0, fmt.Errorf("unable to cast %#v to uint64: %s", i, err)
	case int:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case int64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case int32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case int16:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case int8:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case uint:
		return uint64(s), nil
	case uint64:
		return s, nil
	case uint32:
		return uint64(s), nil
	case uint16:
		return uint64(s), nil
	case uint8:
		return uint64(s), nil
	case float32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case float64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint64(s), nil
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to uint64", i, i)
	}
}

// ToUint32E casts an interface to a uint32 type.
func ToUint32E(i interface{}) (uint32, error) {
	i = indirect(i)

	switch s := i.(type) {
	case string:
		v, err := strconv.ParseUint(s, 0, 32)
		if err == nil {
			return uint32(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v to uint32: %s", i, err)
	case int:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case int64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case int32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case int16:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case int8:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case uint:
		return uint32(s), nil
	case uint64:
		return uint32(s), nil
	case uint32:
		return s, nil
	case uint16:
		return uint32(s), nil
	case uint8:
		return uint32(s), nil
	case float64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case float32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint32(s), nil
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to uint32", i, i)
	}
}

// ToUint16E casts an interface to a uint16 type.
func ToUint16E(i interface{}) (uint16, error) {
	i = indirect(i)

	switch s := i.(type) {
	case string:
		v, err := strconv.ParseUint(s, 0, 16)
		if err == nil {
			return uint16(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v to uint16: %s", i, err)
	case int:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case int64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case int32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case int16:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case int8:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case uint:
		return uint16(s), nil
	case uint64:
		return uint16(s), nil
	case uint32:
		return uint16(s), nil
	case uint16:
		return s, nil
	case uint8:
		return uint16(s), nil
	case float64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case float32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint16(s), nil
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to uint16", i, i)
	}
}

// ToUint8E casts an interface to a uint type.
func ToUint8E(i interface{}) (uint8, error) {
	i = indirect(i)

	switch s := i.(type) {
	case string:
		v, err := strconv.ParseUint(s, 0, 8)
		if err == nil {
			return uint8(v), nil
		}
		return 0, fmt.Errorf("unable to cast %#v to uint8: %s", i, err)
	case int:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case int64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case int32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case int16:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case int8:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case uint:
		return uint8(s), nil
	case uint64:
		return uint8(s), nil
	case uint32:
		return uint8(s), nil
	case uint16:
		return uint8(s), nil
	case uint8:
		return s, nil
	case float64:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case float32:
		if s < 0 {
			return 0, errNegativeNotAllowed
		}
		return uint8(s), nil
	case bool:
		if s {
			return 1, nil
		}
		return 0, nil
	case nil:
		return 0, nil
	default:
		return 0, fmt.Errorf("unable to cast %#v of type %T to uint8", i, i)
	}
}

// From html/template/content.go
// Copyright 2011 The Go Authors. All rights reserved.
// indirect returns the value, after dereferencing as many times
// as necessary to reach the base type (or nil).
func indirect(a interface{}) interface{} {
	if a == nil {
		return nil
	}
	if t := reflect.TypeOf(a); t.Kind() != reflect.Ptr {
		// Avoid creating a reflect.Value if it's not a pointer.
		return a
	}
	v := reflect.ValueOf(a)
	for v.Kind() == reflect.Ptr && !v.IsNil() {
		v = v.Elem()
	}
	return v.Interface()
}

// From html/template/content.go
// Copyright 2011 The Go Authors. All rights reserved.
// indirectToStringerOrError returns the value, after dereferencing as many times
// as necessary to reach the base type (or nil) or an implementation of fmt.Stringer
// or error,
func indirectToStringerOrError(a interface{}) interface{} {
	if a == nil {
		return nil
	}

	var errorType = reflect.TypeOf((*error)(nil)).Elem()
	var fmtStringerType = reflect.TypeOf((*fmt.Stringer)(nil)).Elem()

	v := reflect.ValueOf(a)
	for !v.Type().Implements(fmtStringerType) && !v.Type().Implements(errorType) && v.Kind() == reflect.Ptr && !v.IsNil() {
		v = v.Elem()
	}
	return v.Interface()
}

// ToStringE casts an interface to a string type.
func ToStringE(i interface{}) (string, error) {
	i = indirectToStringerOrError(i)

	switch s := i.(type) {
	case string:
		return s, nil
	case bool:
		return strconv.FormatBool(s), nil
	case float64:
		return strconv.FormatFloat(s, 'f', -1, 64), nil
	case float32:
		return strconv.FormatFloat(float64(s), 'f', -1, 32), nil
	case int:
		return strconv.Itoa(s), nil
	case int64:
		return strconv.FormatInt(s, 10), nil
	case int32:
		return strconv.Itoa(int(s)), nil
	case int16:
		return strconv.FormatInt(int64(s), 10), nil
	case int8:
		return strconv.FormatInt(int64(s), 10), nil
	case uint:
		return strconv.FormatUint(uint64(s), 10), nil
	case uint64:
		return strconv.FormatUint(uint64(s), 10), nil
	case uint32:
		return strconv.FormatUint(uint64(s), 10), nil
	case uint16:
		return strconv.FormatUint(uint64(s), 10), nil
	case uint8:
		return strconv.FormatUint(uint64(s), 10), nil
	case []byte:
		return string(s), nil
	case template.HTML:
		return string(s), nil
	case template.URL:
		return string(s), nil
	case template.JS:
		return string(s), nil
	case template.CSS:
		return string(s), nil
	case template.HTMLAttr:
		return string(s), nil
	case nil:
		return "", nil
	case fmt.Stringer:
		return s.String(), nil
	case error:
		return s.Error(), nil
	default:
		return "", fmt.Errorf("unable to cast %#v of type %T to string", i, i)
	}
}

// ToStringMapStringE casts an interface to a map[string]string type.
func ToStringMapStringE(i interface{}) (map[string]string, error) {
	var m = map[string]string{}

	switch v := i.(type) {
	case map[string]string:
		return v, nil
	case map[string]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToString(val)
		}
		return m, nil
	case map[interface{}]string:
		for k, val := range v {
			m[ToString(k)] = ToString(val)
		}
		return m, nil
	case map[interface{}]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToString(val)
		}
		return m, nil
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	default:
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]string", i, i)
	}
}

// ToStringMapStringSliceE casts an interface to a map[string][]string type.
func ToStringMapStringSliceE(i interface{}) (map[string][]string, error) {
	var m = map[string][]string{}

	switch v := i.(type) {
	case map[string][]string:
		return v, nil
	case map[string][]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[string]string:
		for k, val := range v {
			m[ToString(k)] = []string{val}
		}
	case map[string]interface{}:
		for k, val := range v {
			switch vt := val.(type) {
			case []interface{}:
				m[ToString(k)] = ToStringSlice(vt)
			case []string:
				m[ToString(k)] = vt
			default:
				m[ToString(k)] = []string{ToString(val)}
			}
		}
		return m, nil
	case map[interface{}][]string:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[interface{}]string:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[interface{}][]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[interface{}]interface{}:
		for k, val := range v {
			key, err := ToStringE(k)
			if err != nil {
				return m, fmt.Errorf("unable to cast %#v of type %T to map[string][]string", i, i)
			}
			value, err := ToStringSliceE(val)
			if err != nil {
				return m, fmt.Errorf("unable to cast %#v of type %T to map[string][]string", i, i)
			}
			m[key] = value
		}
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	default:
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string][]string", i, i)
	}
	return m, nil
}

// ToStringMapBoolE casts an interface to a map[string]bool type.
func ToStringMapBoolE(i interface{}) (map[string]bool, error) {
	var m = map[string]bool{}

	switch v := i.(type) {
	case map[interface{}]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToBool(val)
		}
		return m, nil
	case map[string]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToBool(val)
		}
		return m, nil
	case map[string]bool:
		return v, nil
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	default:
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]bool", i, i)
	}
}

// ToStringMapE casts an interface to a map[string]interface{} type.
func ToStringMapE(i interface{}) (map[string]interface{}, error) {
	var m = map[string]interface{}{}

	switch v := i.(type) {
	case map[interface{}]interface{}:
		for k, val := range v {
			m[ToString(k)] = val
		}
		return m, nil
	case map[string]interface{}:
		return v, nil
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	default:
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]interface{}", i, i)
	}
}

// ToStringMapIntE casts an interface to a map[string]int{} type.
func ToStringMapIntE(i interface{}) (map[string]int, error) {
	var m = map[string]int{}
	if i == nil {
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]int", i, i)
	}

	switch v := i.(type) {
	case map[interface{}]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToInt(val)
		}
		return m, nil
	case map[string]interface{}:
		for k, val := range v {
			m[k] = ToInt(val)
		}
		return m, nil
	case map[string]int:
		return v, nil
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	}

	if reflect.TypeOf(i).Kind() != reflect.Map {
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]int", i, i)
	}

	mVal := reflect.ValueOf(m)
	v := reflect.ValueOf(i)
	for _, keyVal := range v.MapKeys() {
		val, err := ToIntE(v.MapIndex(keyVal).Interface())
		if err != nil {
			return m, fmt.Errorf("unable to cast %#v of type %T to map[string]int", i, i)
		}
		mVal.SetMapIndex(keyVal, reflect.ValueOf(val))
	}
	return m, nil
}

// ToStringMapInt64E casts an interface to a map[string]int64{} type.
func ToStringMapInt64E(i interface{}) (map[string]int64, error) {
	var m = map[string]int64{}
	if i == nil {
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]int64", i, i)
	}

	switch v := i.(type) {
	case map[interface{}]interface{}:
		for k, val := range v {
			m[ToString(k)] = ToInt64(val)
		}
		return m, nil
	case map[string]interface{}:
		for k, val := range v {
			m[k] = ToInt64(val)
		}
		return m, nil
	case map[string]int64:
		return v, nil
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	}

	if reflect.TypeOf(i).Kind() != reflect.Map {
		return m, fmt.Errorf("unable to cast %#v of type %T to map[string]int64", i, i)
	}
	mVal := reflect.ValueOf(m)
	v := reflect.ValueOf(i)
	for _, keyVal := range v.MapKeys() {
		val, err := ToInt64E(v.MapIndex(keyVal).Interface())
		if err != nil {
			return m, fmt.Errorf("unable to cast %#v of type %T to map[string]int64", i, i)
		}
		mVal.SetMapIndex(keyVal, reflect.ValueOf(val))
	}
	return m, nil
}

// ToSliceE casts an interface to a []interface{} type.
func ToSliceE(i interface{}) ([]interface{}, error) {
	var s []interface{}

	switch v := i.(type) {
	case []interface{}:
		return append(s, v...), nil
	case []map[string]interface{}:
		for _, u := range v {
			s = append(s, u)
		}
		return s, nil
	default:
		return s, fmt.Errorf("unable to cast %#v of type %T to []interface{}", i, i)
	}
}

// ToBoolSliceE casts an interface to a []bool type.
func ToBoolSliceE(i interface{}) ([]bool, error) {
	if i == nil {
		return []bool{}, fmt.Errorf("unable to cast %#v of type %T to []bool", i, i)
	}

	switch v := i.(type) {
	case []bool:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]bool, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := ToBoolE(s.Index(j).Interface())
			if err != nil {
				return []bool{}, fmt.Errorf("unable to cast %#v of type %T to []bool", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []bool{}, fmt.Errorf("unable to cast %#v of type %T to []bool", i, i)
	}
}

// ToStringSliceE casts an interface to a []string type.
func ToStringSliceE(i interface{}) ([]string, error) {
	var a []string

	switch v := i.(type) {
	case []interface{}:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case []string:
		return v, nil
	case []int8:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case []int:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case []int32:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case []int64:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case []float32:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case []float64:
		for _, u := range v {
			a = append(a, ToString(u))
		}
		return a, nil
	case string:
		return strings.Fields(v), nil
	case []error:
		for _, err := range i.([]error) {
			a = append(a, err.Error())
		}
		return a, nil
	case interface{}:
		str, err := ToStringE(v)
		if err != nil {
			return a, fmt.Errorf("unable to cast %#v of type %T to []string", i, i)
		}
		return []string{str}, nil
	default:
		return a, fmt.Errorf("unable to cast %#v of type %T to []string", i, i)
	}
}

// ToIntSliceE casts an interface to a []int type.
func ToIntSliceE(i interface{}) ([]int, error) {
	if i == nil {
		return []int{}, fmt.Errorf("unable to cast %#v of type %T to []int", i, i)
	}

	switch v := i.(type) {
	case []int:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]int, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := ToIntE(s.Index(j).Interface())
			if err != nil {
				return []int{}, fmt.Errorf("unable to cast %#v of type %T to []int", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []int{}, fmt.Errorf("unable to cast %#v of type %T to []int", i, i)
	}
}

// ToDurationSliceE casts an interface to a []time.Duration type.
func ToDurationSliceE(i interface{}) ([]time.Duration, error) {
	if i == nil {
		return []time.Duration{}, fmt.Errorf("unable to cast %#v of type %T to []time.Duration", i, i)
	}

	switch v := i.(type) {
	case []time.Duration:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]time.Duration, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := ToDurationE(s.Index(j).Interface())
			if err != nil {
				return []time.Duration{}, fmt.Errorf("unable to cast %#v of type %T to []time.Duration", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []time.Duration{}, fmt.Errorf("unable to cast %#v of type %T to []time.Duration", i, i)
	}
}

// StringToDate attempts to parse a string into a time.Time type using a
// predefined list of formats.  If no suitable format is found, an error is
// returned.
func StringToDate(s string) (time.Time, error) {
	return parseDateWith(s, time.UTC, timeFormats)
}

// StringToDateInDefaultLocation casts an empty interface to a time.Time,
// interpreting inputs without a timezone to be in the given location,
// or the local timezone if nil.
func StringToDateInDefaultLocation(s string, location *time.Location) (time.Time, error) {
	return parseDateWith(s, location, timeFormats)
}

type timeFormatType int

const (
	timeFormatNoTimezone timeFormatType = iota
	timeFormatNamedTimezone
	timeFormatNumericTimezone
	timeFormatNumericAndNamedTimezone
	timeFormatTimeOnly
)

type timeFormat struct {
	format string
	typ    timeFormatType
}

func (f timeFormat) hasTimezone() bool {
	// We don't include the formats with only named timezones, see
	// https://github.com/golang/go/issues/19694#issuecomment-289103522
	return f.typ >= timeFormatNumericTimezone && f.typ <= timeFormatNumericAndNamedTimezone
}

var (
	timeFormats = []timeFormat{
		timeFormat{time.RFC3339, timeFormatNumericTimezone},
		timeFormat{"2006-01-02T15:04:05", timeFormatNoTimezone}, // iso8601 without timezone
		timeFormat{time.RFC1123Z, timeFormatNumericTimezone},
		timeFormat{time.RFC1123, timeFormatNamedTimezone},
		timeFormat{time.RFC822Z, timeFormatNumericTimezone},
		timeFormat{time.RFC822, timeFormatNamedTimezone},
		timeFormat{time.RFC850, timeFormatNamedTimezone},
		timeFormat{"2006-01-02 15:04:05.999999999 -0700 MST", timeFormatNumericAndNamedTimezone}, // Time.String()
		timeFormat{"2006-01-02T15:04:05-0700", timeFormatNumericTimezone},                        // RFC3339 without timezone hh:mm colon
		timeFormat{"2006-01-02 15:04:05Z0700", timeFormatNumericTimezone},                        // RFC3339 without T or timezone hh:mm colon
		timeFormat{"2006-01-02 15:04:05", timeFormatNoTimezone},
		timeFormat{time.ANSIC, timeFormatNoTimezone},
		timeFormat{time.UnixDate, timeFormatNamedTimezone},
		timeFormat{time.RubyDate, timeFormatNumericTimezone},
		timeFormat{"2006-01-02 15:04:05Z07:00", timeFormatNumericTimezone},
		timeFormat{"2006-01-02", timeFormatNoTimezone},
		timeFormat{"02 Jan 2006", timeFormatNoTimezone},
		timeFormat{"2006-01-02 15:04:05 -07:00", timeFormatNumericTimezone},
		timeFormat{"2006-01-02 15:04:05 -0700", timeFormatNumericTimezone},
		timeFormat{time.Kitchen, timeFormatTimeOnly},
		timeFormat{time.Stamp, timeFormatTimeOnly},
		timeFormat{time.StampMilli, timeFormatTimeOnly},
		timeFormat{time.StampMicro, timeFormatTimeOnly},
		timeFormat{time.StampNano, timeFormatTimeOnly},
	}
)

func parseDateWith(s string, location *time.Location, formats []timeFormat) (d time.Time, e error) {

	for _, format := range formats {
		if d, e = time.Parse(format.format, s); e == nil {

			// Some time formats have a zone name, but no offset, so it gets
			// put in that zone name (not the default one passed in to us), but
			// without that zone's offset. So set the location manually.
			if format.typ <= timeFormatNamedTimezone {
				if location == nil {
					location = time.Local
				}
				year, month, day := d.Date()
				hour, min, sec := d.Clock()
				d = time.Date(year, month, day, hour, min, sec, d.Nanosecond(), location)
			}

			return
		}
	}
	return d, fmt.Errorf("unable to parse date: %s", s)
}

// jsonStringToObject attempts to unmarshall a string as JSON into
// the object passed as pointer.
func jsonStringToObject(s string, v interface{}) error {
	data := []byte(s)
	return json.Unmarshal(data, v)
}
