/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package jsonlike

import (
	"encoding/json"
	"fmt"
)

// Object is a map[string]interface{} that contains only []interface{},
// map[string]interface{}, string, number, boolean, or nil values.
type Object map[string]interface{}

// FieldCopy returns a deep copy of the value of a nested field.
// false is returned if the value is missing.
// nil, true is returned for a nil field.
func (obj Object) FieldCopy(fields ...string) (interface{}, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return nil, false
	}
	return DeepCopyValue(val), true
}

// Field returns a nested field or false if the field does not exist
// or if any level in fields is not a map.
func (obj Object) Field(fields ...string) (interface{}, bool) {
	var val interface{} = map[string]interface{}(obj)
	for _, field := range fields {
		if m, ok := val.(map[string]interface{}); ok {
			val, ok = m[field]
			if !ok {
				return nil, false
			}
		} else {
			// Expected map[string]interface{}, got something else
			return nil, false
		}
	}
	return val, true
}

// String returns the string value of a nested field.
// Returns false if value is not found or is not a string.
func (obj Object) String(fields ...string) (string, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return "", false
	}
	s, ok := val.(string)
	return s, ok
}

// Bool returns the bool value of a nested field.
// Returns false if value is not found or is not a bool.
func (obj Object) Bool(fields ...string) (bool, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return false, false
	}
	b, ok := val.(bool)
	return b, ok
}

// Float64 returns the bool value of a nested field.
// Returns false if value is not found or is not a float64.
func (obj Object) Float64(fields ...string) (float64, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return 0, false
	}
	f, ok := val.(float64)
	return f, ok
}

// Int64 returns the int64 value of a nested field.
// Returns false if value is not found or is not an int64.
func (obj Object) Int64(fields ...string) (int64, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return 0, false
	}
	i, ok := val.(int64)
	return i, ok
}

// StringSlice returns a copy of []string value of a nested field.
// Returns false if value is not found, is not a []interface{} or contains non-string items in the slice.
func (obj Object) StringSlice(fields ...string) ([]string, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return nil, false
	}
	if m, ok := val.([]interface{}); ok {
		strSlice := make([]string, 0, len(m))
		for _, v := range m {
			if str, ok := v.(string); ok {
				strSlice = append(strSlice, str)
			} else {
				return nil, false
			}
		}
		return strSlice, true
	}
	return nil, false
}

// Slice returns a deep copy of []interface{} value of a nested field.
// Returns false if value is not found or is not a []interface{}.
func (obj Object) Slice(fields ...string) ([]interface{}, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return nil, false
	}
	if _, ok := val.([]interface{}); ok {
		return DeepCopyValue(val).([]interface{}), true
	}
	return nil, false
}

// StringMap returns a copy of map[string]string value of a nested field.
// Returns false if value is not found, is not a map[string]interface{} or contains non-string values in the map.
func (obj Object) StringMap(fields ...string) (map[string]string, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return nil, false
	}
	if m, ok := val.(map[string]interface{}); ok {
		strMap := make(map[string]string, len(m))
		for k, v := range m {
			if str, ok := v.(string); ok {
				strMap[k] = str
			} else {
				return nil, false
			}
		}
		return strMap, true
	}
	return nil, false
}

// Map returns a deep copy of map[string]interface{} value of a nested field.
// Returns false if value is not found or is not a map[string]interface{}.
func (obj Object) Map(fields ...string) (map[string]interface{}, bool) {
	val, ok := obj.Field(fields...)
	if !ok {
		return nil, false
	}
	if m, ok := val.(map[string]interface{}); ok {
		return Object(m).DeepCopy(), true
	}
	return nil, false
}

// SetField sets the value of a nested field to a deep copy of the value provided.
// Returns false if value cannot be set because one of the nesting levels is not a map[string]interface{}.
func (obj Object) SetField(value interface{}, fields ...string) bool {
	return obj.setFieldNoCopy(DeepCopyValue(value), fields...)
}

func (obj Object) setFieldNoCopy(value interface{}, fields ...string) bool {
	m := obj
	for _, field := range fields[:len(fields)-1] {
		if val, ok := m[field]; ok {
			if valMap, ok := val.(map[string]interface{}); ok {
				m = valMap
			} else {
				return false
			}
		} else {
			newVal := make(map[string]interface{})
			m[field] = newVal
			m = newVal
		}
	}
	m[fields[len(fields)-1]] = value
	return true
}

// SetStringSlice sets the string slice value of a nested field.
// Returns false if value cannot be set because one of the nesting levels is not a map[string]interface{}.
func (obj Object) SetStringSlice(value []string, fields ...string) bool {
	m := make([]interface{}, 0, len(value)) // convert []string into []interface{}
	for _, v := range value {
		m = append(m, v)
	}
	return obj.setFieldNoCopy(m, fields...)
}

// SetSlice sets the slice value of a nested field.
// Returns false if value cannot be set because one of the nesting levels is not a map[string]interface{}.
func (obj Object) SetSlice(value []interface{}, fields ...string) bool {
	return obj.SetField(value, fields...)
}

// SetStringMap sets the map[string]string value of a nested field.
// Returns false if value cannot be set because one of the nesting levels is not a map[string]interface{}.
func (obj Object) SetStringMap(value map[string]string, fields ...string) bool {
	m := make(map[string]interface{}, len(value)) // convert map[string]string into map[string]interface{}
	for k, v := range value {
		m[k] = v
	}
	return obj.setFieldNoCopy(m, fields...)
}

// SetMap sets the map[string]interface{} value of a nested field.
// Returns false if value cannot be set because one of the nesting levels is not a map[string]interface{}.
func (obj Object) SetMap(value map[string]interface{}, fields ...string) bool {
	return obj.SetField(value, fields...)
}

// RemoveField removes the nested field from the obj.
func (obj Object) RemoveField(fields ...string) {
	if obj == nil {
		return
	}
	m := obj
	for _, field := range fields[:len(fields)-1] {
		if x, ok := m[field].(map[string]interface{}); ok {
			m = x
		} else {
			return
		}
	}
	delete(m, fields[len(fields)-1])
}

func (obj Object) GetString(fields ...string) string {
	val, ok := obj.String(fields...)
	if !ok {
		return ""
	}
	return val
}

// DeepCopy deep copies the object, assuming it is a valid JSON representation i.e. only contains
// types produced by json.Unmarshal().
func (obj Object) DeepCopy() map[string]interface{} {
	return DeepCopyValue(obj).(map[string]interface{})
}

// DeepCopyValue deep copies the passed value, assuming it is a valid JSON representation i.e. only contains
// types produced by json.Unmarshal().
func DeepCopyValue(x interface{}) interface{} {
	switch x := x.(type) {
	case Object:
		return DeepCopyValue(map[string]interface{}(x))
	case map[string]interface{}:
		clone := make(map[string]interface{}, len(x))
		for k, v := range x {
			clone[k] = DeepCopyValue(v)
		}
		return clone
	case []interface{}:
		clone := make([]interface{}, len(x))
		for i, v := range x {
			clone[i] = DeepCopyValue(v)
		}
		return clone
	case string, int64, bool, float64, nil, json.Number:
		return x
	default:
		panic(fmt.Errorf("cannot deep copy %T", x))
	}
}
