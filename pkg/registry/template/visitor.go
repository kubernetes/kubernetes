/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package template

import (
	"reflect"

	"github.com/golang/glog"
)

// VisitObjectStrings visits recursively all string fields in the object and call the
// visitor function on them. The visitor function can be used to modify the
// value of the string fields.
func VisitObjectStrings(obj interface{}, visitor func(string) string) {
	visitValue(reflect.ValueOf(obj), visitor)
}

func visitValue(v reflect.Value, visitor func(string) string) {
	// you'll never be able to substitute on a nil.  Check the kind first or you'll accidentally
	// end up panic-ing
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		if v.IsNil() {
			return
		}
	}
	if !v.IsValid() {
		// Ingore.  Don't Visit Empty values.
		return
	}

	switch v.Kind() {

	case reflect.Ptr:
		visitValue(v.Elem(), visitor)

	case reflect.Interface:
		visitValue(reflect.ValueOf(v.Interface()), visitor)

	case reflect.Slice, reflect.Array:
		vt := v.Type().Elem()
		for i := 0; i < v.Len(); i++ {
			val := visitUnsettableValues(vt, v.Index(i), visitor)
			v.Index(i).Set(val)
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			visitValue(v.Field(i), visitor)
		}

	case reflect.Map:
		vt := v.Type().Elem()
		for _, k := range v.MapKeys() {
			val := visitUnsettableValues(vt, v.MapIndex(k), visitor)
			v.SetMapIndex(k, val)
		}

	case reflect.String:
		if !v.CanSet() {
			glog.Errorf("Unable to set String value '%v'", v)
			return
		}
		v.SetString(visitor(v.String()))

	case reflect.Bool:
		// Ignore.  Only Visit Strings and Recursable types

	case reflect.Int, reflect.Uint,
		reflect.Int64, reflect.Uint64,
		reflect.Int32, reflect.Uint32,
		reflect.Int16, reflect.Uint16,
		reflect.Int8, reflect.Uint8:
		// Ignore.  Only Visit Strings and Recursable types

	default:
		glog.Errorf("Unknown field type '%s': %v", v.Kind(), v)
	}
}

// visitUnsettableValues creates a copy of the object you want to modify and returns the modified result
func visitUnsettableValues(typeOf reflect.Type, original reflect.Value, visitor func(string) string) reflect.Value {
	val := reflect.New(typeOf).Elem()
	existing := original
	// if the value type is interface, we must resolve it to a concrete value prior to setting it back.
	if existing.CanInterface() {
		existing = reflect.ValueOf(existing.Interface())
	}
	switch existing.Kind() {
	case reflect.String:
		s := visitor(existing.String())
		val.Set(reflect.ValueOf(s))
	default:
		if existing.IsValid() && existing.Kind() != reflect.Invalid {
			val.Set(existing)
		}
		visitValue(val, visitor)
	}

	return val
}
