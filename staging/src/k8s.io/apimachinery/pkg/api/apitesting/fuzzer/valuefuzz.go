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

package fuzzer

import (
	"reflect"
)

// ValueFuzz recursively changes all basic type values in an object. Any kind of references will not
// be touch, i.e. the addresses of slices, maps, pointers will stay unchanged.
func ValueFuzz(obj interface{}) {
	valueFuzz(reflect.ValueOf(obj))
}

func valueFuzz(obj reflect.Value) {
	switch obj.Kind() {
	case reflect.Array:
		for i := 0; i < obj.Len(); i++ {
			valueFuzz(obj.Index(i))
		}
	case reflect.Slice:
		if obj.IsNil() {
			// TODO: set non-nil value
		} else {
			for i := 0; i < obj.Len(); i++ {
				valueFuzz(obj.Index(i))
			}
		}
	case reflect.Interface, reflect.Pointer:
		if obj.IsNil() {
			// TODO: set non-nil value
		} else {
			valueFuzz(obj.Elem())
		}
	case reflect.Struct:
		for i, n := 0, obj.NumField(); i < n; i++ {
			valueFuzz(obj.Field(i))
		}
	case reflect.Map:
		if obj.IsNil() {
			// TODO: set non-nil value
		} else {
			for _, k := range obj.MapKeys() {
				// map values are not addressable. We need a copy.
				v := obj.MapIndex(k)
				copy := reflect.New(v.Type())
				copy.Elem().Set(v)
				valueFuzz(copy.Elem())
				obj.SetMapIndex(k, copy.Elem())
			}
			// TODO: set some new value
		}
	case reflect.Func: // ignore, we don't have function types in our API
	default:
		if !obj.CanSet() {
			return
		}
		switch obj.Kind() {
		case reflect.String:
			obj.SetString(obj.String() + "x")
		case reflect.Bool:
			obj.SetBool(!obj.Bool())
		case reflect.Float32, reflect.Float64:
			obj.SetFloat(obj.Float()*2.0 + 1.0)
		case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Int:
			obj.SetInt(obj.Int() + 1)
		case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uint:
			obj.SetUint(obj.Uint() + 1)
		default:
		}
	}
}
