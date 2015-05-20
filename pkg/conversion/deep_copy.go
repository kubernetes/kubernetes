/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package conversion

import (
	"fmt"
	"reflect"
)

// DeepCopy makes a deep copy of source or returns an error.
func DeepCopy(source interface{}) (interface{}, error) {
	v, err := deepCopy(reflect.ValueOf(source))
	return v.Interface(), err
}

func deepCopy(src reflect.Value) (reflect.Value, error) {
	switch src.Kind() {
	case reflect.Chan, reflect.Func, reflect.UnsafePointer, reflect.Uintptr:
		return src, fmt.Errorf("cannot deep copy kind: %s", src.Kind())

	case reflect.Array:
		dst := reflect.New(src.Type())
		for i := 0; i < src.Len(); i++ {
			copyVal, err := deepCopy(src.Index(i))
			if err != nil {
				return src, err
			}
			dst.Elem().Index(i).Set(copyVal)
		}
		return dst.Elem(), nil

	case reflect.Interface:
		if src.IsNil() {
			return src, nil
		}
		return deepCopy(src.Elem())

	case reflect.Map:
		if src.IsNil() {
			return src, nil
		}
		dst := reflect.MakeMap(src.Type())
		for _, k := range src.MapKeys() {
			copyVal, err := deepCopy(src.MapIndex(k))
			if err != nil {
				return src, err
			}
			dst.SetMapIndex(k, copyVal)
		}
		return dst, nil

	case reflect.Ptr:
		if src.IsNil() {
			return src, nil
		}
		dst := reflect.New(src.Type().Elem())
		copyVal, err := deepCopy(src.Elem())
		if err != nil {
			return src, err
		}
		dst.Elem().Set(copyVal)
		return dst, nil

	case reflect.Slice:
		if src.IsNil() {
			return src, nil
		}
		dst := reflect.MakeSlice(src.Type(), 0, src.Len())
		for i := 0; i < src.Len(); i++ {
			copyVal, err := deepCopy(src.Index(i))
			if err != nil {
				return src, err
			}
			dst = reflect.Append(dst, copyVal)
		}
		return dst, nil

	case reflect.Struct:
		dst := reflect.New(src.Type())
		for i := 0; i < src.NumField(); i++ {
			if !dst.Elem().Field(i).CanSet() {
				// Can't set private fields. At this point, the
				// best we can do is a shallow copy. For
				// example, time.Time is a value type with
				// private members that can be shallow copied.
				return src, nil
			}
			copyVal, err := deepCopy(src.Field(i))
			if err != nil {
				return src, err
			}
			dst.Elem().Field(i).Set(copyVal)
		}
		return dst.Elem(), nil

	default:
		// Value types like numbers, booleans, and strings.
		return src, nil
	}
}
